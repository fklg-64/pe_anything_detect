import os
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from core.torch.feature_extractor import TorchCLIPFeatureExtractor
from core.torch.edgetam_runner import EdgeTamRunner
from core.torch.analysis_function import export_complete_vit_analysis_to_excel
from core.utils import *

from scipy.ndimage import distance_transform_edt

class SegmentationRunner:
    """抽出された特徴量・クラスタリングを用いた推論・マスク生成の実行クラス"""

    def __init__(
        self, 
        extractor: TorchCLIPFeatureExtractor, 
        edgetam: EdgeTamRunner, 
        config: yaml
    ):
        # モデル
        self.extractor = extractor
        self.edgetam = edgetam

        # 設定値の初期化
        self.init_value(config)

        self.register_fgbg_text_features()

    def init_value(self, config: yaml):
        # config関係
        self.prompts_dir = config["prompts_dir"]
        self.k_clusters = config["k_clusters"]
        self.niter = config["n_iter"]
        self.foreground_threshold = config["foreground_threshold"]
        self.device = config["device"]
        self.output_dir = config["output_dir"]
        self.color_palette = config["color_palette"]

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size

        # プロンプトとプロンプトのテキスト特徴量
        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: Optional[torch.Tensor] = None

    # =========================================================================
    # 1. テキストプロンプト & 特徴量準備
    # =========================================================================

    def load_prompts(self, path: str) -> List[str]:
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data["prompts"]

    def common_process(self, prompts: List[str]) -> np.ndarray:
        text_tokens = self.extractor.tokenizer(
            prompts, context_length=self.extractor.model.context_length
        )
        text_features = self.extractor.extract_text_features(text_tokens)
        return text_features

    def register_fgbg_text_features(self) -> None:
        foreground_prompts_path = Path(self.prompts_dir) / "foreground.json"
        background_prompts_path = Path(self.prompts_dir) / "background.json"

        self.foreground_prompts = self.load_prompts(foreground_prompts_path)
        self.background_prompts = self.load_prompts(background_prompts_path)

        fg_feats = self.common_process(self.foreground_prompts)
        bg_feats = self.common_process(self.background_prompts)

        self.fgbg_text_features = torch.vstack([fg_feats, bg_feats])

    # =========================================================================
    # 2. クラスタリング & 可視化ユーティリティ
    # =========================================================================

    def fit_predict_kmeans(
        self, 
        features: np.ndarray, 
        k: int = 12, 
        niter: int = 10
    ) -> np.ndarray:
        N, D = features.shape
        features_contig = (
            np.ascontiguousarray(features, dtype=np.float32)
            if not features.flags["C_CONTIGUOUS"]
            else features
        )
        faiss.omp_set_num_threads(2)
        kmeans = faiss.Kmeans(
            d=D, k=k, niter=niter, spherical=False, verbose=False, seed=42
        )
        kmeans.train(features_contig)
        _, cluster_labels = kmeans.index.search(features_contig, 1)
        return cluster_labels.reshape(-1)

    def colorize_clusters_PIL(self, cluster_labels: np.ndarray) -> Image.Image:
        color_map = Image.fromarray(cluster_labels.astype(np.uint8))
        color_map.putpalette(self.color_palette)
        return color_map

    # =========================================================================
    # 3. プロンプトポイント選出 & 座標変換 (ロジック不変)
    # =========================================================================

    def select_n_prompt_points(
        self, 
        target_mask: np.ndarray, 
        opponent_mask: np.ndarray, 
        num_points: int = 1
    ) -> np.ndarray:
        """
        対象領域（target_mask）の中から、条件を満たす点を順に num_points 個抽出する:
        1. 領域の内側に位置する（境界から遠い）
        2. 対立領域（opponent_mask）から遠い
        3. 既に選ばれている他のプロンプト点から遠い
        """
        if not np.any(target_mask) or num_points <= 0:
            return np.empty((0, 2), dtype=np.int32)

        dist_target = distance_transform_edt(target_mask)
        dist_opponent = distance_transform_edt(1 - opponent_mask)

        norm_target = dist_target / (np.max(dist_target) + 1e-8)
        norm_opponent = dist_opponent / (np.max(dist_opponent) + 1e-8)

        base_score_map = norm_target * norm_opponent
        base_score_map[target_mask == 0] = -1.0  # 対象領域外は除外

        selected_points = []
        h, w = target_mask.shape
        grid_y, grid_x = np.ogrid[:h, :w]

        for i in range(num_points):
            if i == 0:
                score_map = base_score_map.copy()
            else:
                dist_to_selected = np.ones((h, w), dtype=np.float32) * np.inf
                for py, px in selected_points:
                    d = np.sqrt((grid_y - py)**2 + (grid_x - px)**2)
                    dist_to_selected = np.minimum(dist_to_selected, d)
                
                norm_dist_selected = dist_to_selected / (np.max(dist_to_selected) + 1e-8)
                score_map = base_score_map * norm_dist_selected

            best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
            selected_points.append((best_y, best_x))

        return np.array([(x, y) for y, x in selected_points], dtype=np.int32)

    def convert_coordinate(self, point: int, img_size: int, feat_size: int) -> float:
        return point * img_size / feat_size

    def transform_coords(self, pts: np.ndarray, width: int, height: int, feature_size: int) -> List[List[float]]:
        return [
            [
                self.convert_coordinate(x + 0.5, width, feature_size),
                self.convert_coordinate(y + 0.5, height, feature_size)
            ]
            for x, y in pts
        ]

    # =========================================================================
    # 4. EdgeTAM 推論バックエンド共通インターフェース (OpenVINO対応化用)
    # =========================================================================

    def _predict_edgetam_encoder(
        self, 
        image_raw: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.edgetam.encode(self.edgetam.transform(image_raw))

    def _predict_edgetam_decoder(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Optional[Tuple],
        height: int,
        width: int,
        mask_input: Optional[Union[np.ndarray, torch.Tensor]] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mask_input is None:
            mask_logits_tensor = torch.tensor([0], dtype=torch.float32, device=self.device)
        else:
            mask_logits_tensor = mask_input

        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self.edgetam.preprocess(
            (height, width),
            point_coords=torch.tensor(point_coords, dtype=torch.float32, device=self.device),
            point_labels=torch.tensor(point_labels, dtype=torch.int64, device=self.device),
            mask_logits=mask_logits_tensor,
        )

        high_res_feats0, high_res_feats1, image_embed = feats
        low_res_masks, iou_predictions = self.edgetam.decode(
            high_res_feats0, high_res_feats1, image_embed,
            point_coords=point_coords_norm,
            point_labels=point_labels_norm,
            mask_input=mask_input_tensor,
            has_mask=has_mask,
        )

        masks, scores, logits = self.edgetam.postprocess(
            low_res_masks, iou_predictions
        )

        return masks.detach().numpy(), scores.detach().numpy(), logits

    # =========================================================================
    # 5. 各パイプライン・サブタスク処理
    # =========================================================================

    def _generate_foreground_map(
        self,
        image_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[torch.Tensor]]:
        """CLIP特徴量とKMeansによるベース前景マップの抽出"""
        image_features, patch_features, cls2patch_attn, layer_features = (
            self.extractor.extract_image_features(image_tensor)
        )
        text_probs = (100.0 * image_features @ self.fgbg_text_features.T).softmax(dim=-1)

        L_START, L_END = 7, 10
        fused_features = torch.cat(layer_features[L_START:L_END], dim=2)
        fused_features = F.normalize(fused_features, dim=-1)

        features_np = fused_features.numpy().reshape(-1, fused_features.shape[-1])
        patch_tokens = features_np[1:, :]

        cluster_labels_list = []
        fg_prompts_num = len(self.foreground_prompts)
        foreground_map = np.zeros((self.feature_size * self.feature_size,), dtype=np.float32)

        for k_clusters in self.k_clusters:
            cluster_labels = self.fit_predict_kmeans(
                patch_tokens, k=k_clusters, niter=self.niter
            )
            cluster_labels_list.append(cluster_labels)

            label_token_list = self.extractor.extract_cluster_tokens(
                patch_features, torch.from_numpy(cluster_labels).to(torch.int64)
            )

            for label_idx, label_features in enumerate(label_token_list):
                label2text_probs = (
                    100.0 * label_features.reshape(1, -1) @ self.fgbg_text_features.T
                ).softmax(dim=-1)
                probs_cpu = label2text_probs[0].cpu().numpy()
                foreground_confidence = np.sum(probs_cpu[:fg_prompts_num])
                foreground_map[cluster_labels == label_idx] += foreground_confidence

        foreground_map = foreground_map / len(self.k_clusters)
        foreground_map = foreground_map.reshape(self.feature_size, self.feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)

        return foreground_map, foreground_mask, cluster_labels_list, layer_features

    def _run_edgetam_refinement(
        self,
        foreground_mask: np.ndarray,
        image_raw: Image.Image,
        width: int,
        height: int,
        init_n_points: int = 5,
        max_refine_steps: int = 5
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[int]]:
        """EdgeTAM による初期プロンプト設定および繰り返しマスク精緻化"""
        fg_mask = (foreground_mask == 1).astype(np.uint8)
        bg_mask = (foreground_mask == 0).astype(np.uint8)

        # --- Step 1: 初期の最良点（前景5点 + 背景5点）でベースマスクを作成 ---
        init_fg_pt = self.select_n_prompt_points(fg_mask, bg_mask, num_points=init_n_points)
        init_bg_pt = self.select_n_prompt_points(bg_mask, fg_mask, num_points=init_n_points)

        point_coords = (
            self.transform_coords(init_fg_pt, width, height, self.feature_size) +
            self.transform_coords(init_bg_pt, width, height, self.feature_size)
        )
        point_labels = [1] * init_n_points + [0] * init_n_points

        with Timer("edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

        with Timer("edgetam1回目マスク推定"):
            masks, scores, logits = self._predict_edgetam_decoder(
                point_coords=point_coords,
                point_labels=point_labels,
                feats=feats,
                height=height,
                width=width,
                multimask_output=True
            )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]

        # --- Step 2: 誤差（差分領域）から動的にポイントを追加して精緻化 ---
        current_mask_list = [current_mask]
        h_gt, w_gt = fg_mask.shape[:2]

        for step in range(max_refine_steps):
            current_mask_resized = cv2.resize(
                current_mask, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST
            )

            fg_fn_mask = (fg_mask == 1) & (current_mask_resized == 0)
            new_pt = self.select_n_prompt_points(fg_fn_mask.astype(np.uint8), bg_mask, num_points=1)

            if len(new_pt) == 0:
                break

            new_coord = self.transform_coords(new_pt, width, height, self.feature_size)[0]
            point_coords.append(new_coord)
            point_labels.append(1)

            with Timer(f"edgetam{step+2}回目マスク推定"):
                masks, scores, logits = self._predict_edgetam_decoder(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    feats=feats,
                    height=height,
                    width=width,
                    mask_input=low_res_logits,
                    multimask_output=False
                )

            best_idx = np.argmax(scores)
            current_mask = masks[best_idx].astype(np.uint8)
            low_res_logits = logits[best_idx:best_idx + 1]

            current_mask_list.append(current_mask)

        # マスク画像を元画像サイズにリサイズ
        resized_mask: np.ndarray = cv2.resize(
            current_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_mask, current_mask_list, point_coords, point_labels

    def _save_results(
        self,
        image_path: Path, 
        image_raw: Image.Image,
        width: int,
        height: int,
        current_mask: np.ndarray,
        current_mask_list: List[np.ndarray],
        foreground_map: np.ndarray,
        foreground_mask: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        layer_features: List[torch.Tensor],
        cluster_labels_list: List[np.ndarray]
    ) -> None:
        """処理結果の画像・マスク・オーバーレイの書き出し"""
        save_dir = Path(self.output_dir) / image_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Saving results to {save_dir} ---")

        # マスク画像を保存
        for i, mask in enumerate(current_mask_list):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{i+1}.png"), mask_img)

        # 各層のクラスタリング結果の画像を保存
        for i, layer_feature in enumerate(layer_features):
            layer_feature = F.normalize(layer_feature, dim=-1)
            lf_np = layer_feature.numpy()[:, 1:, :]
            lf_np = lf_np.reshape(self.feature_size * self.feature_size, -1)
            for k_clusters in self.k_clusters:
                cluster_labels = self.fit_predict_kmeans(lf_np, k=k_clusters)
                cluster_labels_2d = cluster_labels.reshape(self.feature_size, self.feature_size)
                cluster_rgb = self.colorize_clusters_PIL(cluster_labels_2d)
                cluster_rgb = cluster_rgb.resize(
                    (width, height), resample=Image.Resampling.NEAREST
                )
                cluster_rgb.save(save_dir / f"layer{i + 1:02d}_k{k_clusters}.png")

        # レイヤー結合した特徴マップでのクラスタリング結果
        for k_clusters, cluster_labels in zip(self.k_clusters, cluster_labels_list):
            fused_cluster_labels_2d = cluster_labels.reshape(self.feature_size, self.feature_size)
            fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
            fused_cluster_rgb = fused_cluster_rgb.resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            fused_cluster_rgb.save(save_dir / f"layer_fused_k{k_clusters}.png")

        # 前景度合いマップを保存
        foreground_map_2d = (
            (foreground_map * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        map_image = Image.fromarray(foreground_map_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        map_image.save(save_dir / "foreground_map.png")

        # 初期前景マスク画像を保存
        foreground_mask_2d = (
            (foreground_mask * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        mask_image = Image.fromarray(foreground_mask_2d).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask_image.save(save_dir / "foreground_mask.png")

        # オーバーレイ画像保存 (CLIP前景プロンプト点付き)
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_image)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = mask_cv
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        
        if np.sum(foreground_mask == 1) > 0:
            for point_label, point_coord in zip(point_labels, point_coords):
                color = (0, 255, 0) if point_label == 1 else (0, 0, 255)
                cv2.circle(
                    overlay, 
                    (int(point_coord[0]), int(point_coord[1])), 
                    20, color, -1
                )
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)

        # オーバーレイ画像（EdgeTAM 最終マスク版）
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = (current_mask * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay_edgetam.png"), overlay)

        print("Save completed.")

    # =========================================================================
    # 6. メイン実行エントリーポイント
    # =========================================================================

    def run(self, image_path: Union[str, Path]) -> None:
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if image_path.exists():
            print(f"====== 画像パス: {image_path.name} ======")
        else:
            print(f"====== 存在しません: {image_path.name} ======")
            return

        # 画像読み込み
        image_raw = load_image(image_path)
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        # 画像前処理
        image_tensor = self.extractor.preprocess(Image.fromarray(image_raw)).unsqueeze(0)

        # 1. CLIP + KMeans による初期前景マップの推定
        with Timer("特徴量抽出"):
            foreground_map, foreground_mask, cluster_labels_list, layer_features = self._generate_foreground_map(image_tensor)

        # 2. 前景が存在する場合は EdgeTAM による精緻化
        point_coords: List[List[float]] = []
        point_labels: List[int] = []

        if np.sum(foreground_mask == 1) > 0:
            current_mask, current_mask_list, point_coords, point_labels = self._run_edgetam_refinement(
                foreground_mask=foreground_mask,
                image_raw=image_raw,
                width=width,
                height=height,
                init_n_points=5,
                max_refine_steps=5
            )
        else:
            current_mask = np.zeros((height, width), dtype=np.uint8)
            current_mask_list = []

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        # 3. 処理結果の保存
        self._save_results(
            image_path=image_path,
            image_raw=image_raw,
            width=width,
            height=height,
            current_mask=current_mask,
            current_mask_list=current_mask_list,
            foreground_map=foreground_map,
            foreground_mask=foreground_mask,
            point_coords=point_coords,
            point_labels=point_labels,
            layer_features=layer_features,
            cluster_labels_list=cluster_labels_list
        )