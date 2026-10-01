import os
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from core.torch.feature_extractor import TorchCLIPFeatureExtractor
from core.torch.edgetam_runner import EdgeTAMPredictor, RepVitModel, DecodeModel
from core.torch.analysis_function import export_complete_vit_analysis_to_excel
from core.utils import *

from scipy.ndimage import distance_transform_edt

class SegmentationRunner:
    """抽出された特徴量・クラスタリングを用いた推論・マスク生成の実行クラス"""

    def __init__(
        self, 
        extractor: TorchCLIPFeatureExtractor, 
        prompts_dir: str=f"./prompts", 
        k_clusters: Optional[int]=12, 
        niter: int=10, 
        foreground_threshold: float=0.5
    ):
        self.extractor = extractor
        self.prompts_dir = prompts_dir
        self.k_clusters = k_clusters
        self.niter = niter
        self.foreground_threshold = foreground_threshold
        self.device = "cpu"

        self.is_custom = False
        if self.is_custom:
            self.edgetam_predictor = EdgeTAMPredictor()
        else:
            self.edgetam_encoder = RepVitModel()
            self.edgetam_decoder = DecodeModel()

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size

        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: Optional[torch.Tensor] = None

        self.register_fgbg_text_features()

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

    def fit_predict_kmeans(
        self, features: np.ndarray, 
        k: int=12, 
        niter: int=10
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
        color_map.putpalette(COLOR_PALETTE)
        return color_map

    def select_n_prompt_points(
        self, 
        target_mask: np.ndarray, 
        opponent_mask: np.ndarray, 
        num_points: int = 1
    ) -> np.ndarray:
        """
        対象領域（target_mask）の中から、以下の条件を満たす点を順に num_points 個抽出する:
        1. 領域の内側に位置する（境界から遠い）
        2. 対立領域（opponent_mask）から遠い
        3. 既に選ばれている他のプロンプト点から遠い
        
        :return: 座標配列 shape: (num_points, 2) -> [[x1, y1], [x2, y2], ...]
        """
        if not np.any(target_mask) or num_points <= 0:
            return np.empty((0, 2), dtype=np.int32)

        # 1. 基本スコアマップの作成（領域の中心度 × 対立領域からの距離）
        dist_target = distance_transform_edt(target_mask)
        dist_opponent = distance_transform_edt(1 - opponent_mask)

        norm_target = dist_target / (np.max(dist_target) + 1e-8)
        norm_opponent = dist_opponent / (np.max(dist_opponent) + 1e-8)

        base_score_map = norm_target * norm_opponent
        base_score_map[target_mask == 0] = -1.0  # 対象領域外は除外

        selected_points = []
        
        # 全ピクセルの y, x 座標格子をあらかじめ用意（距離計算の高速化用）
        h, w = target_mask.shape
        grid_y, grid_x = np.ogrid[:h, :w]

        for i in range(num_points):
            if i == 0:
                # 1点目: 純粋に基本スコアが最も高い位置（初期候補点）
                score_map = base_score_map.copy()
            else:
                # 2点目以降: 既存の選択済み点との「最小距離マップ」をスコアに乗算
                # （＝どの選択済み点からも遠い場所のスコアが高くなる）
                dist_to_selected = np.ones((h, w), dtype=np.float32) * np.inf
                for py, px in selected_points:
                    # 既存点とのユーグリッド距離
                    d = np.sqrt((grid_y - py)**2 + (grid_x - px)**2)
                    dist_to_selected = np.minimum(dist_to_selected, d)
                
                # 正規化して基本スコアと掛け合わせる
                norm_dist_selected = dist_to_selected / (np.max(dist_to_selected) + 1e-8)
                score_map = base_score_map * norm_dist_selected

            # 最高スコアの座標を取得
            best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
            selected_points.append((best_y, best_x))

        # (y, x) -> (x, y) に変換して返す
        return np.array([(x, y) for y, x in selected_points], dtype=np.int32)

    # 座標を元画像のピクセル位置に変換
    def convert_coordinate(self, point: int, img_size: int, feat_size):
        return point * img_size / feat_size

    def transform_coords(self, pts: np.ndarray, width: int, height: int, feature_size: int) -> List[List[float]]:
        return [
            [
                self.convert_coordinate(x + 0.5, width, feature_size),
                self.convert_coordinate(y + 0.5, height, feature_size)
            ]
            for x, y in pts
        ]

    def run(
        self,
        image_path: str,
        output_dir: str = "./outputs/cluster",
    ):
        feature_size = self.extractor.feature_size

        img_path = Path(image_path)
        if img_path.exists():
            image_raw = Image.open(img_path).convert("RGB")
            width, height = image_raw.size
            print(f"\n--- Processing Image: {img_path.name} ---")
        else:
            print(f"\n--- 存在しません: {img_path.name} ---")
            return
        
        save_dir = Path(output_dir) / img_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()

        image_tensor = self.extractor.preprocess(image_raw).unsqueeze(0)

        # --- 計測対象の処理（推論・特徴量抽出・類似度計算） ---
        with Timer("特徴量抽出"):
            image_features, patch_features, cls2patch_attn, layer_features = (
                self.extractor.extract_image_features(image_tensor)
            )
        text_probs = (100.0 * image_features @ self.fgbg_text_features.T).softmax(dim=-1)

        # 指定したレイヤーの特徴量をチャネル方向に結合
        L_START, L_END = 7, 10
        fused_features = torch.cat(layer_features[L_START:L_END], dim=2)
        fused_features = F.normalize(fused_features, dim=-1)

        # NumPy配列に変換 (形状: [1 (CLS) + パッチ数, 特徴量次元])
        features_np = fused_features.numpy().reshape(-1, fused_features.shape[-1])
        # CLSトークンの特徴量
        cls_token = features_np[:1, :]
        # パッチごとの特徴量
        patch_tokens = features_np[1:, :]

        # 複数のクラスタ数で前景度合いの画像を作成
        cluster_labels_list = []
        K_CLUSTERS_LIST = [8]
        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_map = np.zeros((feature_size * feature_size, ), dtype=np.float32)
        for k_clusters in K_CLUSTERS_LIST:
            cluster_labels = self.fit_predict_kmeans(
                patch_tokens, 
                k=k_clusters, 
                niter=self.niter
            )
            cluster_labels_list += [cluster_labels]

            # クラスタごとの特徴量トークン取得
            label_token_list: List[torch.Tensor] = self.extractor.extract_cluster_tokens(
                patch_features, torch.from_numpy(cluster_labels).to(torch.int64)
            )

            # 前景マスクの抽出
            for label_idx, label_features in enumerate(label_token_list):
                label2text_probs = (
                    100.0 * label_features.reshape(1, -1) @ self.fgbg_text_features.T
                ).softmax(dim=-1)
                probs_cpu = label2text_probs[0].cpu().numpy()
                # 前景度合い
                foreground_confidence = np.sum(probs_cpu[:fg_prompts_num])
                # 前景度合いのマップ
                foreground_map[cluster_labels == label_idx] += foreground_confidence

        # 前景マスクを作成（閾値以上となっている箇所）
        foreground_map = foreground_map / len(K_CLUSTERS_LIST)
        foreground_map = foreground_map.reshape(feature_size, feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)

        # 前景がある場合
        if np.sum(foreground_mask == 1) > 0:
            # --- Step 1: 初期の最良点（前景5点 + 背景5点）でベースマスクを作成 ---
            fg_mask = (foreground_mask == 1).astype(np.uint8)
            bg_mask = (foreground_mask == 0).astype(np.uint8)

            init_n_points = 5
            init_fg_pt = self.select_n_prompt_points(fg_mask, bg_mask, num_points=init_n_points)
            init_bg_pt = self.select_n_prompt_points(bg_mask, fg_mask, num_points=init_n_points)

            # 初期プロンプト座標とラベルの作成
            point_coords = self.transform_coords(init_fg_pt, width, height, feature_size) + self.transform_coords(init_bg_pt, width, height, feature_size)
            point_labels = [1] * init_n_points + [0] * init_n_points

            # 初回推論 (multimask_output=True で最大のスコアを選択)
            with Timer(f"edgetam特徴量抽出"):
                if self.is_custom:
                    self.edgetam_predictor.set_image(image_raw)
                else:
                    high_res_feats0, high_res_feats1, image_embed = self.edgetam_encoder(self.edgetam_encoder.transform(image_raw))
            with Timer(f"edgetam{1}回目マスク推定"):
                if self.is_custom:
                    masks, scores, logits = self.edgetam_predictor.predict(
                        point_coords=np.array(point_coords),
                        point_labels=np.array(point_labels),
                        multimask_output=True
                    )
                else:
                    mask_input, point_coords_norm, point_labels_norm, has_mask = self.edgetam_decoder.preprocess(
                        (height, width), 
                        point_coords=torch.tensor(point_coords, dtype=torch.float32, device=self.device), 
                        point_labels=torch.tensor(point_labels, dtype=torch.int64, device=self.device), 
                        mask_logits=torch.tensor([0], dtype=torch.float32, device=self.device)
                    )
                    low_res_masks, iou_predictions = self.edgetam_decoder(
                        high_res_feats0, high_res_feats1, image_embed, 
                        point_coords = point_coords_norm, 
                        point_labels = point_labels_norm, 
                        mask_input = mask_input, 
                        has_mask = has_mask
                    )
                    masks, scores, logits = self.edgetam_decoder.postprocess(
                        low_res_masks, iou_predictions, (height, width)
                    )
                    masks = masks.detach().numpy()
                    scores = scores.detach().numpy()

            best_idx = np.argmax(scores)
            current_mask = masks[best_idx].astype(np.uint8)
            low_res_logits = logits[best_idx:best_idx + 1]

            # --- Step 2: 誤差（差分領域）から動的にポイントを追加して精緻化 ---
            max_refine_steps: int = 5
            current_mask_list: List[np.ndarray] = [current_mask]

            h_gt, w_gt = fg_mask.shape[:2]

            for step in range(max_refine_steps):
                # 1. current_mask を fg_mask と同じ画像サイズ (w_gt, h_gt) にリサイズ
                current_mask_resized = cv2.resize(
                    current_mask, 
                    (w_gt, h_gt), 
                    interpolation=cv2.INTER_NEAREST
                )

                # 2. 差分領域の計算
                fg_fn_mask = (fg_mask == 1) & (current_mask_resized == 0)  # 前景の未検出領域
                fg_fp_mask = (fg_mask == 0) & (current_mask_resized == 1)  # 前景の誤検出領域
                bg_fn_mask = (bg_mask == 1) & (current_mask_resized == 0)  # 背景の未検出領域

                # 3. 未検出領域から修正ポイントを1点選出
                new_pt = self.select_n_prompt_points(fg_fn_mask.astype(np.uint8), bg_mask, num_points=1)

                if len(new_pt) == 0:
                    break

                # 4. 新しい座標を変換してプロンプトリストへ追加（蓄積）
                new_coord = self.transform_coords(new_pt, width, height, feature_size)[0]
                point_coords.append(new_coord)
                point_labels.append(1)

                # 5. 蓄積された全プロンプト + 前回の logits で再推論
                with Timer(f"edgetam{step+2}回目マスク推定"):
                    if self.is_custom:
                        masks, scores, logits = self.edgetam_predictor.predict(
                            point_coords=np.array(point_coords),
                            point_labels=np.array(point_labels),
                            mask_input=low_res_logits,
                            multimask_output=False
                        )
                    else:
                        mask_input, point_coords_norm, point_labels_norm, has_mask = self.edgetam_decoder.preprocess(
                            (height, width), 
                            point_coords=torch.tensor(point_coords, dtype=torch.float32, device=self.device), 
                            point_labels=torch.tensor(point_labels, dtype=torch.int64, device=self.device), 
                            mask_logits= low_res_logits
                        )
                        low_res_masks, iou_predictions = self.edgetam_decoder(
                            high_res_feats0, high_res_feats1, image_embed, 
                            point_coords = point_coords_norm, 
                            point_labels = point_labels_norm, 
                            mask_input = mask_input, 
                            has_mask = has_mask
                        )
                        masks, scores, logits = self.edgetam_decoder.postprocess(
                            low_res_masks, iou_predictions, (height, width)
                        )
                        masks = masks.detach().numpy()
                        scores = scores.detach().numpy()
                        
                best_idx = np.argmax(scores)
                current_mask = masks[best_idx].astype(np.uint8)
                low_res_logits = logits[best_idx:best_idx + 1]

                current_mask_list.append(current_mask)
        else:
            current_mask = np.zeros((height, width), dtype=np.uint8)
            current_mask_list = []

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        # --- 処理後の画像保存処理（カラー化とリサイズを保存時に実行） ---
        print(f"\n--- Saving results to {save_dir} ---")

        # マスク画像を保存
        for i, mask in enumerate(current_mask_list):
            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{i+1}.png"), mask)

        # 各層のクラスタリング結果の画像を保存
        for i, layer_feature in enumerate(layer_features):
            layer_feature = F.normalize(layer_feature, dim=-1)
            lf_np = layer_feature.numpy()[:, 1:, :]
            lf_np = lf_np.reshape(feature_size * feature_size, -1)
            cluster_labels = self.fit_predict_kmeans(lf_np, k=self.k_clusters)
            cluster_labels_2d = cluster_labels.reshape(feature_size, feature_size)
            cluster_rgb = self.colorize_clusters_PIL(cluster_labels_2d)
            cluster_rgb = cluster_rgb.resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            cluster_rgb.save(save_dir / f"layer{i + 1:02d}.png")

        # アーティファクトの影響調査（アテンション周りのノルムを調査）
        # export_complete_vit_analysis_to_excel(layer_features, feature_size, feature_size, save_dir / "x_norm.xlsx")
        # export_complete_vit_analysis_to_excel(self.extractor.model.visual.trunk.forward_hook["x_mlp"], feature_size, feature_size, save_dir / "x_mlp.xlsx")
        # export_complete_vit_analysis_to_excel(self.extractor.model.visual.trunk.forward_hook["x_attn"], feature_size, feature_size, save_dir / "x_attn.xlsx")
        # export_complete_vit_analysis_to_excel(self.extractor.model.visual.trunk.forward_hook["x_attn_logits_head1"], feature_size, feature_size, save_dir / "x_attn_logits_head1.xlsx")
        # export_complete_vit_analysis_to_excel(self.extractor.model.visual.trunk.forward_hook["x_attn_logits_head2"], feature_size, feature_size, save_dir / "x_attn_logits_head2.xlsx")
        # export_complete_vit_analysis_to_excel(self.extractor.model.visual.trunk.forward_hook["x_attn_logits_head3"], feature_size, feature_size, save_dir / "x_attn_logits_head3.xlsx")

        # レイヤー結合した特徴マップでもクラスタリング結果を保存
        for k_clusters, cluster_labels in zip(K_CLUSTERS_LIST, cluster_labels_list):
            fused_cluster_labels_2d = cluster_labels.reshape(feature_size, feature_size)
            fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
            fused_cluster_rgb = fused_cluster_rgb.resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            fused_cluster_rgb.save(save_dir / f"layer_fused_k{k_clusters}.png")

        # 前景度合いを保存
        foreground_map_2d = (
            (foreground_map * 255)
            .astype(np.uint8)
            .reshape(feature_size, feature_size)
        )
        map_image = Image.fromarray(foreground_map_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        map_image.save(save_dir / "foreground_map.png")

        # マスク画像生成・保存
        foreground_mask_2d = (
            (foreground_mask * 255)
            .astype(np.uint8)
            .reshape(feature_size, feature_size)
        )
        mask_image = Image.fromarray(foreground_mask_2d).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask_image.save(save_dir / "foreground_mask.png")

        # オーバーレイ画像保存
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_image)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = mask_cv
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        if np.sum(foreground_mask == 1) > 0:
            for point_label, point_coord in zip(point_labels, point_coords):
                if point_label == 1:
                    cv2.circle(
                        overlay, 
                        (int(point_coord[0]), int(point_coord[1])), 
                        20, (0, 255, 0), -1
                    )
                elif point_label == 0:
                    cv2.circle(
                        overlay, 
                        (int(point_coord[0]), int(point_coord[1])), 
                        20, (0, 0, 255), -1
                    )
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)

        # オーバーレイ画像（edgetam版）
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = (current_mask * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay_edgetam.png"), overlay)

        print("Save completed.")