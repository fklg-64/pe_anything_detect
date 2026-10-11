import time
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import faiss
import numpy as np
from PIL import Image

from core.utils import Timer, load_image, preprocess_image
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.edgetam_runner import OVEdgeTamRunner

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import convolve

class SegmentationRunner:
    """抽出された特徴量・クラスタリングを用いた推論・マスク生成の実行クラス"""

    def __init__(
        self, 
        extractor: OVCLIPFeatureExtractor, 
        edgetam: OVEdgeTamRunner, 
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
        self.output_dir = config["output_dir"]
        self.color_palette = config["color_palette"]
        self.init_n_points = config["init_n_points"]
        self.max_refine_steps = config["max_refine_steps"]

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size

        # プロンプトとプロンプトのテキスト特徴量
        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: Optional[np.ndarray] = None

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    # =========================================================================
    # 1. テキストプロンプト & 特徴量準備
    # =========================================================================

    def load_prompts(self, path: str, mode: int=0) -> List[str]:
        if mode == 0:
            with open(path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data["base"] + data["custom"]
        elif mode == 1:
            prompts = []
            with open(path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for color in data["color"]:
                for shape in data["shape"]:
                    article = "an" if color[0].lower() in "aeiou" else "a"
                    prompts += [f"a photo of {article} {color} {shape} object."]
            return prompts + data["custom"]
        elif mode == 2:
            prompts = []
            with open(path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for cond in data["condition"]:
                for mat in data["material"]:
                    for ctx in data["context"]:
                        article = "an" if cond[0].lower() in "aeiou" else "a"
                        prompt = f"a photo of {article} {cond} {mat} {ctx}."
                        prompts.append(prompt)
            return prompts + data["custom"]
    
    def common_process(self, prompts: List[str]) -> np.ndarray:
        text_features = self.extractor.extract_text_features(prompts)
        return text_features
    
    def register_fgbg_text_features(self) -> None:
        foreground_prompts_path = Path(self.prompts_dir) / "foreground.yaml"
        background_prompts_path = Path(self.prompts_dir) / "background.yaml"

        self.foreground_prompts = self.load_prompts(foreground_prompts_path, mode=1)
        self.background_prompts = self.load_prompts(background_prompts_path, mode=0)

        fg_feats = self.common_process(self.foreground_prompts)
        bg_feats = self.common_process(self.background_prompts)

        self.fgbg_text_features = np.vstack([fg_feats, bg_feats])

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

    def select_prompt_points(
        self,
        foreground_mask: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """粗い前景マスクからSAM用の前景・背景ポイントを抽出する。

        Returns:
            foreground_points: 前景連結成分の重心 (N, 2)。座標は (x, y)。
            background_points: 各グリッドから選択した背景ポイント (M, 2)。
        """
        # 1. 前景・背景マスクを二値化
        foreground = (foreground_mask != 0).astype(np.float32)
        background = 1.0 - foreground

        # 2. 3x3近傍の前景・背景画素数を計算
        kernel = np.ones((3, 3), dtype=np.float32)

        foreground_response = convolve(foreground, kernel)
        background_response = convolve(background, kernel)

        # 3. 3x3近傍がすべて前景の領域からポイントを抽出
        foreground_interior = (
            foreground_response == 9
        ).astype(np.uint8) * 255

        _, _, _, centroids = cv2.connectedComponentsWithStats(
            foreground_interior,
            connectivity=8,
        )

        # ラベル0は背景なので除外
        foreground_points = centroids[1:].astype(np.float32)

        # 4. 背景ポイントをグリッドごとに抽出
        background_interior = background_response == 9

        grid_divisions = 8
        mask_height, mask_width = foreground.shape
        selected_points = []

        for row in range(grid_divisions):
            y1 = row * mask_height // grid_divisions
            y2 = (row + 1) * mask_height // grid_divisions

            for col in range(grid_divisions):
                x1 = col * mask_width // grid_divisions
                x2 = (col + 1) * mask_width // grid_divisions

                cell = background_interior[y1:y2, x1:x2]

                # セル内の半数を超える画素が背景内部の場合のみ選択
                if cell.size == 0 or np.count_nonzero(cell) <= cell.size // 2:
                    continue

                # セル内の背景画素から中心に最も近い点を選択
                bg_cell = background[y1:y2, x1:x2]
                y_coords, x_coords = np.where(bg_cell == 1)

                if x_coords.size == 0:
                    continue

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                distances = np.hypot(
                    x_coords + x1 - center_x,
                    y_coords + y1 - center_y,
                )

                idx = np.argmin(distances)

                selected_points.append([
                    x_coords[idx] + x1,
                    y_coords[idx] + y1,
                ])

        background_points = np.asarray(
            selected_points, dtype=np.float32
        ).reshape(-1, 2)

        # 5. デバッグ画像を保存
        if False:
            for idx, response in enumerate(
                (foreground_response, background_response)
            ):
                response_norm = response / kernel.size
                image = Image.fromarray(
                    (response_norm * 255).astype(np.uint8)
                )
                image = image.resize(
                    (target_width, target_height),
                    resample=Image.Resampling.NEAREST,
                )
                image.save(self.save_dir / f"tmp{idx}.png")

        return foreground_points, background_points

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

    def _predict_edgetam_encoder(self, image_raw: Image.Image) -> Optional[Tuple]:
        """EdgeTAM Encoder の実行（PyTorch / Custom / 将来のOpenVINOの違いをカプセル化）"""
        return self.edgetam.encode(
            preprocess_image(
                image_raw, 
                self.edgetam.target_size, 
                self.edgetam.mean, self.edgetam.std
            )
        )

    def _predict_edgetam_decoder(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Optional[Tuple],
        height: int,
        width: int,
        mask_input: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """EdgeTAM Decoder の実行（PyTorch / Custom / 将来のOpenVINOの違いをカプセル化）"""
        if mask_input is None:
            mask_logits_tensor = np.array([0], dtype=np.float32)
        else:
            mask_logits_tensor = mask_input

        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self.edgetam.preprocess(
            (height, width),
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
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
            low_res_masks, iou_predictions, (height // 8, width // 8)
        )

        return masks, scores, logits

    # =========================================================================
    # 5. 各パイプライン・サブタスク処理
    # =========================================================================

    def _feature_extraction(
        self, 
        image_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_features, patch_features, cls2patch_attn, fused_patch_features = (
            self.extractor.extract_image_features(image_tensor)
        )
        return image_features, patch_features, fused_patch_features

    def _image_classification(
        self, 
        image_features: np.ndarray
    ):
        # クラス分類
        text_probs = 100.0 * image_features @ self.fgbg_text_features.T
        text_probs = self._softmax(text_probs, axis=-1)

    def _logmeanexp(
        self,
        x: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """Numerically stable log-mean-exp."""
        max_x = np.max(x, axis=axis, keepdims=True)
        lse = max_x + np.log(
            np.mean(np.exp(x - max_x), axis=axis, keepdims=True) + 1e-12
        )
        return np.squeeze(lse, axis=axis)

    def _generate_foreground_map(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """CLIP特徴量とKMeansによるベース前景マップの抽出"""

        # パッチレベルの特徴量を取得
        patch_tokens = fused_patch_features[0, 1:, :]

        cluster_labels_list: List[np.ndarray] = []
        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_map: np.ndarray = np.zeros((self.feature_size * self.feature_size,), dtype=np.float32)
        for k_clusters in self.k_clusters:
            with Timer("    ┌ Kmeansクラスタ処理時間"):
                cluster_labels = self.fit_predict_kmeans(patch_tokens, k=k_clusters, niter=self.niter)
            cluster_labels_list.append(cluster_labels)

            with Timer("    ┌ クラスタごとのCLSトークン抽出処理時間"):
                label_token_list: List[np.ndarray] = self.extractor.extract_cluster_tokens(patch_features, cluster_labels)

            for label_idx, label_features in enumerate(label_token_list):
                # feature normalization
                label_features: np.ndarray = label_features.reshape(1, -1)
                label_norm: np.ndarray = np.linalg.norm(label_features, axis=-1, keepdims=True) + 1e-8
                label_features: np.ndarray = label_features / label_norm

                # similarity / logits
                logits = (100.0 * label_features @ self.fgbg_text_features.T)

                # foreground / background を分離
                fg_logits = logits[:, :fg_prompts_num]
                bg_logits = logits[:, fg_prompts_num:]

                # prompt数を補正したグループスコア
                fg_score = self._logmeanexp(fg_logits, axis=-1)
                bg_score = self._logmeanexp(bg_logits, axis=-1)

                # FG vs BG の2クラスsoftmax
                group_logits = np.concatenate([fg_score[:, None], bg_score[:, None]], axis=-1)
                group_probs = self._softmax(group_logits, axis=-1)
                foreground_confidence = float(group_probs[0, 0])

                # クラスタ内の全patchにconfidenceを付与
                foreground_map[cluster_labels == label_idx] += foreground_confidence

        # KMeansの結果を平均
        foreground_map /= len(self.k_clusters)
        foreground_map = foreground_map.reshape(self.feature_size, self.feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)

        return (
            foreground_map,
            foreground_mask,
            cluster_labels_list
        )

    def _run_edgetam_refinement(
        self,
        foreground_mask: np.ndarray,
        image_raw: Image.Image,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[int]]:
        """マスク生成"""

        # edgeTAMの特徴量抽出
        with Timer("    ┌ edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

        # 前景と背景のポイントを抽出
        with Timer("    ┌ ポイント抽出処理時間"):
            foreground_points, background_points = self.select_prompt_points(foreground_mask, width, height)

        # 前景が存在しない場合はマスクなしとして処理を終了
        current_mask_list: List[np.ndarray] = []
        point_coords: List[List[float]] = []
        point_labels: List[int] = []
        if len(foreground_points) == 0:
            resized_mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            return resized_mask, current_mask_list, [], []

        # 前景のポイントのみでセグメンテーション
        point_coords += self.transform_coords(foreground_points.tolist(), width, height, self.feature_size)
        point_labels += [1] * len(foreground_points)

        with Timer("    ┌ edgetam1回目マスク推定"):
            masks, scores, logits = self._predict_edgetam_decoder(
                point_coords=point_coords,
                point_labels=point_labels,
                feats=feats,
                height=height,
                width=width
            )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]
        current_mask_list += [current_mask]

        # 背景のポイントも追加してセグメンテーション
        point_coords += self.transform_coords(background_points.tolist(), width, height, self.feature_size)
        point_labels += [0] * len(background_points)

        with Timer("    ┌ edgetam2回目マスク推定"):
            masks, scores, logits = self._predict_edgetam_decoder(
                point_coords=point_coords,
                point_labels=point_labels,
                feats=feats,
                height=height,
                width=width,
                mask_input=low_res_logits
            )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]
        current_mask_list += [current_mask]

        # マスク画像を元画像サイズにリサイズ
        resized_mask: np.ndarray = cv2.resize(
            current_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_mask, current_mask_list, point_coords, point_labels

    def _save_results(
        self,
        save_dir: Path, 
        image_raw: Image.Image,
        width: int,
        height: int,
        current_mask: np.ndarray,
        current_mask_list: List[np.ndarray],
        foreground_map: np.ndarray,
        foreground_mask: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        cluster_labels_list: List[np.ndarray]
    ) -> None:
        """処理結果の画像・マスク・オーバーレイの書き出し"""
        # マスク画像を保存
        for i, mask in enumerate(current_mask_list):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{i+1}.png"), mask_img)

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

        print(f"Saving results to {save_dir}")

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

        self.save_dir = Path(self.output_dir) / image_path.stem
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 画像読み込み
        image_raw = load_image(image_path)
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        # 画像前処理
        with Timer("画像前処理時間"):
            image_tensor = preprocess_image(image_raw, self.extractor.target_size, self.extractor.mean, self.extractor.std)

        # 特徴量抽出
        with Timer("特徴量抽出処理時間"):
            image_features, patch_features, fused_patch_features = self._feature_extraction(image_tensor)

        # 画像分類
        with Timer("画像分類処理時間"):
            self._image_classification(image_features)

        # CLIP + KMeans による初期前景マップの推定
        with Timer("粗い前景マスク取得処理時間"):
            foreground_map, foreground_mask, cluster_labels_list = self._generate_foreground_map(patch_features, fused_patch_features)

        # 前景が存在する場合は EdgeTAM による精緻化
        with Timer("詳細な前景マスク取得処理時間"):
            point_coords: List[List[float]] = []
            point_labels: List[int] = []
            if np.sum(foreground_mask == 1) > 0:
                current_mask, current_mask_list, point_coords, point_labels = self._run_edgetam_refinement(
                    foreground_mask=foreground_mask,
                    image_raw=image_raw,
                    width=width,
                    height=height
                )
            else:
                current_mask = np.zeros((height, width), dtype=np.uint8)
                current_mask_list = []

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        # 処理結果の保存
        self._save_results(
            save_dir=self.save_dir,
            image_raw=image_raw,
            width=width,
            height=height,
            current_mask=current_mask,
            current_mask_list=current_mask_list,
            foreground_map=foreground_map,
            foreground_mask=foreground_mask,
            point_coords=point_coords,
            point_labels=point_labels,
            cluster_labels_list=cluster_labels_list
        )