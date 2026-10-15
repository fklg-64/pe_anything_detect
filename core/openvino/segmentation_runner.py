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

import numpy as np

def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """
    points: (M, 2) の座標配列。各行は (x, y)
    k:      抽出する点の数

    return: (k, 2) の座標配列
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")

    m = len(points)

    if not 0 <= k <= m:
        raise ValueError("k must be between 0 and len(points)")

    if k == 0:
        return points[:0].copy()

    if k == m:
        return points.copy()

    # 距離計算用
    pts = points.astype(np.float64, copy=False)

    # 最初は重心に最も近い点を選ぶ
    center = pts.mean(axis=0)
    first = np.argmin(np.sum((pts - center) ** 2, axis=1))

    selected = np.empty(k, dtype=np.intp)
    selected[0] = first

    # 各点から、選択済みの点までの最小二乗距離
    min_dist2 = np.sum((pts - pts[first]) ** 2, axis=1)
    min_dist2[first] = -1

    for i in range(1, k):
        # 選択済みの点から最も遠い候補を選ぶ
        idx = np.argmax(min_dist2)
        selected[i] = idx

        # 新しく選んだ点との距離で最小距離を更新
        dist2 = np.sum((pts - pts[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

        # 重複選択を防ぐ
        min_dist2[selected[:i + 1]] = -1

    return points[selected]

def clean_mask(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    mask: 2次元の二値マスク（0/1 または 0/255）
    min_area: 除去する連結成分の最小面積（ピクセル数）

    return: 0/1 の二値マスク
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)

    # 小さな独立成分を除去
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    # 輪郭の細かなノイズを平滑化
    cleaned = cv2.medianBlur(cleaned, 3)

    return cleaned

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
        self.et_fds = config["edgetam_high_res_feature_downscale_factor"]
        self.fg_pt_per_cls = config["edgetam_foreground_points_per_class"]
        self.bg_pt_cnt = config["edgetam_background_point_count"]

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size
        # edgetamの高解像度特徴量サイズ
        self.et_high_res_feature_size: int = self.edgetam.target_size // self.et_fds

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
        image_raw: np.ndarray, 
        foreground_mask: np.ndarray, 
        semantic_labels: np.ndarray, 
        width: int,
        height: int
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """粗い前景マスクからSAM用の前景・背景ポイントを抽出する。

        Returns:
            foreground_points: 前景連結成分の重心 (N, 2)。座標は (x, y)。
            background_points: 各グリッドから選択した背景ポイント (M, 2)。
        """
        # EdgeTAMの特徴量抽出
        with Timer("    ┌ edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

        if semantic_labels.shape != foreground_mask.shape:
            raise ValueError(
                f"Shape mismatch: semantic_labels={semantic_labels.shape}, "
                f"foreground_mask={foreground_mask.shape}"
            )


        # 1. 前景と完全な背景を取得
        foreground_binary = foreground_mask != 0
        background_binary = ~foreground_binary

        # 背景点はクラス間で共通の候補から1点選択
        kernel = np.ones((3, 3), dtype=np.float32)
        background_map = convolve(background_binary.astype(np.uint8), kernel) / np.sum(kernel)
        background_point = np.argwhere(background_map > 0.5)[:, ::-1]
        background_point = farthest_point_sampling(background_point, self.bg_pt_cnt)

        # 3. セマンティッククラスごとにマスクを精密化
        semantic_ids = np.unique(semantic_labels[foreground_binary])

        for semantic_id in semantic_ids:
            # 対象クラスの粗いマスク
            coarse_semantic_mask = ((semantic_labels == semantic_id) & foreground_binary).astype(np.uint8)
            if not np.any(coarse_semantic_mask):
                continue

            coarse_semantic_map = convolve(coarse_semantic_mask, kernel) / np.sum(kernel)

            # 4. 粗いセマンティックマスクをロジットに変換
            # 背景: -8、前景: +8
            mask_logits = coarse_semantic_map * 16 - 8

            # 5. マスクをedgetamの高解像度特徴量サイズにリサイズ
            mask_input = cv2.resize(
                mask_logits,
                (self.et_high_res_feature_size, self.et_high_res_feature_size),
                interpolation=cv2.INTER_LINEAR,
            )
            # mask_inputに入力できる形状に変更
            mask_input = mask_input[None, ...]

            # 6. 正例点を対象セマンティック領域の内部から1点選択
            foreground_point = np.argwhere(coarse_semantic_map > 0.5)[:, ::-1]
            foreground_point = farthest_point_sampling(foreground_point, self.fg_pt_per_cls)
            if foreground_point is None:
                continue

            # 正例
            selected_points: List[List[float]] = foreground_point.tolist()
            point_labels: List[int] = [1] * self.fg_pt_per_cls
            # 負例
            if background_point is not None:
                selected_points += background_point.tolist()
                point_labels += [0] * self.bg_pt_cnt

            # 入力座標をEdgeTAM用の座標に変換
            point_coords = self.transform_coords(
                selected_points,
                width,
                height,
                self.feature_size,
            )

            # 7. EdgeTAMによるマスク推定
            for _ in range(5):
                with Timer(f"    ┌ edgetam{semantic_id}回目マスク推定"):
                    masks, scores, logits = self._predict_edgetam_decoder(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        feats=feats,
                        height=height,
                        width=width,
                        mask_input=mask_input,
                    )

                if len(scores) == 0:
                    continue

                # 品質スコア最大の候補を採用
                best_idx = int(np.argmax(scores))

                current_mask = np.asarray(
                    masks[best_idx]
                ).squeeze().astype(np.uint8)

                current_mask = clean_mask(current_mask, min_area=8)

                mask_input = logits[best_idx:best_idx + 1]

            # 8. デバッグ画像
            if getattr(self, "debug", True):
                cv2.imwrite(
                    f"./tmp/f{semantic_id}.png",
                    (coarse_semantic_mask * 255).astype(np.uint8),
                )

                cv2.imwrite(
                    f"./tmp/fm{semantic_id}.png",
                    (coarse_semantic_map * 255).astype(np.uint8),
                )

                cv2.imwrite(
                    f"./tmp/b{semantic_id}.png",
                    (background_binary * 255).astype(np.uint8) ,
                )

                cv2.imwrite(
                    f"./tmp/bm{semantic_id}.png",
                    (background_map * 255).astype(np.uint8) ,
                )

                cv2.imwrite(
                    f"./tmp/segmentation{semantic_id}.png",
                    (current_mask * 255).astype(np.uint8) ,
                )

        foreground_points = np.zeros((1,2))
        background_points = np.zeros((1,2))
        foreback_image_list = []
        return foreground_points, background_points, foreback_image_list

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

        with Timer("    ┌ デコード"):
            high_res_feats0, high_res_feats1, image_embed = feats
            low_res_masks, iou_predictions = self.edgetam.decode(
                high_res_feats0, high_res_feats1, image_embed,
                point_coords=point_coords_norm,
                point_labels=point_labels_norm,
                mask_input=mask_input_tensor,
                has_mask=has_mask,
            )

        with Timer("    ┌ リサイズ"):
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """CLIP特徴量とKMeansによるベース前景マップの抽出"""

        # パッチレベルの特徴量を取得
        patch_tokens = fused_patch_features[0, 1:, :]

        cluster_index_list: List[int] = []
        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_map: np.ndarray = np.zeros((self.feature_size * self.feature_size,), dtype=np.float32)
        with Timer("    ┌ Kmeansクラスタ処理時間"):
            cluster_labels = self.fit_predict_kmeans(patch_tokens, k=self.k_clusters, niter=self.niter)

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

            # 前景か背景か
            if foreground_confidence > self.foreground_threshold:
                foreground_index = np.argmax(fg_logits, axis=1)
                cluster_index_list += [foreground_index[0]]
            else:
                background_index = np.argmax(bg_logits, axis=1)
                cluster_index_list += [fg_prompts_num + background_index[0]]
                
            # クラスタ内の全patchにconfidenceを付与
            foreground_map[cluster_labels == label_idx] += foreground_confidence

        # KMeansの結果を平均
        foreground_map = foreground_map.reshape(self.feature_size, self.feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)

        return (
            foreground_map,
            foreground_mask,
            cluster_labels, 
            cluster_index_list
        )

    def _run_edgetam_refinement(
        self,
        foreground_points: np.ndarray,
        background_points: np.ndarray,
        image_raw: Image.Image,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[int]]:
        """マスク生成"""

        # edgeTAMの特徴量抽出
        with Timer("    ┌ edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

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

    def semantic_segmentation(
        self,
        cluster_labels: np.ndarray,
        cluster_index_list: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        cluster_indices = np.asarray(cluster_index_list)
        semantic_indices = cluster_indices[cluster_labels]

        semantic_list, semantic_labels = np.unique(
            semantic_indices,
            return_inverse=True,
        )

        return semantic_list, semantic_labels.reshape(self.feature_size, self.feature_size)

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
        foreback_image_list: List[np.ndarray],
        point_coords: List[List[float]],
        point_labels: List[int],
        cluster_labels: np.ndarray, 
        semantic_labels: np.ndarray, 
    ) -> None:
        """処理結果の画像・マスク・オーバーレイの書き出し"""
        # マスク画像を保存
        for i, mask in enumerate(current_mask_list):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{i+1}.png"), mask_img)

        # レイヤー結合した特徴マップでのクラスタリング結果
        fused_cluster_labels_2d = cluster_labels.reshape(self.feature_size, self.feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused_k{self.k_clusters}.png")

        # セマンティックセグメンテーション処理
        semantic_labels_2d = semantic_labels.reshape(self.feature_size, self.feature_size)
        semantic_labels_rgb = self.colorize_clusters_PIL(semantic_labels_2d)
        semantic_labels_rgb = semantic_labels_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        semantic_labels_rgb.save(save_dir / f"semantic_segmentation.png")

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

        # 前景と背景っぽさを保存
        for i, foreback_map in enumerate(foreback_image_list):
            image = Image.fromarray(
                (foreback_map * 255).astype(np.uint8)
            )
            image = image.resize(
                (width, height),
                resample=Image.Resampling.NEAREST,
            )
            image.save(self.save_dir / f"foreback_map{i}.png")

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
            foreground_map, foreground_mask, cluster_labels, cluster_index_list  = self._generate_foreground_map(patch_features, fused_patch_features)

        # セマンティックセグメンテーションのラベルを取得
        with Timer("セマンティックセグメンテーション処理時間"):
            semantic_list, semantic_labels = self.semantic_segmentation(cluster_labels, cluster_index_list)

        # 前景と背景のポイントを抽出
        with Timer("ポイント抽出処理時間"):
            foreground_points, background_points, foreback_image_list = self.select_prompt_points(image_raw, foreground_mask, semantic_labels, width, height)

        # 前景が存在する場合は EdgeTAM による精緻化
        with Timer("詳細な前景マスク取得処理時間"):
            point_coords: List[List[float]] = []
            point_labels: List[int] = []
            if np.sum(foreground_mask == 1) > 0:
                current_mask, current_mask_list, point_coords, point_labels = self._run_edgetam_refinement(
                    foreground_points=foreground_points,
                    background_points=background_points,
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
            foreback_image_list=foreback_image_list,
            point_coords=point_coords,
            point_labels=point_labels,
            cluster_labels=cluster_labels,
            semantic_labels=semantic_labels
        )