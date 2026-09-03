import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import faiss
import numpy as np
from PIL import Image

from core.utils import *
from core.openvino.feature_extractor import OVCLIPFeatureExtractor

class SegmentationRunner:
    """抽出された特徴量・クラスタリングを用いた推論・マスク生成の実行クラス"""

    def __init__(
        self, 
        extractor: OVCLIPFeatureExtractor, 
        prompts_dir: str=f"./prompts", 
        k_clusters: Optional[int]=12, 
        niter: int=10
    ):
        self.extractor = extractor
        self.prompts_dir = prompts_dir
        self.k_clusters = k_clusters
        self.niter = niter

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size

        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: Optional[np.ndarray] = None

        self.register_fgbg_text_features()

    def load_prompts(self, path: str) -> List[str]:
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data["prompts"]

    def common_process(self, prompts: List[str]) -> np.ndarray:
        text_features = self.extractor.extract_text_features(prompts)
        return text_features
    
    def register_fgbg_text_features(self) -> None:
        foreground_prompts_path = Path(self.prompts_dir) / "foreground.json"
        background_prompts_path = Path(self.prompts_dir) / "background.json"

        self.foreground_prompts = self.load_prompts(foreground_prompts_path)
        self.background_prompts = self.load_prompts(background_prompts_path)

        fg_feats = self.common_process(self.foreground_prompts)
        bg_feats = self.common_process(self.background_prompts)

        self.fgbg_text_features = np.vstack([fg_feats, bg_feats])

    def fit_predict_kmeans(
        self, features: np.ndarray, k: int = 12, niter: int = 10
    ) -> np.ndarray:
        N, D = features.shape
        features_contig = (
            np.ascontiguousarray(features, dtype=np.float32)
            if not features.flags["C_CONTIGUOUS"]
            else features
        )
        faiss.omp_set_num_threads(2)
        kmeans = faiss.Kmeans(
            d=D, k=k, niter=niter, spherical=False, verbose=False, seed=42, nredo=1
        )
        kmeans.train(features_contig)
        _, cluster_labels = kmeans.index.search(features_contig, 1)
        return cluster_labels.reshape(-1)

    def colorize_clusters_PIL(self, cluster_labels: np.ndarray) -> Image.Image:
        color_map = Image.fromarray(cluster_labels.astype(np.uint8))
        color_map.putpalette(COLOR_PALETTE)
        return color_map

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def postprocess_mask(
        self, 
        img_cv: np.ndarray, 
        fg_mask: np.ndarray, 
        post_size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        img_cv_resize = cv2.resize(img_cv, post_size, interpolation=cv2.INTER_LINEAR)
        fg_mask_resize = cv2.resize(fg_mask, post_size, interpolation=cv2.INTER_NEAREST)

        r = max(1, self.extractor.patch_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask_uint8 = fg_mask_resize.astype(np.uint8)

        dil = cv2.dilate(mask_uint8, kernel)
        ero = cv2.erode(mask_uint8, kernel)
        band = ~( (ero > 0) | (dil == 0) )

        refined_float = cv2.ximgproc.guidedFilter(
            guide=img_cv_resize, 
            src=mask_uint8.astype(np.float32), 
            radius=r, 
            eps=1e-3
        )
        mask_refined = refined_float > 0.5

        result = fg_mask_resize.copy().astype(bool)
        result[band] = mask_refined[band]
        return result

    def save_patch_similarity_maps_numpy(
        self, 
        fused_layer_feature: np.ndarray,
        feat_size: Tuple[int, int] = (32, 32), 
        save_size: Tuple[int, int] = (512, 512), 
        save_dir: Path = Path("similarity_maps")
    ):
        """fused_layer_feature: [B, H*W+1, C] の NumPy 配列 (B=1想定)"""
        save_dir = save_dir / "conf1"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. CLSトークン (0番目) を除外して L2 正規化 -> [H*W, C]
        patch_tokens = fused_layer_feature[0, 1:, :]
        norms = np.linalg.norm(patch_tokens, axis=-1, keepdims=True)
        patch_norm = patch_tokens / (norms + 1e-8)

        # 2. 全パッチペアのコサイン類似度を一括計算 (行列積) -> [H*W, H*W]
        sim_matrix = np.dot(patch_norm, patch_norm.T)

        # 解析用に生の類似度行列データを .npy 形式で保存
        np.save(save_dir / "similarity_matrix.npy", sim_matrix)

        W, H = feat_size
        SW, SH = save_size
        N = H * W

        # save_size に対する 1パッチあたりのスケール比率を算出
        patch_w = SW / W
        patch_h = SH / H

        print(
            f"全 {N} 個のパッチに対する類似度マップの生成を開始します... (保存先: {save_dir})"
        )

        # 3. 各パッチを基準（クエリ）として類似度マップを可視化・保存
        for i in range(N):
            y, x = divmod(i, W)

            # i番目のパッチに対する全領域の類似度マップ -> [H, W]
            sim_map = sim_matrix[i].reshape(H, W)

            # 0.0 ~ 1.0 に Min-Max 正規化して 8bit 画像化
            sim_min, sim_max = sim_map.min(), sim_map.max()
            sim_normalized = (sim_map - sim_min) / (sim_max - sim_min + 1e-8)
            sim_uint8 = (sim_normalized * 255).astype(np.uint8)

            # JET カラーマップを適用 (類似度高: 赤, 低: 青)
            heatmap = cv2.applyColorMap(sim_uint8, cv2.COLORMAP_JET)

            # save_size (SW, SH) にサイズを変更 (ドット感を保持)
            heatmap_large = cv2.resize(
                heatmap,
                (SW, SH),
                interpolation=cv2.INTER_NEAREST,
            )

            # save_size 上の基準パッチ座標（x1, y1, x2, y2）を算出
            x1 = int(round(x * patch_w))
            y1 = int(round(y * patch_h))
            x2 = int(round((x + 1) * patch_w)) - 1
            y2 = int(round((y + 1) * patch_h)) - 1

            # 基準（クエリ）となっているパッチの位置を白枠で強調表示
            cv2.rectangle(
                heatmap_large,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            # 画像出力
            filename = f"sim_patch_y{y:02d}_x{x:02d}_idx{i:04d}.png"
            cv2.imwrite(save_dir / filename, heatmap_large)

        print("保存が完了しました。")

    def run(
        self,
        image_path: str,
        output_dir: str = "./outputs/cluster",
    ):
        img_path = Path(image_path)
        if img_path.exists():
            print(f"====== Processing Image: {img_path.name} ======")
        else:
            print(f"====== 存在しません: {img_path.name} ======")
            return

        save_dir = Path(output_dir) / img_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        # 画像読み込み
        image_raw, is_bgr = self.extractor.load_image(image_path)
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        # 前処理
        image_tensor = self.extractor.preprocess_image(image_raw, is_bgr)

        # 特徴量抽出
        with Timer("特徴量抽出処理時間"):
            image_features, patch_features, cls2patch_attn, fused_layer_feature = (
                self.extractor.extract_image_features(image_tensor)
            )

        # クラス分類
        text_probs = (100.0 * image_features @ self.fgbg_text_features.T)
        text_probs = self._softmax(text_probs, axis=-1)

        # クラスタリング
        with Timer("クラスタリング処理時間"):
            fused_np = fused_layer_feature[:, 1:, :]
            fused_np = fused_np.reshape(self.feature_size * self.feature_size, -1)
            fused_cluster_labels = self.fit_predict_kmeans(fused_np, k=self.k_clusters, niter=self.niter)

        # クラスタごとのCLS特徴量のリスト
        with Timer("クラスタごとのCLSトークン処理時間"):
            cluster_image_features_list: List[np.ndarray] = self.extractor.extract_cluster_tokens(
                patch_features, fused_cluster_labels
            )

        # 前景度合いを計算
        with Timer("前景度合い"):
            foreground_threshold = 0.5
            fg_prompts_num: int = len(self.foreground_prompts)
            bg_prompts_num: int = len(self.background_prompts)
            foreground_mask = np.zeros_like(fused_cluster_labels, dtype=np.int32)
            foreground_map = np.zeros_like(fused_cluster_labels, dtype=np.float32)
            for label_idx, cluster_image_features in enumerate(cluster_image_features_list):
                cluster2text_probs = (
                    100.0 * cluster_image_features.reshape(1, -1) @ self.fgbg_text_features.T
                )
                cluster2text_probs = self._softmax(cluster2text_probs, axis=-1)
                # 前景度合い
                foreground_confidence = np.sum(cluster2text_probs[0, :fg_prompts_num])
                # 前景度合いのマップ
                foreground_map[fused_cluster_labels == label_idx] = foreground_confidence
                # 閾値以上の場合は前景と判断
                if foreground_confidence > foreground_threshold:
                    foreground_mask[fused_cluster_labels == label_idx] = 1

        with Timer("後処理時間"):
            foreground_mask = self.postprocess_mask(
                image_raw, 
                foreground_mask.reshape(self.feature_size, self.feature_size), 
                (self.extractor.target_size, self.extractor.target_size)
            )

        elapsed_time = time.time() - start
        print(f"経過時間    : {elapsed_time * 1000:.2f} ms")

        # レイヤー結合した特徴マップでもクラスタリング結果を保存
        fused_cluster_labels_2d = fused_cluster_labels.reshape(self.feature_size, self.feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused.png")

        # 前景度合いを保存
        foreground_map_2d = (
            (foreground_map * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        map_image = Image.fromarray(foreground_map_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        map_image.save(save_dir / "foreground_map.png")

        # マスク画像生成・保存
        mask_2d = (
            (foreground_mask * 255)
            .astype(np.uint8)
            # .reshape(self.feature_size, self.feature_size)
        )
        mask_image = Image.fromarray(mask_2d).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask_image.save(save_dir / "foreground_mask.png")

        # 
        self.save_patch_similarity_maps_numpy(
            fused_layer_feature, 
            (self.feature_size, self.feature_size), 
            (width, height),
            save_dir
        )

        # アテンション画像を保存
        attn_2d = (
            (cls2patch_attn * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        attn_image = Image.fromarray(attn_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        attn_image.save(save_dir / "attention_map.png")

        # オーバーレイ画像保存
        if is_bgr:
            image_cv = image_raw
        else:
            image_cv = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_image)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = mask_cv
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)

        # --- 処理後の画像保存処理（カラー化とリサイズを保存時に実行） ---
        print(f"Saving results : {save_dir}")
        print(f"=============================================\n")