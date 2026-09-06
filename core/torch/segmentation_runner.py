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
from core.utils import *

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
            # sim_min, sim_max = sim_map.min(), sim_map.max()
            # sim_normalized = (sim_map - sim_min) / (sim_max - sim_min + 1e-8)
            sim_normalized = (sim_map + 1) / 2
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
        feature_size = self.extractor.feature_size

        img_path = Path(image_path)
        mask_path = img_path.with_name(img_path.name.replace(".png", "_mask.png"))

        if img_path.exists():
            # 画像読み込みと前処理
            image_raw = Image.open(img_path).convert("RGB")
            # dx, dy = 200, 200
            # bg_color = (255, 255, 255)  # 余白の背景色 (RGB)
            # shifted_img = Image.new(image_raw.mode, image_raw.size, bg_color)
            # shifted_img.paste(image_raw, (dx, dy))
            # image_raw = shifted_img
            width, height = image_raw.size
            print(f"\n--- Processing Image: {img_path.name} ---")
        else:
            print(f"\n--- 存在しません: {img_path.name} ---")
            return
        if mask_path.exists():
            mask_raw = Image.open(mask_path).convert("L")
            # mask_raw = mask_raw.resize((width, height), resample=Image.Resampling.NEAREST)
            # white_bg = Image.open("/home/llm/datasets/val2017/val2017/000000000285.jpg").convert("RGB")
            # white_bg = white_bg.resize((width, height), resample=Image.Resampling.BILINEAR)
            # image_raw = Image.composite(image_raw, white_bg, mask_raw)
            mask_tensor = np.array(mask_raw)
            mask_tensor = cv2.resize(mask_tensor, (feature_size, feature_size), interpolation=cv2.INTER_NEAREST)
            # mask_tensor = mask_tensor == 255
            # mask_tensor = cv2.resize(mask_tensor, (self.extractor.force_image_size, self.extractor.force_image_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_tensor = None
        
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

        # 指定レイヤーの融合とクラスタリング
        fused_layer_feature = torch.cat(layer_features[7:10], dim=2)
        fused_layer_feature = F.normalize(fused_layer_feature, dim=-1)
        fused_np = fused_layer_feature.numpy()
        fused_np = fused_np.reshape(feature_size * feature_size + 1, -1)
        fused_cluster_labels = self.fit_predict_kmeans(fused_np[1:, :].reshape(feature_size * feature_size, -1), k=self.k_clusters, niter=self.niter)
        # fused_np[1:, :][mask_tensor.reshape(-1) == 255, :] = fused_np[1:, :][mask_tensor.reshape(-1) == 255, :] - fused_np[:1, :]
        # fused_np_norm = (fused_np[1:, :] - fused_np[:1, :]) / np.linalg.norm(fused_np[1:, :] - fused_np[:1, :], axis=-1, keepdims=True)
        # fused_cluster_labels = self.fit_predict_kmeans(fused_np_norm.reshape(feature_size * feature_size, -1), k=self.k_clusters, niter=self.niter)

        # cls = fused_layer_feature[:, 0, :]       # [B, C]
        # patch = fused_layer_feature[:, 1:, :]    # [B, P, C]
        # denom = (cls * cls).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # coeff = (
        #     patch * cls[:, None, :]
        # ).sum(dim=-1, keepdim=True) / denom[:, None, :]
        # residual = patch - coeff * cls[:, None, :]
        # residual = F.normalize(residual, dim=-1)
        # residual = residual.numpy()
        # fused_cluster_labels = self.fit_predict_kmeans(residual.reshape(feature_size * feature_size, -1), k=self.k_clusters, niter=self.niter)

        # bg_mask = torch.where(torch.from_numpy(mask_tensor).reshape(1, -1) < 127, 1, 0)
        # residual = remove_background_direction(fused_layer_feature, bg_mask.bool()).numpy()
        # fused_cluster_labels = self.fit_predict_kmeans(residual[:, 1:, :].reshape(feature_size * feature_size, -1), k=self.k_clusters, niter=self.niter)

        # CLSトークンとパッチごとの類似度
        fused_cls2patch_sim = fused_np[:1, :] @ fused_np[1:, :].T
        fused_cls2patch_sim = (fused_cls2patch_sim + 1) / 2
        fused_cls2patch_sim = fused_cls2patch_sim.reshape(-1)

        # # クラスタごとの類似度
        # for i in range(self.k_clusters):
        #     fused_np_custom = fused_np[1:, :] - np.mean(fused_np[1:, :][fused_cluster_labels == i, :], axis=0, keepdims=True)
        #     fused_cluster_labels_custom = self.fit_predict_kmeans(fused_np_custom, k=self.k_clusters, niter=self.niter)
        #     fused_cluster_labels_2d = fused_cluster_labels_custom.reshape(feature_size, feature_size)
        #     fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        #     fused_cluster_rgb = fused_cluster_rgb.resize(
        #         (width, height), resample=Image.Resampling.NEAREST
        #     )
        #     fused_cluster_rgb.save(save_dir / f"kmeans_cluster{i:02d}.png")

        # クラスタごとの特徴量トークン取得
        label_token_list = self.extractor.extract_cluster_tokens(
            patch_features, torch.from_numpy(fused_cluster_labels).to(torch.int64)
        )

        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_mask = np.zeros_like(fused_cluster_labels, dtype=np.int32)
        foreground_map = np.zeros_like(fused_cluster_labels, dtype=np.float32)
        print("\n--- Cluster Probabilities ---")
        for label_idx, label_features in enumerate(label_token_list):
            label2text_probs = (
                100.0 * label_features.reshape(1, -1) @ self.fgbg_text_features.T
            ).softmax(dim=-1)
            probs_cpu = label2text_probs[0].cpu().numpy()
            # 前景度合い
            foreground_confidence = np.sum(probs_cpu[:fg_prompts_num])
            # 前景度合いのマップ
            foreground_map[fused_cluster_labels == label_idx] = foreground_confidence
            # 閾値以上の場合は前景と判断
            if foreground_confidence > self.foreground_threshold:
                foreground_mask[fused_cluster_labels == label_idx] = 1

        # 上位K個
        K = 20
        fused_cls2patch_mask = np.zeros_like(fused_cls2patch_sim)
        fused_cls2patch_mask[np.argsort(fused_cls2patch_sim)[::-1][:K]] = 1
        fused_np_2 = fused_np.copy()[1:, :]
        fused_np_2[foreground_mask == 0, :] = \
            fused_np_2[foreground_mask == 0, :] - np.mean(fused_np_2[fused_cls2patch_mask == 1, :], axis=0, keepdims=True) * 100
        fused_np_2 = fused_np_2 / np.linalg.norm(fused_np_2, axis=-1, keepdims=True)
        fused_cluster_labels2 = self.fit_predict_kmeans(fused_np_2, k=self.k_clusters, niter=self.niter)

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time:.3f} sec")

        # --- 処理後の画像保存処理（カラー化とリサイズを保存時に実行） ---
        print(f"\n--- Saving results to {save_dir} ---")

        # 各層のクラスタリング結果の画像を保存
        for i, layer_feature in enumerate(layer_features):
            lf_np = layer_feature.numpy()[:, 1:, :]
            lf_np = lf_np.reshape(feature_size * feature_size, -1)
            cluster_labels = self.fit_predict_kmeans(lf_np, k=self.k_clusters)
            cluster_labels_2d = cluster_labels.reshape(feature_size, feature_size)
            cluster_rgb = self.colorize_clusters_PIL(cluster_labels_2d)
            cluster_rgb = cluster_rgb.resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            cluster_rgb.save(save_dir / f"layer{i + 1:02d}.png")

        # レイヤー結合した特徴マップでもクラスタリング結果を保存
        fused_cluster_labels_2d = fused_cluster_labels.reshape(feature_size, feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused.png")

        # レイヤー結合した特徴マップでもクラスタリング結果を保存
        fused_cluster_labels2_2d = fused_cluster_labels2.reshape(feature_size, feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels2_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused2.png")

        # 
        # self.save_patch_similarity_maps_numpy(
        #     fused_layer_feature, 
        #     (self.feature_size, self.feature_size), 
        #     (width, height),
        #     save_dir
        # )

        # fused特徴マップのCLSとパッチの類似度
        fused_cls2patch_2d = (
            (fused_cls2patch_sim * 255)
            .astype(np.uint8)
            .reshape(feature_size, feature_size)
        )
        fused_cls2patch_map_image = Image.fromarray(fused_cls2patch_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        fused_cls2patch_map_image.save(save_dir / "fused_cls2patch_image.png")

        # fused特徴マップのCLSとパッチの上位K個のマスク
        fused_cls2patch_mask_2d = (
            (fused_cls2patch_mask * 255)
            .astype(np.uint8)
            .reshape(feature_size, feature_size)
        )
        fused_cls2patch_map_mask = Image.fromarray(fused_cls2patch_mask_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        fused_cls2patch_map_mask.save(save_dir / "fused_cls2patch_mask.png")

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
        mask_2d = (
            (foreground_mask * 255)
            .astype(np.uint8)
            .reshape(feature_size, feature_size)
        )
        mask_image = Image.fromarray(mask_2d).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask_image.save(save_dir / "foreground_mask.png")

        # オーバーレイ画像保存
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_image)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = mask_cv
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)

        print("Save completed.")


def make_background_prototype(
    features: torch.Tensor,
    bg_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    背景パッチ集合から、画像ごとの背景プロトタイプを作る。

    Args:
        features:
            [B, P+1, C]
            0番目がCLS、1番目以降がパッチ

        bg_mask:
            [B, P] のboolテンソル
            True  = 背景パッチ
            False = 前景パッチ

    Returns:
        bg_proto:
            [B, C]
    """
    if features.ndim != 3:
        raise ValueError(
            f"features must be [B, P+1, C], got {features.shape}"
        )

    if bg_mask.ndim != 2:
        raise ValueError(
            f"bg_mask must be [B, P], got {bg_mask.shape}"
        )

    B, P_plus_1, C = features.shape
    P = P_plus_1 - 1

    if bg_mask.shape != (B, P):
        raise ValueError(
            f"bg_mask must have shape {(B, P)}, "
            f"got {tuple(bg_mask.shape)}"
        )

    patch = features[:, 1:, :]  # [B, P, C]
    bg_mask = bg_mask.bool()     # [B, P]

    # 背景パッチ数: [B, 1]
    bg_count = bg_mask.sum(dim=1, keepdim=True)

    if (bg_count == 0).any():
        raise ValueError(
            "At least one sample has no background patch."
        )

    # 背景パッチを選択するための空間マスク
    mask = bg_mask.unsqueeze(-1).to(dtype=patch.dtype)  # [B, P, 1]

    # 各パッチを正規化してから平均する。
    # 背景パッチの特徴方向の平均を求める。
    patch_unit = F.normalize(
        patch,
        dim=-1,
        eps=eps,
    )  # [B, P, C]

    bg_mean = (
        (patch_unit * mask).sum(dim=1)
        / bg_count.to(dtype=patch.dtype).clamp_min(1.0)
    )  # [B, C]

    # 背景プロトタイプを単位ベクトルにする
    bg_proto = F.normalize(
        bg_mean,
        dim=-1,
        eps=eps,
    )  # [B, C]

    return bg_proto


def remove_background_direction(
    features: torch.Tensor,
    bg_mask: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    各パッチから、背景プロトタイプ方向の成分を除去する。

    Args:
        features:
            [B, P+1, C]
            0番目がCLS

        bg_mask:
            [B, P] のboolテンソル
            True  = 背景パッチ

        alpha:
            背景方向の除去強度。
            0.0: 補正なし
            1.0: 背景方向を完全除去

        eps:
            数値安定化用

    Returns:
        corrected_features:
            [B, P+1, C]

            CLSは元のまま。
            パッチのみ背景方向を補正。
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            f"alpha must be in [0, 1], got {alpha}"
        )

    cls = features[:, :1, :]    # [B, 1, C]
    patch = features[:, 1:, :]  # [B, P, C]

    bg_proto = make_background_prototype(
        features=features,
        bg_mask=bg_mask,
        eps=eps,
    )  # [B, C]

    # 背景プロトタイプ方向への射影係数
    # bg_protoは正規化済みなので分母は不要
    coeff = (
        patch * bg_proto[:, None, :]
    ).sum(dim=-1, keepdim=True)  # [B, P, 1]

    # 背景方向成分
    bg_component = (
        coeff * bg_proto[:, None, :]
    )  # [B, P, C]

    # 背景方向を部分的、または完全に除去
    corrected_patch = (
        patch - alpha * bg_component
    )  # [B, P, C]

    # CLSは元のまま戻す
    corrected_features = torch.cat(
        [cls, corrected_patch],
        dim=1,
    )  # [B, P+1, C]

    return corrected_features