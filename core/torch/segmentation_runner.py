import os
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import faiss
import torch
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
        niter: int=10
    ):
        self.extractor = extractor
        self.prompts_dir = prompts_dir
        self.k_clusters = k_clusters
        self.niter = niter

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

    def run(
        self,
        image_path: str,
        output_dir: str = "./outputs/cluster",
    ):
        img_path = Path(image_path)
        if img_path.exists():
            print(f"\n--- Processing Image: {img_path.name} ---")
        else:
            print(f"\n--- 存在しません: {img_path.name} ---")
            return
        
        save_dir = Path(output_dir) / img_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)
        
        feature_size = self.extractor.feature_size

        # 1. 画像読み込みと前処理
        image_raw = Image.open(img_path).convert("RGB")
        width, height = image_raw.size

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
        fused_np = fused_layer_feature.numpy()[:, 1:, :]
        fused_np = fused_np.reshape(feature_size * feature_size, -1)
        fused_cluster_labels = self.fit_predict_kmeans(fused_np, k=self.k_clusters, niter=self.niter)

        # クラスタごとの特徴量トークン取得
        label_token_list = self.extractor.extract_cluster_tokens(
            patch_features, torch.from_numpy(fused_cluster_labels).to(torch.int64)
        )
        # print(label_token_list.shape)

        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_threshold = 0.5
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
            if foreground_confidence > foreground_threshold:
                foreground_mask[fused_cluster_labels == label_idx] = 1

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

        # 前景度合いを保存
        print(foreground_map.max(), foreground_map.min())
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