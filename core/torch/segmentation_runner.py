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
from core.torch.edgetam_runner import EdgeTAMPredictor
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

        self.edgetam_predictor = EdgeTAMPredictor()

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

        # 行列積で類似度を計算し、[0, 1] の範囲に正規化
        cls2patch_sim = (cls_token @ patch_tokens.T).squeeze(0)
        cls2patch_sim = (cls2patch_sim + 1.0) / 2.0

        # 類似度上位K個のパッチ（背景バイアス）の平均を引いてセンタリング
        K = 10
        topk_indices = np.argsort(cls2patch_sim)[::-1][:K]
        # 類似度上位K個のパッチの特徴量平均（背景成分）を算出
        bg_bias = np.mean(patch_tokens[topk_indices, :], axis=0, keepdims=True)
        # 全特徴量（CLS + 全パッチ）から背景バイアスを引いて再正規化
        centered_features = patch_tokens - bg_bias * 0.5
        centered_features = centered_features / np.linalg.norm(centered_features, axis=-1, keepdims=True)
        # クラスタリング
        centered_cluster_labels = self.fit_predict_kmeans(
            centered_features, 
            k=self.k_clusters, 
            niter=self.niter
        )
        fused_cluster_labels = self.fit_predict_kmeans(
            patch_tokens, 
            k=self.k_clusters, 
            niter=self.niter
        )
        # 最終的に前景/背景の分離に使用するクラスタ画像
        cluster_labels = fused_cluster_labels

        # クラスタごとの特徴量トークン取得
        label_token_list: List[torch.Tensor] = self.extractor.extract_cluster_tokens(
            patch_features, torch.from_numpy(cluster_labels).to(torch.int64)
        )

        # 前景マスクの抽出
        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_mask = np.zeros_like(cluster_labels, dtype=np.int32)
        foreground_map = np.zeros_like(cluster_labels, dtype=np.float32)
        for label_idx, label_features in enumerate(label_token_list):
            label2text_probs = (
                100.0 * label_features.reshape(1, -1) @ self.fgbg_text_features.T
            ).softmax(dim=-1)
            probs_cpu = label2text_probs[0].cpu().numpy()
            # 前景度合い
            foreground_confidence = np.sum(probs_cpu[:fg_prompts_num])
            # 前景度合いのマップ
            foreground_map[cluster_labels == label_idx] = foreground_confidence
            # 閾値以上の場合は前景と判断
            if foreground_confidence > self.foreground_threshold:
                foreground_mask[cluster_labels == label_idx] = 1

        # # edgeTAMによるセグメンテーション
        # masks, scores, logits = self.edgetam_predictor.predict(image_raw, point_coords=None, point_labels=None)

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time:.3f} sec")

        # --- 処理後の画像保存処理（カラー化とリサイズを保存時に実行） ---
        print(f"\n--- Saving results to {save_dir} ---")

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
        fused_cluster_labels_2d = fused_cluster_labels.reshape(feature_size, feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused_01_ori.png")

        # レイヤー結合した特徴マップでもクラスタリング結果を保存
        fused_cluster_labels_2d = centered_cluster_labels.reshape(feature_size, feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused_02_center.png")

        # fused特徴マップのCLSとパッチの類似度
        fused_cls2patch_2d = (
            (cls2patch_sim * 255)
            .astype(np.uint8)
            .reshape(feature_size, feature_size)
        )
        fused_cls2patch_map_image = Image.fromarray(fused_cls2patch_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        fused_cls2patch_map_image.save(save_dir / "fused_cls2patch_image.png")

        # fused特徴マップのCLSとパッチの上位K個のマスク
        cls2patch_mask = cls2patch_sim >= topk_indices[K-1]
        fused_cls2patch_mask_2d = (
            (cls2patch_mask * 255)
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


import io
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.chart import LineChart, Reference
from openpyxl.drawing.image import Image as DrawImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import seaborn as sns
import torch

def export_complete_vit_analysis_to_excel(
    layer_features, H, W, output_filename="vit_layer_analysis_complete.xlsx"
):
    """バイオリンプロット画像、Excel純正折れ線グラフ、サマリー表、正方形2Dマップを1つの.xlsxに出力"""
    num_layers = len(layer_features)
    num_patches = H * W

    # --- 1. ノルム計算とパーセンタイル算出 ---
    all_norms = []  # [12, HW+1] (CLS含む)
    patch_norms_list = []  # [12, HW] (CLS除外)

    percentiles = {
        "P50": [],
        "P90": [],
        "P99": [],
        "P99.9": [],
        "Max": [],
    }

    for feat in layer_features:
        norms = torch.norm(feat[0], p=2, dim=-1).cpu().numpy()
        all_norms.append(norms)

        p_norms = norms[1:]  # CLS除外
        patch_norms_list.append(p_norms)

        percentiles["P50"].append(float(np.percentile(p_norms, 50)))
        percentiles["P90"].append(float(np.percentile(p_norms, 90)))
        percentiles["P99"].append(float(np.percentile(p_norms, 99)))
        percentiles["P99.9"].append(float(np.percentile(p_norms, 99.9)))
        percentiles["Max"].append(float(np.max(p_norms)))

    all_norms = np.array(all_norms)

    # --- 2. Matplotlibでバイオリンプロットを作成しメモリ上に保存 ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.violinplot(
        data=patch_norms_list,
        inner=None,
        color="#e0e0e0",
        linewidth=1,
        cut=0,
        ax=ax,
    )

    x = np.arange(num_layers)
    ax.plot(x, percentiles["P50"], marker="o", color="#1f77b4", label="P50")
    ax.plot(x, percentiles["P90"], marker="s", color="#2ca02c", label="P90")
    ax.plot(x, percentiles["P99"], marker="^", color="#ff7f0e", label="P99")
    ax.plot(x, percentiles["P99.9"], marker="d", color="#d62728", label="P99.9")
    ax.plot(
        x,
        percentiles["Max"],
        marker="x",
        color="#8c564b",
        linestyle="--",
        label="Max",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}" for i in range(num_layers)])
    ax.set_title(
        "Layer-wise Patch Norm Distribution (Violin + Percentiles)",
        fontweight="bold",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Patch Norm")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    # 画像をファイルに保存せずバイト列として取得
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)

    # --- 3. OpenPyXL Workbookの初期化 ---
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    font_small = Font(name="Segoe UI", size=8)
    font_bold_small = Font(name="Segoe UI", size=8, bold=True)
    font_header = Font(name="Segoe UI", size=9, bold=True, color="FFFFFF")
    fill_header = PatternFill(
        start_color="366092", end_color="366092", fill_type="solid"
    )
    fill_cls = PatternFill(
        start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"
    )
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    color_scale = ColorScaleRule(
        start_type="min",
        start_color="63BE7B",
        mid_type="percentile",
        mid_value=50,
        mid_color="FFEB84",
        end_type="max",
        end_color="F8696B",
    )

    # ==========================================
    # シート1: 分布・グラフシート (プロット1)
    # ==========================================
    ws_dist = wb.create_sheet(title="Norm_Distribution")
    ws_dist.views.sheetView[0].showGridLines = True

    # バイオリンプロット画像を挿入 (セル A1)
    img = DrawImage(img_buf)
    img.width = 650
    img.height = 325
    ws_dist.add_image(img, "A1")

    # パーセンタイル集計表の書き込み (セル A18 から)
    start_row_pct = 18
    pct_headers = ["Layer"] + list(percentiles.keys())
    for c_idx, h in enumerate(pct_headers, 1):
        cell = ws_dist.cell(row=start_row_pct, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center")

    for l_idx in range(num_layers):
        r = start_row_pct + 1 + l_idx
        ws_dist.cell(row=r, column=1, value=f"L{l_idx+1}")
        for c_idx, key in enumerate(percentiles.keys(), 2):
            val_cell = ws_dist.cell(
                row=r, column=c_idx, value=percentiles[key][l_idx]
            )
            val_cell.number_format = "0.00"
            val_cell.border = thin_border
            val_cell.font = font_small

    # Excel純正の折れ線グラフを追加 (集計表から動的生成)
    chart = LineChart()
    chart.title = "Percentile Spikes Across Layers (Excel Native Chart)"
    chart.style = 13
    chart.y_axis.title = "Patch Norm"
    chart.x_axis.title = "Layer"

    data_ref = Reference(
        ws_dist,
        min_col=2,
        min_row=start_row_pct,
        max_col=6,
        max_row=start_row_pct + num_layers,
    )
    cats_ref = Reference(
        ws_dist,
        min_col=1,
        min_row=start_row_pct + 1,
        max_row=start_row_pct + num_layers,
    )

    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)
    chart.width = 14
    chart.height = 8

    # 折れ線グラフを H1 に挿入
    ws_dist.add_chart(chart, "H1")

    # ==========================================
    # シート2: 全層サマリー
    # ==========================================
    ws_summary = wb.create_sheet(title="Norm_Summary")
    ws_summary.views.sheetView[0].showGridLines = True

    headers = ["Index", "Type"] + [f"Layer_{i+1}" for i in range(num_layers)]
    ws_summary.append(headers)

    for col_num in range(1, len(headers) + 1):
        cell = ws_summary.cell(row=1, column=col_num)
        cell.font = font_header
        cell.fill = fill_header

    for p_idx in range(num_patches + 1):
        row_num = p_idx + 2
        token_type = "CLS" if p_idx == 0 else f"Patch_{p_idx-1}"
        row_data = [p_idx, token_type] + [
            float(all_norms[l, p_idx]) for l in range(num_layers)
        ]
        ws_summary.append(row_data)

        if p_idx == 0:
            for c in range(1, len(row_data) + 1):
                cell = ws_summary.cell(row=row_num, column=c)
                cell.fill = fill_cls
                cell.font = font_bold_small

    for r in range(2, num_patches + 3):
        for c in range(3, num_layers + 3):
            cell = ws_summary.cell(row=r, column=c)
            cell.font = font_small
            cell.number_format = "0.00"
            cell.border = thin_border

    for col in range(3, num_layers + 3):
        col_letter = openpyxl.utils.get_column_letter(col)
        ws_summary.conditional_formatting.add(
            f"{col_letter}3:{col_letter}{num_patches+2}", color_scale
        )

    # ==========================================
    # シート3: 各層 2D Map (正方形・小さめセル)
    # ==========================================
    SQUARE_ROW_HEIGHT = 20.0
    SQUARE_COL_WIDTH = 5.2

    for l in range(num_layers):
        ws_2d = wb.create_sheet(title=f"L{l+1}_2D_Map")
        ws_2d.views.sheetView[0].showGridLines = True

        ws_2d.cell(
            row=1, column=1, value=f"L{l+1} CLS Norm:"
        ).font = font_bold_small
        cls_cell = ws_2d.cell(row=1, column=3, value=float(all_norms[l, 0]))
        cls_cell.font = font_bold_small
        cls_cell.fill = fill_cls
        cls_cell.number_format = "0.00"

        patch_norms_2d = all_norms[l, 1:].reshape(H, W)
        start_row = 3

        for r in range(H):
            current_row = start_row + r
            ws_2d.row_dimensions[current_row].height = SQUARE_ROW_HEIGHT

            for c in range(W):
                col_letter = openpyxl.utils.get_column_letter(c + 1)
                ws_2d.column_dimensions[col_letter].width = SQUARE_COL_WIDTH

                cell = ws_2d.cell(
                    row=current_row,
                    column=c + 1,
                    value=float(patch_norms_2d[r, c]),
                )
                cell.font = font_small
                cell.number_format = "0.00"
                cell.border = thin_border
                cell.alignment = Alignment(
                    horizontal="center", vertical="center"
                )

        end_col_letter = openpyxl.utils.get_column_letter(W)
        ws_2d.conditional_formatting.add(
            f"A{start_row}:{end_col_letter}{start_row + H - 1}", color_scale
        )

    wb.save(output_filename)
    print(f"全解析データ集約Excelを出力しました: {output_filename}")
