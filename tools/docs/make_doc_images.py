"""説明資料（docs/）用の画像を作る: 1 枚の画像をパイプラインに通し、処理の段階ごとの中間結果を docs/images/ に保存する。

    python tools/docs/make_doc_images.py [--image PATH] [--out-dir docs/images] [--prefix coin] [--model-dir weights/openvino_models_l6]

画像は JPEG（幅 1200px 以下・品質 80）で保存する（git に載せるため。設定は shrink_images.py）。
--image の既定は configs/dataset.yaml の先頭の画像。--model-dir の既定は configs/ov_config.yaml の model_dir。
OpenVINO 版（configs/ov_config.yaml）で実行する。

本体（SegmentationRunner.segment）と同じ段階の関数を、同じ順に呼び出して中間結果を取り出す（スレッドは使わない）。
最後に、本体の segment（スレッドあり）と、粗いラベル・semantic_map・クラスの名前が完全に一致するか確かめる。

作る画像（<prefix>_<名前>.jpg）。名前の番号は docs/coarse_segmentation.md・docs/detailed_segmentation.md の節の番号:
  00_input        入力とパッチ格子                      01_features    PE と EdgeTAM の特徴（主成分）
  ① 粗い段階（core/common/coarse/）
  c1_kmeans       ① KMeans                             c2_judge / c2_confidence  ② クラスタの判定
  c3_propagation  ③ 確信度の伝播                        c3_foreground  伝播の前後の前景マスク
  c4_relabel      ④ 前景のクラスの付け直し（各段のラベル） c5_labels      ⑤ 粗いセマンティックラベル
  ② 詳細段階（core/common/detail/）
  d1_background   1 背景の負例点                        d2_components  2 連結成分・正例点・core
  d2_objectness   2 物体らしさ                          d2_prior / d2_prior_zoom  2 初期マスク
  d2_iterations   2 デコーダの反復                      d2_class_mask  2 クラスごとのマスクとスコア
  d3_overlap      3 重なりの解決                        d4_rescue      4 取りこぼしの再推定（作動したときだけ）
  d5_cleanup      5 後処理（小領域の除去・隙間埋め）     d6_lattice     6 格子の検出（作動の判定）
  d7_naming       7 最終の名前付け                      d8_final       最終結果
  overview        全体の 1 枚
"""
import argparse
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.common.coarse.clusters import classify_logits, judge_clusters, kmeans_labels
from core.common.coarse.propagation import edge_graph_of, propagate_confidence
from core.common.coarse.relabel import (
    has_split_object,
    merge_weak_classes,
    needs_relabel,
    object_components,
    rejudge_regions,
    relabel_foreground,
)
from core.common.detail import lattice as lattice_utils
from core.common.detail.mask_cleanup import fill_narrow_gaps, remove_small_regions
from core.common.detail.naming import final_regions
from core.common.detail.prompt_points import objectness_map, select_background_points
from core.common.detail.refiner import RefineState
from core.common.mask_utils import connected_components, dilate, guided_filter, patch_mean
from core.common.parallel import Workers
from core.common.params import (
    EDGE_PROPAGATION_ALPHA,
    EDGE_PROPAGATION_ITERATIONS,
    GUIDED_FILTER_EPS,
    LATTICE_DETECT_SIZE,
    LATTICE_MIN_LAG,
    LATTICE_MIN_STRENGTH,
)
from core.common.segmentation_runner import SegmentationRunner
from core.common.visualization import clean_prompt
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_dataset_paths, load_image, resize_image
from tools.docs.shrink_images import save_jpeg

DISPLAY = 512  # 表示用の一辺（モデル入力と同じ。1 パッチ = 16 画素）
FONTS = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic"]  # 日本語が出る最初のフォントを使う
CLUSTER_COLORS = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
], dtype=np.uint8)
_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((font for font in FONTS if font in _available_fonts), "sans-serif")


# =============================================================================
# 描画の小物
# =============================================================================

def to_display(array: np.ndarray, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    return cv2.resize(array, (DISPLAY, DISPLAY), interpolation=interpolation)


def heat(values: np.ndarray, vmin=0.0, vmax=1.0, cmap="viridis") -> np.ndarray:
    """2 次元の値を、表示サイズの RGB（uint8）にする（パッチは四角のまま拡大）。"""
    normalized = np.clip((values.astype(np.float32) - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgb = (plt.get_cmap(cmap)(normalized)[..., :3] * 255).astype(np.uint8)
    return to_display(rgb)


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    mask = to_display(mask.astype(np.uint8)).astype(bool)
    out = image.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def outline(image: np.ndarray, mask: np.ndarray, color=(255, 255, 0), thickness=2) -> np.ndarray:
    mask = to_display(mask.astype(np.uint8)) if mask.shape[0] != DISPLAY else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def draw_grid(image: np.ndarray, grid: tuple, color=(255, 255, 255), alpha=0.35) -> np.ndarray:
    out = image.copy()
    layer = image.copy()
    for i in range(1, grid[0]):
        cv2.line(layer, (0, i * DISPLAY // grid[0]), (DISPLAY, i * DISPLAY // grid[0]), color, 1)
    for j in range(1, grid[1]):
        cv2.line(layer, (j * DISPLAY // grid[1], 0), (j * DISPLAY // grid[1], DISPLAY), color, 1)
    return cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)


def patch_box(image: np.ndarray, x: int, y: int, grid: tuple, color, thickness=2) -> None:
    """パッチ (x, y) を四角で囲む（in-place）。"""
    ph, pw = DISPLAY // grid[0], DISPLAY // grid[1]
    cv2.rectangle(image, (x * pw, y * ph), ((x + 1) * pw - 1, (y + 1) * ph - 1), color, thickness)


def pca_rgb(features: np.ndarray, grid: tuple) -> np.ndarray:
    """(N, C) の特徴の主成分 3 つを RGB にして、パッチ格子の画像（表示サイズ）にする。"""
    centered = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vt[:3].T
    lo, hi = np.percentile(components, 2, axis=0), np.percentile(components, 98, axis=0)
    rgb = (np.clip((components - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
    return to_display(rgb.reshape(*grid, 3))


def points_on(image: np.ndarray, coords, labels, scale: float, radius=7) -> np.ndarray:
    """プロンプト点（元画像座標）を描く。正例=緑、負例=赤。"""
    out = image.copy()
    for (x, y), label in zip(coords, labels):
        center = (int(x * scale), int(y * scale))
        cv2.circle(out, center, radius + 2, (255, 255, 255), -1)
        cv2.circle(out, center, radius, (0, 200, 0) if label == 1 else (230, 0, 0), -1)
    return out


def patch_dot(image: np.ndarray, x: float, y: float, grid: tuple, color=(255, 255, 0), radius=6) -> None:
    """パッチ座標 (x, y)（整数がパッチの中心）に点を描く（in-place）。"""
    center = (int((x + 0.5) * DISPLAY / grid[1]), int((y + 0.5) * DISPLAY / grid[0]))
    cv2.circle(image, center, radius, color, -1)
    cv2.circle(image, center, radius, (0, 0, 0), 1)


def save_figure(fig, path: Path, **kwargs) -> None:
    """図を JPEG（幅の上限つき。git に載せるため小さくする）で保存する。設定は shrink_images.py。"""
    from PIL import Image
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, **kwargs)
    buffer.seek(0)
    save_jpeg(Image.open(buffer), path)


def save_panels(path: Path, panels, ncols=None, size=3.4, wspace=0.04, legend: Optional[Sequence[Tuple[str, np.ndarray]]] = None) -> None:
    """(タイトル, RGB 画像) のリストを格子状に並べて 1 枚の図にする。legend = [(名前, RGB の色), ...] なら、下に凡例を付ける。"""
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    legend_rows = -(-len(legend) // 3) if legend else 0
    height = (size + 0.35) * nrows + 0.3 * legend_rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, height), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
    bottom = 0.01 + 0.3 * legend_rows / height
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.35 / height * 1.1, bottom=bottom, wspace=wspace, hspace=0.18)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.asarray(color) / 255) for _, color in legend]
        fig.legend(handles, [name for name, _ in legend], loc="lower center", ncol=min(len(legend), 3), fontsize=9, frameon=False)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


def tab_colors(count: int, cmap="tab10") -> np.ndarray:
    """count 個の見分けやすい色 (count, 3) uint8。"""
    return (plt.get_cmap(cmap)(np.arange(count) % plt.get_cmap(cmap).N)[:, :3] * 255).astype(np.uint8)


def label_view(base: np.ndarray, patch_values: np.ndarray, colors: dict, alpha=0.62) -> np.ndarray:
    """パッチ格子の値（colors のキーの値だけ着色。他は着色しない）を、入力画像に重ねる。"""
    colored = np.zeros((*patch_values.shape, 3), np.uint8)
    painted = np.zeros(patch_values.shape, bool)
    for value, color in colors.items():
        colored[patch_values == value] = color
        painted |= patch_values == value
    big = to_display(colored)
    mask = to_display(painted.astype(np.uint8)).astype(bool)
    out = base.copy()
    out[mask] = ((1 - alpha) * base[mask] + alpha * big[mask]).astype(np.uint8)
    return out


def class_view(base: np.ndarray, class_map: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """クラス番号のマップ（0 = 背景、1~ = クラス。任意の大きさ）を、入力画像に重ねる。colors[k - 1] がクラス k の色。"""
    small = cv2.resize(class_map.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    colored = colors[np.clip(small.astype(int) - 1, 0, None) % len(colors)]
    return np.where((small > 0)[..., None], cv2.addWeighted(base, 0.45, colored, 0.55, 0), base)


def diff_view(base: np.ndarray, before: np.ndarray, after: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """after のクラスマップに、before から変わった画素を赤で重ねる（作業解像度のまま比べ、表示サイズへ拡大）。"""
    view = class_view(base, after, colors)
    changed = cv2.resize((before != after).astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    changed = cv2.dilate(changed, np.ones((9, 9), np.uint8)).astype(bool)  # 数画素の変化でも見えるよう、太く描く
    view[changed] = (255, 0, 0)
    return view


def show_table(path: Path, header: List[str], rows: List[List[str]], widths: List[float], row_labels: Optional[List[str]] = None, emphasize=None, color_col=None, fontsize=10) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 0.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, rowLabels=row_labels, loc="center", cellLoc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)
    for i in range(len(rows)):
        if emphasize and emphasize[i]:
            for j in range(len(header)):
                table[i + 1, j].set_text_props(fontweight="bold")
        if color_col is not None and color_col[i] is not None:
            table[i + 1, len(header) - 1].set_facecolor(tuple(np.asarray(color_col[i]) / 255))
    save_figure(fig, path, bbox_inches="tight")
    plt.close(fig)
    print("保存:", path)


# =============================================================================
# 段階ごとの画像
# =============================================================================

def stage_input(S) -> None:
    """前処理（リサイズ・パッチ格子）と特徴量。"""
    S.save_panels("00_input", [
        (f"元画像（{S.width}×{S.height}）", cv2.resize(S.image_raw, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)),
        (f"モデル入力（{S.resized.shape[1]}×{S.resized.shape[0]}）+ パッチ格子 {S.grid[1]}×{S.grid[0]}", draw_grid(S.base, S.grid)),
    ])
    S.edgetam_features = S.session.encode(S.resized)
    S.patch_features, S.fused_patch_features = S.coarse.extract_features(S.resized)
    S.edge_embed = S.edgetam_features[2]
    S.pe_tokens = S.fused_patch_features[0, 1:, :]
    edge_flat = S.edge_embed[0].reshape(S.edge_embed.shape[1], -1).T
    print("PE トークン", S.pe_tokens.shape, "EdgeTAM image_embed", S.edge_embed.shape)
    S.save_panels("01_features", [
        ("入力", S.base),
        ("PE のパッチ特徴（主成分 → RGB）", pca_rgb(S.pe_tokens, S.grid)),
        (f"EdgeTAM の image_embed（{S.edge_embed.shape[2]}×{S.edge_embed.shape[3]}）", pca_rgb(edge_flat, S.edge_embed.shape[2:])),
    ])


def stage_coarse(S) -> None:
    """① 粗い段階: ① KMeans → ② 判定 → ③ 伝播 → ④ 付け直し → ⑤ ラベル。本体の generate_foreground_map と同じ順に呼ぶ。"""
    coarse, prompts, grid, threshold = S.coarse, S.coarse.prompts, S.grid, S.coarse.foreground_threshold
    num_fg = prompts.num_foreground
    all_prompts = prompts.all

    # ① KMeans
    pe_labels = kmeans_labels(S.pe_tokens, k=coarse.k_clusters, niter=coarse.niter)
    cluster_map = to_display(CLUSTER_COLORS[pe_labels.reshape(grid)])
    S.save_panels("c1_kmeans", [
        ("入力", S.base),
        (f"KMeans（k={coarse.k_clusters}）のクラスタ", cluster_map),
        ("重ね合わせ", cv2.addWeighted(S.base, 0.45, cluster_map, 0.55, 0)),
    ])
    S.cluster_map = cluster_map

    # ② クラスタの判定
    confidence0, prompt0, cluster_logits = judge_clusters(S.extractor, prompts, threshold, S.patch_features, pe_labels)
    tokens = S.extractor.extract_cluster_tokens(S.patch_features, pe_labels)
    rows, emphasize, colors = [], [], []
    for index, token in enumerate(tokens):
        logits = prompts.logits(token)
        fg_best, bg_best = int(np.argmax(logits[:num_fg])), int(np.argmax(logits[num_fg:]))
        confidence, prompt = classify_logits(logits, num_fg, threshold)
        is_fg = confidence > threshold
        rows.append([str(int((pe_labels == index).sum())), f"{logits[fg_best]:.1f}", f"{logits[num_fg + bg_best]:.1f}", f"{confidence:.3f}",
                     "前景" if is_fg else "背景", clean_prompt(all_prompts[prompt])])
        emphasize.append(is_fg)
        colors.append(CLUSTER_COLORS[index])
        print("  cluster", index, "n=%s fg=%s(%s) bg=%s(%s) conf=%.3f -> %s" % (
            rows[-1][0], rows[-1][1], clean_prompt(all_prompts[fg_best]), rows[-1][2], clean_prompt(all_prompts[num_fg + bg_best]), confidence, rows[-1][5]))
    show_table(S.path("c2_judge"), ["パッチ数", "前景の最大類似度", "背景の最大類似度", "前景確信度", "判定", "ラベル（最も近いプロンプト）"], rows,
               [0.1, 0.16, 0.16, 0.13, 0.08, 0.37], [f"クラスタ{i}" for i in range(len(rows))], emphasize, colors)
    S.save_panels("c2_confidence", [
        ("クラスタの判定を各パッチへ（前景確信度）", heat(confidence0.reshape(grid))),
        (f"前景確信度 > {threshold}", overlay(S.base, (confidence0 > threshold).reshape(grid), alpha=0.55)),
    ])

    # ③ EdgeTAM の特徴での確信度の伝播
    graph = edge_graph_of(S.edge_embed)
    propagated, prompt1 = propagate_confidence(confidence0, prompt0, graph, num_fg, threshold)
    change = np.abs(propagated - confidence0)
    flipped = (confidence0 > threshold) != (propagated > threshold)
    query = int(np.argmax(np.where(flipped, change, -1))) if flipped.any() else int(np.argmax(change))  # 近傍の例: しきい値をまたいだパッチのうち、変化が最大のもの
    qy, qx = divmod(query, grid[1])
    neighbor_view = draw_grid(S.base, grid)
    for n in graph.indices[graph.indptr[query]:graph.indptr[query + 1]]:
        patch_box(neighbor_view, int(n % grid[1]), int(n // grid[1]), grid, (255, 230, 0), 2)
    patch_box(neighbor_view, qx, qy, grid, (255, 0, 0), 3)
    print("近傍の例のパッチ", (qx, qy), "conf0=%.3f -> %.3f" % (confidence0[query], propagated[query]), "反転したパッチ数", int(flipped.sum()), "（前景に %d・背景に %d）" % (
        int(((propagated > threshold) & ~(confidence0 > threshold)).sum()), int(((propagated <= threshold) & (confidence0 > threshold)).sum())))
    S.save_panels("c3_propagation", [
        ("PE の判定（前景確信度）", heat(confidence0.reshape(grid))),
        ("赤: 注目パッチ / 黄: EdgeTAM 特徴の近傍（kNN）", neighbor_view),
        ("伝播後の前景確信度", heat(propagated.reshape(grid))),
        ("確信度の変化（|伝播後 − 伝播前|）", heat(change.reshape(grid), 0, max(float(change.max()), 1e-3), "magma")),
    ], ncols=4, size=3.2)
    S.save_panels("c3_foreground", [
        ("伝播の前（② の判定）", overlay(S.base, (confidence0 > threshold).reshape(grid), alpha=0.55)),
        ("伝播の後（粗い前景マスク）", overlay(S.base, (propagated > threshold).reshape(grid), alpha=0.55)),
    ], size=4.2)

    # ④ 前景のクラスの付け直し
    foreground = propagated > threshold
    components = object_components(foreground, grid)
    stages = [("③ の後\n（クラスタ由来の名前）", prompt1)]
    blobs = None
    if needs_relabel(prompt1, components):
        blobs, blob_logits = coarse.relabeler.foreground_blocks(foreground, S.patch_features, S.pe_tokens, S.edge_embed)
        a = rejudge_regions(prompt1, components, cluster_logits[pe_labels] + blob_logits[blobs], coarse.k_clusters)
        b = merge_weak_classes(a, foreground.astype(np.int64), blobs, blob_logits)
        c = relabel_foreground(b, components, blobs, blob_logits, graph, EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS) if has_split_object(b, components) else b
        stages += [("(a) 名前の判定し直し", a), ("(b) 小さいクラスの統合", b), ("(c) 同じ物体の中で伝播", c)]
        final_prompt = c
    else:
        stages.append(("付け直し後", prompt1))
        final_prompt = prompt1
    present = sorted({int(p) for _, values in stages for p in np.unique(values[foreground])})
    prompt_colors = {p: tab_colors(max(len(present), 1), "tab10")[i] for i, p in enumerate(present)}
    panels, previous = [], None
    for title, values in stages:
        changed = "" if previous is None else f"\n変わったパッチ {int((values[foreground] != previous[foreground]).sum())}"
        panels.append((title + changed, label_view(S.base, np.where(foreground.reshape(-1), values, -1).reshape(grid), prompt_colors)))
        previous = values
    if blobs is not None:
        blob_colors = {k: tab_colors(32, "tab20")[k % 20] for k in range(int(blobs.max()) + 1)}
        panels.insert(1, (f"前景のブロック\n（KMeans {int(blobs.max()) + 1} 個）", label_view(S.base, np.where(foreground, blobs, -1).reshape(grid), blob_colors, 0.7)))
    S.save_panels("c4_relabel", panels, ncols=len(panels), size=3.0,
                  legend=[(clean_prompt(all_prompts[p]), prompt_colors[p]) for p in present])
    print("④ 付け直し: 領域", len({(int(components[i]), int(prompt1[i])) for i in np.flatnonzero(foreground)}), "個",
          "→ 名前", [clean_prompt(all_prompts[p]) for p in present])

    # 本体の generate_foreground_map と一致を確認する
    generated = coarse.generate_foreground_map(S.patch_features, S.fused_patch_features, S.edge_embed, graph)
    assert np.allclose(generated.foreground_map.reshape(-1), propagated, atol=1e-5), "伝播の再現が本体と合わない"
    assert np.array_equal(np.asarray(generated.cluster_index_list)[generated.cluster_labels], final_prompt), "付け直しの再現が本体と合わない"
    S.generated = generated
    S.foreground_map, S.foreground_mask = generated.foreground_map, generated.foreground_mask

    # ⑤ ラベルの付与
    S.labels_list, S.semantic_labels = coarse.assign_semantic_labels(generated.cluster_labels, generated.cluster_index_list)
    label_colors = tab_colors(len(S.labels_list), "tab20")
    print("ラベル:", [clean_prompt(all_prompts[i]) for i in S.labels_list])
    names = [("前景: " if S.labels_list[i] < num_fg else "背景: ") + clean_prompt(all_prompts[S.labels_list[i]]) for i in range(len(S.labels_list))]
    S.save_panels("c5_labels", [(f"粗いセマンティックラベル（{len(S.labels_list)} 種）", to_display(label_colors[S.semantic_labels]))],
                  size=4.6, legend=[(n, label_colors[i]) for i, n in enumerate(names)])


def stage_detail(S) -> None:
    """② 詳細段階: 本体の SemanticRefiner.refine と同じ順に、段階の関数を呼ぶ。"""
    refiner, estimator, grid = S.refiner, S.estimator, S.grid
    base = S.base
    foreground_binary = S.foreground_mask != 0
    semantic_ids = np.unique(S.semantic_labels[foreground_binary])
    num_classes = len(semantic_ids)
    prompt_indices = [int(S.labels_list[i]) for i in semantic_ids]
    all_prompts = S.coarse.prompts.all
    class_colors = tab_colors(max(num_classes, 1))
    S.class_colors = class_colors
    class_names = [clean_prompt(all_prompts[p]) for p in prompt_indices]
    class_legend = [(name, class_colors[i]) for i, name in enumerate(class_names)]

    # 1. 準備
    inputs = estimator.prepare(S.resized, S.edgetam_features, S.foreground_map, foreground_binary, S.width, S.height)
    work_w, work_h = inputs.size
    print("作業解像度", inputs.size, "min_area", inputs.min_area, "クラス数", num_classes)
    background_pts = select_background_points(foreground_binary, estimator.background_point_count, S.foreground_map)
    background_view = draw_grid(overlay(base, foreground_binary, alpha=0.4), grid)
    for x, y in background_pts:
        patch_box(background_view, int(x), int(y), grid, (255, 0, 0), 3)
    S.save_panels("d1_background", [
        ("粗い前景マスク", overlay(base, foreground_binary, alpha=0.5)),
        ("前景確信度（低いほど暗い）", heat(S.foreground_map)),
        ("背景の負例点（赤枠）", background_view),
    ])

    # 2. クラスごと（連結成分・正例点・物体らしさ）
    classes = []
    for class_id, semantic_id in enumerate(semantic_ids, start=1):
        coarse_mask = (S.semantic_labels == semantic_id) & foreground_binary
        comps = connected_components(coarse_mask)
        main = refiner._main_components(comps)
        coarse_main = np.any([c.region for c in main], axis=0)
        core = np.zeros_like(coarse_main, dtype=bool)
        for component in main:
            core |= component.core
        points = estimator.foreground_points(inputs, coarse_main, main, core)
        classes.append(SimpleNamespace(id=class_id, mask=coarse_mask, comps=comps, main=main, coarse_main=coarse_main, core=core, points=points))
        print(f"  クラス{class_id}（{class_names[class_id - 1]}）: 成分", [(c.peak, int(c.region.sum())) for c in comps], "正例点", len(points))
    component_view = base.copy()
    for k in classes:
        component_view = overlay(component_view, k.mask, tuple(int(v) for v in class_colors[k.id - 1]), 0.45)
    for k in classes:
        component_view = outline(component_view, np.any([c.core for c in k.comps], axis=0).astype(np.uint8), (255, 255, 255), 2)
        for c in k.comps:
            patch_dot(component_view, c.peak[0], c.peak[1], grid)
    S.component_view, S.classes = component_view, classes

    # 3.・4. クラスごとの推定（本体と同じ関数）→ 取りこぼしの再推定
    state = RefineState(semantic_map=np.zeros((work_h, work_w), dtype=np.uint8), best_score=np.full((work_h, work_w), -1.0, dtype=np.float32))
    class_components = refiner._refine_classes(inputs, state, semantic_ids, S.semantic_labels, foreground_binary)
    debug = state.debug_images
    merged_by_score = state.semantic_map.copy()
    uncovered = [
        (class_id, component)
        for class_id, comps in class_components.items()
        for component in comps
        if not refiner._is_covered(state, refiner._peak_in_work_size(inputs, component))
    ]
    refiner._rescue_uncovered(inputs, state, class_components)
    after_rescue = state.semantic_map.copy()
    print("再推定した成分:", [(c, comp.peak, int(comp.region.sum())) for c, comp in uncovered], " 足された画素:", int(((merged_by_score == 0) & (after_rescue > 0)).sum()))

    # --- 2. 点と初期マスクの画像（ここまでの結果を使う） ---
    save_points_images(S, inputs, state, classes, class_legend)
    save_prior_images(S, inputs, classes)
    masks = [debug[f"c{k.id}_edgetam_mask"] for k in classes]
    panels = []
    for k in classes:
        panels += [
            (f"クラス{k.id}（{class_names[k.id - 1]}）\n粗いマスク", overlay(base, k.mask, alpha=0.55)),
            ("EdgeTAM のマスク\n（clean_mask 後）", overlay(base, cv2.resize(debug[f"c{k.id}_edgetam_mask"].astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), alpha=0.55)),
            ("重なり解決用スコア\n（EdgeTAM の確信度 × 粗い前景らしさ）", heat(cv2.resize(debug[f"c{k.id}_score"].astype(np.float32), (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA), 0, 1, "viridis")),
        ]
    S.save_panels("d2_class_mask", panels, ncols=3, size=3.2)

    # --- 3. 重なりの解決 ---
    overlap_count = np.sum([m.astype(np.uint8) for m in masks], axis=0)
    overlap_panels = [(f"クラス{k.id}（{class_names[k.id - 1]}）\nのマスク", overlay(base, cv2.resize(m.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), tuple(int(v) for v in class_colors[k.id - 1]), 0.6)) for k, m in zip(classes, masks)]
    overlap_view = overlay(base, cv2.resize((overlap_count >= 2).astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), (255, 0, 0), 0.75)
    overlap_panels.append((f"2 クラス以上が主張した画素\n（{int((overlap_count >= 2).sum())} 画素）", overlap_view))
    overlap_panels.append(("スコアが高いクラスを優先した結果\n（重なりの解決後）", class_view(base, merged_by_score, class_colors)))
    S.save_panels("d3_overlap", overlap_panels, ncols=min(len(overlap_panels), 3), size=3.2, legend=class_legend)

    # --- 4. 取りこぼしの再推定 ---
    if uncovered:
        uncovered_view = class_view(base, merged_by_score, class_colors)
        for _, component in uncovered:
            uncovered_view = outline(uncovered_view, to_display(component.region.astype(np.uint8)), (255, 0, 0), 3)
            patch_dot(uncovered_view, component.peak[0], component.peak[1], grid, radius=7)
        added = (merged_by_score == 0) & (after_rescue > 0)
        added_big = cv2.dilate(cv2.resize(added.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8))  # 小さいので、太く描く
        added_view = overlay(base, added_big, (255, 0, 0), 0.75)
        S.save_panels("d4_rescue", [
            ("再推定の前（クラスごとのマスクの和）", class_view(base, merged_by_score, class_colors)),
            ("覆われていない成分（赤枠・黄点: 最内点）", uncovered_view),
            (f"再推定で足された画素（{int(added.sum())}）\n赤: 太く描いた", added_view),
            ("再推定の後", class_view(base, after_rescue, class_colors)),
        ], ncols=4, size=3.2, legend=class_legend)
    S.rescued = bool(uncovered)

    # 5. 後処理: 小領域の除去 -> 隙間埋め
    cleaned = after_rescue.copy()
    remove_small_regions(cleaned, inputs.min_area, Workers(0))
    filled = cleaned.copy()
    fill_narrow_gaps(filled, foreground_binary)
    print("後処理: 小領域の除去で変わった画素", int((cleaned != after_rescue).sum()), " 隙間埋めで変わった画素", int((filled != cleaned).sum()))
    S.save_panels("d5_cleanup", [
        ("重ね合わせ + 再推定の後", class_view(base, after_rescue, class_colors)),
        (f"小領域の除去\n（赤: 変わった画素 {int((cleaned != after_rescue).sum())}。太く描いた）", diff_view(base, after_rescue, cleaned, class_colors)),
        (f"隙間埋め\n（赤: 変わった画素 {int((filled != cleaned).sum())}。太く描いた）", diff_view(base, cleaned, filled, class_colors)),
    ], size=3.6, legend=class_legend)

    # 6. 格子の検出（作動の判定）
    save_lattice_image(S, filled > 0)
    state.semantic_map = filled
    refiner._separate_lattice(inputs, state, S.resized, S.edgetam_features)
    S.lattice_active = "lattice_instances" in state.debug_images

    # 7. 最終の名前付け
    before_naming, indices_before = state.semantic_map.copy(), list(prompt_indices)
    named, indices_after = S.namer.name(state.semantic_map, prompt_indices, S.patch_features)
    save_naming_images(S, before_naming, indices_before, named, indices_after)

    # 6. 元画像の大きさへ戻す → 最終結果
    final = named if inputs.size == (S.width, S.height) else cv2.resize(named, (S.width, S.height), interpolation=cv2.INTER_NEAREST)
    S.final, S.final_indices, S.point_coords, S.point_labels = final, indices_after, state.point_coords, state.point_labels
    names_after = [clean_prompt(all_prompts[p]) for p in indices_after]
    final_colors = tab_colors(max(len(indices_after), 1))
    final_small = cv2.resize(final, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    S.save_panels("d8_final", [
        ("最終結果の semantic_map（クラスごとの色）", class_view(base, final, final_colors)),
        ("輪郭を元画像に重ねる", outline(cv2.resize(S.image_raw, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA), final_small > 0, (255, 255, 0), 2)),
    ], size=4.2, legend=[(n, final_colors[i]) for i, n in enumerate(names_after)])
    S.final_colors = final_colors
    n_fg = sum(S.point_labels)
    print("プロンプト点: 正例", n_fg, "負例", len(S.point_labels) - n_fg, " クラス", names_after)


def save_points_images(S, inputs, state, classes, class_legend) -> None:
    """2. 連結成分・正例点・物体らしさ。"""
    base, grid = S.base, S.grid
    scale = DISPLAY / max(S.width, S.height)
    S.save_panels("d2_components", [
        ("連結成分（色: クラス / 白線: core / 黄点: 最内点）", S.component_view),
        ("EdgeTAM に渡すプロンプト点（緑: 正例 / 赤: 負例）", points_on(base, state.point_coords, state.point_labels, scale)),
    ], size=4.2, legend=class_legend)
    panels = []
    for k in classes:
        score = objectness_map(inputs.normalized_feats, inputs.background_means, k.core)
        if score is None:
            continue
        view = heat(score, -float(np.abs(score).max()), float(np.abs(score).max()), "RdBu_r")
        view = outline(view, to_display(k.coarse_main.astype(np.uint8)), (255, 255, 255), 2)
        for x, y in k.points:
            patch_dot(view, x, y, grid, (0, 200, 0))
        panels.append((f"クラス{k.id}: 物体らしさ（赤 > 0）+ 正例点", view))
    S.save_panels("d2_objectness", panels, ncols=min(len(panels), 4), size=3.0)


def save_prior_images(S, inputs, classes) -> None:
    """2. 初期マスク（事前確率）とデコーダの反復。面積が最大のクラスを使う。"""
    base, grid, estimator, session = S.base, S.grid, S.estimator, S.session
    k = max(classes, key=lambda c: int(c.coarse_main.sum()))
    coarse_map = patch_mean(k.coarse_main)
    prior = estimator.build_prior_map(inputs, k.coarse_main, coarse_map, k.core)
    core_map = patch_mean(k.core)
    unknown = np.maximum(coarse_map, dilate(k.coarse_main).astype(np.float32))
    patch_prior = np.maximum(core_map, 0.5 * unknown)
    upsampled = cv2.resize(patch_prior, estimator.high_res_size, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(prior, np.clip(guided_filter(inputs.guide, upsampled, estimator.guide_radius, GUIDED_FILTER_EPS), 0.0, 1.0)), "初期マスクの再現が本体と合わない"
    high = upsampled.shape
    S.save_panels("d2_prior", [
        ("① core（内側 = 前景）", heat(core_map)),
        ("② 粗いマスク + 外側 1 パッチ（不明）", heat(unknown)),
        ("③ 事前確率（core=1 / 不明=0.5 / 背景=0）", heat(patch_prior)),
        (f"④ {high[1]}×{high[0]} に拡大", heat(upsampled)),
        ("⑤ ガイデッドフィルタ後", heat(prior)),
    ], size=3.0)
    # 縁の違いを見る図: 面積が中央の成分の周り
    comps = sorted((c for kk in classes for c in kk.main), key=lambda c: c.region.sum())
    target = comps[len(comps) // 2]
    ty, tx = np.nonzero(target.region)
    cy0, cy1, cx0, cx1 = max(ty.min() - 2, 0), min(ty.max() + 3, grid[0]), max(tx.min() - 2, 0), min(tx.max() + 3, grid[1])

    def zoom(a, nearest=True):
        h, w = a.shape[:2]
        sub = a[cy0 * h // grid[0]:cy1 * h // grid[0], cx0 * w // grid[1]:cx1 * w // grid[1]]
        rgb = (plt.get_cmap("viridis")(np.clip(sub, 0, 1))[..., :3] * 255).astype(np.uint8)
        return cv2.resize(rgb, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

    img_sub = cv2.resize(S.resized[cy0 * 16:cy1 * 16, cx0 * 16:cx1 * 16], (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)
    S.save_panels("d2_prior_zoom", [
        ("入力画像（1 つの成分の周り）", img_sub),
        ("パッチ格子の事前確率（③）", zoom(patch_prior)),
        ("ガイデッドフィルタ後（⑤）", zoom(prior, nearest=False)),
    ], size=3.6)

    # デコーダの反復（このクラスの点と初期マスクで、decode_prompts の入出力を記録する）
    recorded = []
    original = session.decode_prompts

    def decode(coords, labels, feats, height_, width_, mask_input):
        scores, logits = original(coords, labels, feats, height_, width_, mask_input)
        recorded.append(dict(mask_input=np.array(mask_input), logit=logits[int(np.argmax(scores))].copy()))
        return scores, logits

    session.decode_prompts = decode
    try:
        estimator.estimate_logit(inputs, prior, k.points, "doc")
    finally:
        del session.decode_prompts  # インスタンスの差し替えを外す（クラスのメソッドに戻る）

    def prob(logit):
        return heat(1 / (1 + np.exp(-np.clip(logit, -20, 20))), 0, 1, "viridis")

    steps = [("初期マスク（確率）", prob(recorded[0]["mask_input"].reshape(recorded[0]["mask_input"].shape[-2:])))]
    for i, call in enumerate(recorded, start=1):
        steps.append((f"{i}回目の出力（確率）", prob(call["logit"])))
    last = cv2.resize(recorded[-1]["logit"].astype(np.float32), (DISPLAY, DISPLAY))
    steps.append(("ロジット > 0 を前景に", overlay(base, to_display((last > 0).astype(np.uint8)), alpha=0.55)))
    S.save_panels("d2_iterations", steps, size=3.2)


def save_lattice_image(S, region: np.ndarray) -> None:
    """6. 格子の検出（作動の判定）: 勾配の大きさと、その自己相関。格子なら、周期の位置に強いピークが出る。"""
    gradient, mask, scale = lattice_utils._gradient_and_mask(S.resized, region, LATTICE_DETECT_SIZE)
    corr, valid = lattice_utils._autocorrelation(gradient, mask)
    found = lattice_utils._lattice_vectors(corr, valid, LATTICE_MIN_LAG * scale)
    lattice = lattice_utils.detect(S.resized, region)
    strength = None if found is None else found[2]
    print("格子の検出:", "なし" if lattice is None else "あり", " 強さ:", None if strength is None else round(strength, 3), f"（しきい値 {LATTICE_MIN_STRENGTH}）")
    masked = gradient * mask
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    half = min(cy, cx) // 2
    corr_view = np.where(valid, corr, 0)[cy - half:cy + half, cx - half:cx + half]
    verdict = "格子ではない（作動しない）" if lattice is None else "格子あり（セルごとに分ける）"
    strength_text = "周期のピークが見つからない" if strength is None or strength <= 0 else f"格子の強さ {strength:.2f}（しきい値 {LATTICE_MIN_STRENGTH}）"
    S.save_panels("d6_lattice", [
        (f"前景の中の勾配の大きさ\n（長辺 {LATTICE_DETECT_SIZE}px に縮小）", heat(masked / max(float(masked.max()), 1e-6))),
        (f"勾配の自己相関（中心 = 遅れ 0）\n{strength_text}\n→ {verdict}", heat(corr_view, -0.2, 1.0, "magma")),
    ], size=3.8)


def save_naming_images(S, before_map, indices_before, after_map, indices_after) -> None:
    """7. 最終の名前付け: 領域（物体 × クラス）ごとの照合と、名前の変化。"""
    base, grid, namer = S.base, S.grid, S.namer
    all_prompts = S.coarse.prompts.all
    regions = final_regions(before_map, grid)
    logits = namer.region_logits(S.patch_features, regions.patch_labels)
    num_regions = len(regions.keys)
    if len(logits) < num_regions:
        logits = np.vstack([logits, np.zeros((num_regions - len(logits), logits.shape[1]), logits.dtype)])
    labels = regions.patch_labels
    foreground = labels > 0
    area = np.bincount(labels, minlength=num_regions)
    big_enough = area >= foreground.sum() / namer.min_region_divisor
    region_class = regions.keys % regions.stride
    rows, emphasize = [], []
    for r in range(1, num_regions):
        obj, cls = int(regions.keys[r] // regions.stride), int(region_class[r])
        pixels = (regions.objects == obj) & (before_map == cls)
        after_class = int(np.bincount(after_map[pixels]).argmax()) if pixels.any() else 0
        best = int(np.argmax(logits[r]))
        current, final = clean_prompt(all_prompts[indices_before[cls - 1]]), clean_prompt(all_prompts[indices_after[after_class - 1]]) if after_class else "-"
        rows.append([f"物体{obj}", current, str(int(area[r])), "する" if big_enough[r] else "しない（今のまま）",
                     clean_prompt(all_prompts[best]), final])
        emphasize.append(current != final)
    if rows:
        show_table(S.path("d7_naming"), ["物体", "今の名前", "パッチ数", "判定し直す", "照合が最も強い名前", "最終の名前"], rows,
                   [0.07, 0.25, 0.08, 0.14, 0.25, 0.25], [f"領域{r}" for r in range(1, num_regions)], emphasize, fontsize=9)
    print("名前付け: 領域", num_regions - 1, "個。今の名前 →", [clean_prompt(all_prompts[p]) for p in indices_before], "→",
          [clean_prompt(all_prompts[p]) for p in indices_after])
    names_before = [clean_prompt(all_prompts[p]) for p in indices_before]
    names_after = [clean_prompt(all_prompts[p]) for p in indices_after]
    present = sorted(set(names_before) | set(names_after))
    palette = {name: tab_colors(max(len(present), 1))[i] for i, name in enumerate(present)}

    def by_name(class_map, names):
        small = cv2.resize(class_map.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
        colored = np.zeros((DISPLAY, DISPLAY, 3), np.uint8)
        for k, name in enumerate(names, start=1):
            colored[small == k] = palette[name]
        return np.where((small > 0)[..., None], cv2.addWeighted(base, 0.4, colored, 0.6, 0), base)

    region_colors = tab_colors(max(num_regions - 1, 1), "Set1")  # 名前の色（tab10）と見分けがつくよう、別のパレットにする
    region_map = np.zeros((*grid, 3), np.uint8)
    for r in range(1, num_regions):
        region_map[labels.reshape(grid) == r] = region_colors[(r - 1) % len(region_colors)]
    region_view = np.where(to_display((labels.reshape(grid) > 0).astype(np.uint8))[..., None].astype(bool), cv2.addWeighted(base, 0.4, to_display(region_map), 0.6, 0), base)
    S.save_panels("d7_naming_maps", [
        ("名前付けの前（粗い段階の名前）", by_name(before_map, names_before)),
        ("領域に割り当てたパッチ（色 = 領域。番号は表の行）", region_view),
        ("名前付けの後", by_name(after_map, names_after)),
    ], size=3.6, legend=[(name, palette[name]) for name in present] + [(f"領域{r}", region_colors[(r - 1) % len(region_colors)]) for r in range(1, num_regions)])


def stage_overview(S) -> None:
    final_small = cv2.resize(S.final, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    scale = DISPLAY / max(S.width, S.height)
    S.save_panels("overview", [
        ("入力", S.base),
        ("KMeans のクラスタ", S.cluster_map),
        ("粗い前景マスク", overlay(S.base, S.foreground_mask, alpha=0.55)),
        ("EdgeTAM に渡す点", points_on(S.base, S.point_coords, S.point_labels, scale)),
        ("最終結果", outline(overlay(S.base, final_small > 0, alpha=0.4), final_small > 0, (255, 255, 0), 2)),
    ], size=3.0)


# =============================================================================
# 本体
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="既定は configs/dataset.yaml の先頭の画像")
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--prefix", default="coin")
    parser.add_argument("--model-dir", default=None, help="OpenVINO モデルのフォルダ。既定は設定の model_dir")
    args = parser.parse_args()
    image_path = args.image or load_dataset_paths()[0]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config()
    if args.model_dir:
        config["model_dir"] = args.model_dir
    extractor = OVCLIPFeatureExtractor(config)
    runner = SegmentationRunner(extractor, OVEdgeTamRunner(config), config)
    threaded = runner.workers
    runner.workers = runner.refiner.workers = Workers(0)  # 段階ごとの記録のため、スレッドを使わずに実行する。最後に、スレッドありの結果と比べる

    S = SimpleNamespace(runner=runner, extractor=extractor, coarse=runner.coarse, refiner=runner.refiner, estimator=runner.refiner.estimator,
                        session=runner.session, namer=runner.namer, grid=runner.coarse.grid)
    S.path = lambda name: out / f"{args.prefix}_{name}.jpg"
    S.save_panels = lambda name, panels, **kwargs: save_panels(S.path(name), panels, **kwargs)
    S.image_raw = load_image(image_path)
    S.height, S.width = S.image_raw.shape[:2]
    S.resized = resize_image(S.image_raw, extractor.target_size)
    S.base = to_display(S.resized, cv2.INTER_AREA)
    print("入力", (S.height, S.width), "->", S.resized.shape[:2], "格子", S.grid)

    stage_input(S)
    stage_coarse(S)
    stage_detail(S)
    stage_overview(S)

    # 本体の segment（スレッドあり）と同じ結果か確認する
    runner.workers = runner.refiner.workers = threaded
    reference = runner.segment(S.image_raw)
    same_map = bool(np.array_equal(reference.semantic_map, S.final))
    same_labels = bool(np.array_equal(reference.semantic_labels, S.semantic_labels)) and bool(np.array_equal(reference.foreground_map, S.foreground_map))
    same_names = list(reference.prompt_indices) == list(S.final_indices)
    print("本体の segment と一致: semantic_map", same_map, "（差のある画素", int((reference.semantic_map != S.final).sum()), "）/ 粗いラベル・前景確信度", same_labels, "/ クラスの名前", same_names)
    if not (same_map and same_labels and same_names):
        sys.exit(1)


if __name__ == "__main__":
    main()
