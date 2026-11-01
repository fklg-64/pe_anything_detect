"""説明資料（docs/）用の画像を作る: 1 枚の画像をパイプラインに通し、各段階の中間結果を docs/images/ に保存する。

    python tools/docs/make_doc_images.py [--image PATH] [--out-dir docs/images] [--prefix coin] [--model-dir weights/openvino_models_l6]

画像は JPEG（幅 1200px 以下・品質 80）で保存する（git に載せるため。設定は shrink_images.py）。
--image の既定は configs/dataset.yaml の先頭の画像。--model-dir の既定は configs/ov_config.yaml の model_dir。

本体（SegmentationRunner.segment）と同じ処理を段階ごとに呼び出す（結果は本体と同じ。最後に semantic_map の一致を確かめる）。
OpenVINO 版（configs/ov_config.yaml）で実行する。
"""
import argparse
import io
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.common.mask_utils import connected_components, dilate, guided_filter, knn_graph, patch_mean, propagate, select_background_points
from core.common.parallel import Workers
from core.common.params import CLIP_LOGIT_SCALE, GUIDED_FILTER_EPS
from core.common.segmentation_runner import SegmentationRunner
from core.common.visualization import clean_prompt
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_dataset_paths, load_image, normalize_image, resize_image
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


def gray(values: np.ndarray) -> np.ndarray:
    v = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return to_display(np.stack([v] * 3, axis=-1), cv2.INTER_LINEAR)


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


def save_figure(fig, path: Path, **kwargs) -> None:
    """図を JPEG（幅の上限つき。git に載せるため小さくする）で保存する。設定は shrink_images.py。"""
    from PIL import Image
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, **kwargs)
    buffer.seek(0)
    save_jpeg(Image.open(buffer), path)


def save_panels(path: Path, panels, ncols=None, size=3.4, wspace=0.04) -> None:
    """(タイトル, RGB 画像) のリストを横に並べて 1 枚の図にする。"""
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, (size + 0.35) * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=wspace, hspace=0.18)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


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

    def path(name: str) -> Path:
        return out / f"{args.prefix}_{name}.jpg"

    config = load_config()
    if args.model_dir:
        config["model_dir"] = args.model_dir
    extractor = OVCLIPFeatureExtractor(config)
    edgetam = OVEdgeTamRunner(config)
    runner = SegmentationRunner(extractor, edgetam, config)
    coarse, refiner, estimator, session = runner.coarse, runner.refiner, runner.refiner.estimator, runner.session
    threaded = runner.workers
    runner.workers = refiner.workers = Workers(0)  # 段階ごとの記録（デコーダーの呼び出しの順など）のため、スレッドを使わずに実行する。最後に、スレッドありの結果と比べる
    all_prompts = runner.foreground_prompts + runner.background_prompts
    num_fg_prompts = len(runner.foreground_prompts)

    image_raw = load_image(image_path)
    height, width = image_raw.shape[:2]

    # ---- 前処理 -------------------------------------------------------------
    resized = resize_image(image_raw, extractor.target_size)
    grid = coarse.grid
    base = to_display(resized, cv2.INTER_AREA)
    scale = DISPLAY / max(width, height)  # 元画像座標 -> 表示座標
    print("入力", (height, width), "->", resized.shape[:2], "格子", grid)

    save_panels(path("c00_input"), [
        (f"元画像（{width}×{height}）", cv2.resize(image_raw, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)),
        (f"モデル入力（{resized.shape[1]}×{resized.shape[0]}）+ パッチ格子 {grid[1]}×{grid[0]}", draw_grid(base, grid)),
    ])

    # ---- 特徴量 -------------------------------------------------------------
    edgetam_features = session.encode(resized)
    image_tensor = normalize_image(resized, extractor.mean, extractor.std)
    patch_features, fused_patch_features = coarse.extract_features(image_tensor)
    edge_embed = edgetam_features[2]
    pe_tokens = fused_patch_features[0, 1:, :]
    edge_flat = edge_embed[0].reshape(edge_embed.shape[1], -1).T
    print("PE トークン", pe_tokens.shape, "EdgeTAM image_embed", edge_embed.shape)

    save_panels(path("c01_features"), [
        ("入力", base),
        ("PE のパッチ特徴（主成分 → RGB）", pca_rgb(pe_tokens, grid)),
        (f"EdgeTAM の image_embed（{edge_embed.shape[2]}×{edge_embed.shape[3]}）", pca_rgb(edge_flat, edge_embed.shape[2:])),
    ])

    # ---- KMeans とクラスタの判定 ---------------------------------------------
    cluster_labels = coarse.fit_predict_kmeans(pe_tokens, k=coarse.k_clusters, niter=coarse.niter)
    confidence0, prompt0, _ = coarse._judge_clusters(patch_features, cluster_labels)
    cluster_map = CLUSTER_COLORS[cluster_labels.reshape(grid)]
    save_panels(path("c02_kmeans"), [
        ("入力", base),
        (f"KMeans（k={coarse.k_clusters}）のクラスタ", to_display(cluster_map)),
        ("重ね合わせ", cv2.addWeighted(base, 0.45, to_display(cluster_map), 0.55, 0)),
    ])

    # クラスタごとの判定（最大類似度の比較）
    tokens = extractor.extract_cluster_tokens(patch_features, cluster_labels)
    rows = []
    for index, token in enumerate(tokens):
        feature = token.reshape(1, -1)
        feature = feature / (np.linalg.norm(feature, axis=-1, keepdims=True) + 1e-8)
        logits = (CLIP_LOGIT_SCALE * feature @ coarse.fgbg_text_features.T)[0]
        fg_best, bg_best = int(np.argmax(logits[:num_fg_prompts])), int(np.argmax(logits[num_fg_prompts:]))
        confidence, prompt = coarse._classify_cluster(token)
        rows.append((index, int((cluster_labels == index).sum()), float(logits[fg_best]), float(logits[num_fg_prompts + bg_best]),
                     confidence, clean_prompt(all_prompts[prompt]), clean_prompt(all_prompts[num_fg_prompts + bg_best]), clean_prompt(all_prompts[fg_best])))
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 0.6))
    ax.axis("off")
    cells = [[f"{r[1]}", f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.3f}", "前景" if r[4] > coarse.foreground_threshold else "背景", r[5]] for r in rows]
    table = ax.table(
        cellText=cells, colLabels=["パッチ数", "前景の最大類似度", "背景の最大類似度", "前景確信度", "判定", "ラベル（最も近いプロンプト）"],
        rowLabels=[f"クラスタ{r[0]}" for r in rows], loc="center", cellLoc="center", colWidths=[0.1, 0.16, 0.16, 0.13, 0.08, 0.37],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for i, r in enumerate(rows, start=1):
        table[i, -1].set_facecolor(tuple(CLUSTER_COLORS[r[0]] / 255))
        table[i, -1].get_text().set_color("black")
        if r[4] > coarse.foreground_threshold:
            for j in range(6):
                table[i, j].set_text_props(fontweight="bold")
    save_figure(fig, path("c03_cluster_judge"), bbox_inches="tight")
    plt.close(fig)
    print("保存:", path("c03_cluster_judge"))
    for r in rows:
        print("  cluster", r[0], "n=%d fg=%.1f bg=%.1f conf=%.3f -> %s (bg最寄り: %s)" % (r[1], r[2], r[3], r[4], r[5], r[6]))

    save_panels(path("c04_confidence0"), [
        ("クラスタの判定を各パッチへ", heat(confidence0.reshape(grid))),
        (f"前景確信度 > {coarse.foreground_threshold}", overlay(base, (confidence0 > coarse.foreground_threshold).reshape(grid), alpha=0.55)),
    ])

    # ---- EdgeTAM の特徴での伝播 ----------------------------------------------
    edge_unit = edge_flat / (np.linalg.norm(edge_flat, axis=-1, keepdims=True) + 1e-8)
    graph = knn_graph(edge_unit, 8)
    from core.common.params import EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS
    propagated = propagate(confidence0, graph, EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS)
    change = np.abs(propagated - confidence0)
    # 近傍の例: 前景確信度がしきい値をまたいで変わったパッチのうち、変化が最大のもの（無ければ変化が最大のもの）
    flipped = (confidence0 > coarse.foreground_threshold) != (propagated > coarse.foreground_threshold)
    query = int(np.argmax(np.where(flipped, change, -1))) if flipped.any() else int(np.argmax(change))
    qy, qx = divmod(query, grid[1])
    neighbor_view = draw_grid(base, grid)
    for n in graph.indices[graph.indptr[query]:graph.indptr[query + 1]]:
        patch_box(neighbor_view, int(n % grid[1]), int(n // grid[1]), grid, (255, 230, 0), 2)
    patch_box(neighbor_view, qx, qy, grid, (255, 0, 0), 3)
    print("近傍の例のパッチ", (qx, qy), "conf0=%.3f -> %.3f" % (confidence0[query], propagated[query]), "反転したパッチ数", int(flipped.sum()))

    save_panels(path("c05_propagation"), [
        ("PE の判定（前景確信度）", heat(confidence0.reshape(grid))),
        ("赤: 注目パッチ / 黄: EdgeTAM 特徴の近傍（kNN）", neighbor_view),
        ("伝播後の前景確信度", heat(propagated.reshape(grid))),
        ("確信度の変化（|伝播後 − 伝播前|）", heat(change.reshape(grid), 0, max(float(change.max()), 1e-3), "magma")),
    ], ncols=4, size=3.2)

    # ---- 本体の generate_foreground_map と一致を確認 -------------------------
    generated = coarse.generate_foreground_map(patch_features, fused_patch_features, edge_embed)
    assert np.allclose(generated.foreground_map.reshape(-1), propagated, atol=1e-5), "伝播の再現が本体と合わない"
    foreground_map, foreground_mask = generated.foreground_map, generated.foreground_mask
    save_panels(path("c06_foreground_mask"), [
        ("PE のみ（伝播なし）", overlay(base, (confidence0 > coarse.foreground_threshold).reshape(grid), alpha=0.55)),
        ("伝播あり（粗い前景マスク）", overlay(base, foreground_mask, alpha=0.55)),
    ], size=4.2)

    # ---- 粗いラベルの付与と整形 ---------------------------------------------
    labels_list, semantic_labels = coarse.assign_semantic_labels(generated.cluster_labels, generated.cluster_index_list)

    # セマンティックラベル: 連番 -> プロンプト。背景のクラス（影のプロンプトなど）も含めて全パッチに付く
    label_colors = (plt.get_cmap("tab20")(np.arange(len(labels_list)) % 20)[:, :3] * 255).astype(np.uint8)
    label_view = to_display(label_colors[semantic_labels])
    print("ラベル:", [clean_prompt(all_prompts[i]) for i in labels_list])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.imshow(label_view)
    ax.set_title(f"粗いセマンティックラベル（{len(labels_list)} 種）", fontsize=11)
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=label_colors[i] / 255) for i in range(len(labels_list))]
    names = [("前景: " if labels_list[i] < num_fg_prompts else "背景: ") + clean_prompt(all_prompts[labels_list[i]]) for i in range(len(labels_list))]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.14)
    save_figure(fig, path("c07_semantic_labels"), bbox_inches="tight")
    plt.close(fig)
    print("保存:", path("c07_semantic_labels"))

    # =========================================================================
    # 詳細セグメンテーション
    # =========================================================================
    fg_after, labels_after = foreground_mask, semantic_labels
    foreground_binary = fg_after != 0
    semantic_ids = np.unique(labels_after[foreground_binary])
    background_pts = select_background_points(foreground_binary, estimator.background_point_count, foreground_map)
    background_view = draw_grid(overlay(base, foreground_binary, alpha=0.4), grid)
    for x, y in background_pts:
        patch_box(background_view, int(x), int(y), grid, (255, 0, 0), 3)
    save_panels(path("r01_background_points"), [
        ("粗い前景マスク", overlay(base, foreground_binary, alpha=0.5)),
        ("前景確信度（低いほど暗い）", heat(foreground_map)),
        ("背景の負例点（赤枠）", background_view),
    ])

    # クラスごとの連結成分・最内点・core
    class_id = 1
    coarse_mask = (labels_after == semantic_ids[class_id - 1]) & foreground_binary
    comps = connected_components(coarse_mask)
    print("連結成分:", [(c.peak, int(c.region.sum())) for c in comps])
    comp_view = overlay(base, coarse_mask, alpha=0.35)
    core_union = np.any([c.core for c in comps], axis=0)
    comp_view = outline(comp_view, to_display(core_union.astype(np.uint8)), (0, 255, 255), 2)
    for c in comps:
        cv2.circle(comp_view, (int((c.peak[0] + 0.5) * DISPLAY / grid[1]), int((c.peak[1] + 0.5) * DISPLAY / grid[0])), 6, (255, 255, 0), -1)
        cv2.circle(comp_view, (int((c.peak[0] + 0.5) * DISPLAY / grid[1]), int((c.peak[1] + 0.5) * DISPLAY / grid[0])), 6, (0, 0, 0), 1)

    # 実行: 途中の入出力を記録しながら本体の refine を呼ぶ（estimator・session・refiner のメソッドを、インスタンスで包む）
    recorded = {"priors": [], "decoder": []}
    original_build_prior, original_decode, original_rescue = estimator.build_prior_map, session.decode_prompts, refiner._rescue_uncovered

    def build_prior(inputs, coarse_mask_, coarse_map_, core_):
        prior = original_build_prior(inputs, coarse_mask_, coarse_map_, core_)
        core_map = patch_mean(core_)
        unknown = np.maximum(coarse_map_, dilate(coarse_mask_).astype(np.float32))
        patch_prior = np.maximum(core_map, 0.5 * unknown)
        upsampled = cv2.resize(patch_prior, estimator.high_res_size, interpolation=cv2.INTER_LINEAR)
        recorded["priors"].append(dict(core_map=core_map, unknown=unknown, patch_prior=patch_prior, upsampled=upsampled, prior=prior, guide=inputs.guide,
                                       filtered_raw=guided_filter(inputs.guide, upsampled, estimator.guide_radius, GUIDED_FILTER_EPS)))
        return prior

    def decode(coords, labels, feats, height_, width_, mask_input):
        scores, logits = original_decode(coords, labels, feats, height_, width_, mask_input)
        recorded["decoder"].append(dict(mask_input=np.array(mask_input), scores=np.array(scores), logit=logits[int(np.argmax(scores))].copy(), points=len(coords)))
        return scores, logits

    pre_rescue = {}

    def rescue(inputs, state, class_components):
        pre_rescue["map"] = state.semantic_map.copy()
        pre_rescue["uncovered"] = [
            (class_id_, component)
            for class_id_, comps_ in class_components.items()
            for component in comps_
            if not refiner._is_covered(state, refiner._peak_in_work_size(inputs, component))
        ]
        return original_rescue(inputs, state, class_components)

    estimator.build_prior_map, session.decode_prompts, refiner._rescue_uncovered = build_prior, decode, rescue
    refined = refiner.refine(resized, foreground_map, fg_after, labels_after, labels_list, width, height, edgetam_features)
    del estimator.build_prior_map, session.decode_prompts, refiner._rescue_uncovered  # インスタンスの差し替えを外す（クラスのメソッドに戻る）

    n_fg = sum(refined.point_labels)
    print("プロンプト点: 正例", n_fg, "負例", len(refined.point_labels) - n_fg, "デコーダー呼び出し", len(recorded["decoder"]))

    save_panels(path("r02_points"), [
        ("連結成分（水色の線: core、黄点: 最内点）", comp_view),
        ("EdgeTAM に渡すプロンプト点（緑: 正例 / 赤: 負例）", points_on(base, refined.point_coords, refined.point_labels, scale)),
    ], size=4.2)

    p = recorded["priors"][0]
    high = p["upsampled"].shape
    save_panels(path("r03_prior"), [
        ("① core（内側 = 前景）", heat(p["core_map"])),
        ("② 粗いマスク + 外側 1 パッチ（不明）", heat(p["unknown"])),
        ("③ 事前確率（core=1 / 不明=0.5 / 背景=0）", heat(p["patch_prior"])),
        (f"④ {high[1]}×{high[0]} に拡大", heat(p["upsampled"])),
        ("⑤ ガイデッドフィルタ後", heat(p["prior"])),
    ], size=3.0)
    # 拡大して見比べる（1 個のコインの周り）
    ys, xs = np.nonzero(p["patch_prior"] > 0)
    y0, y1, x0, x1 = ys.min() * high[0] // grid[0], (ys.max() + 1) * high[0] // grid[0], xs.min() * high[1] // grid[1], (xs.max() + 1) * high[1] // grid[1]
    def crop(a):
        c = a[y0:y1, x0:x1]
        return cv2.resize(plt.get_cmap("viridis")(np.clip(c, 0, 1))[..., :3].astype(np.float32), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    # 縁の違いを見る図: 1 つの成分の周り
    target = sorted(comps, key=lambda c: c.region.sum())[len(comps) // 2]  # 面積が中央の成分（結合した大きな成分は避ける）
    ty, tx = np.nonzero(target.region)
    cy0, cy1, cx0, cx1 = max(ty.min() - 2, 0), min(ty.max() + 3, grid[0]), max(tx.min() - 2, 0), min(tx.max() + 3, grid[1])
    def zoom(a, nearest=True):
        h, w = a.shape[:2]
        sub = a[cy0 * h // grid[0]:cy1 * h // grid[0], cx0 * w // grid[1]:cx1 * w // grid[1]]
        rgb = (plt.get_cmap("viridis")(np.clip(sub, 0, 1))[..., :3] * 255).astype(np.uint8)
        return cv2.resize(rgb, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    img_sub = cv2.resize(resized[cy0 * 16:cy1 * 16, cx0 * 16:cx1 * 16], (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)
    save_panels(path("r04_prior_zoom"), [
        ("入力画像（1 つのコインの周り）", img_sub),
        ("パッチ格子の事前確率（③）", zoom(p["patch_prior"])),
        ("ガイデッドフィルタ後（⑤）", zoom(p["prior"], nearest=False)),
    ], size=3.6)

    # デコーダーの反復
    decoder_calls = recorded["decoder"]
    mask_in = decoder_calls[0]["mask_input"].reshape(decoder_calls[0]["mask_input"].shape[-2:])
    def logit_view(l):
        prob = 1 / (1 + np.exp(-np.clip(l, -20, 20)))
        return heat(prob, 0, 1, "viridis")
    steps = [("初期マスク（確率）", logit_view(mask_in))]
    for i, call in enumerate(decoder_calls[:config["edgetam_refine_iterations"]], start=1):
        steps.append((f"{i}回目の出力（確率）", logit_view(call["logit"])))
    steps.append(("ロジット > 0 を前景に", overlay(base, to_display((cv2.resize(decoder_calls[config["edgetam_refine_iterations"] - 1]["logit"].astype(np.float32), (DISPLAY, DISPLAY)) > 0).astype(np.uint8)), alpha=0.55)))
    save_panels(path("r05_iterations"), steps, size=3.2)

    # クラスのマスクとスコア
    dbg = refined.debug_images
    binary = dbg["c1_edgetam_mask"]
    score = dbg["c1_score"]
    save_panels(path("r06_class_mask"), [
        ("粗いマスク（パッチ）", overlay(base, dbg["c1_coarse_mask"], alpha=0.55)),
        ("EdgeTAM のマスク（clean_mask 後）", overlay(base, cv2.resize(binary.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), alpha=0.55)),
        ("重なり解決用スコア", heat(cv2.resize(score.astype(np.float32), (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA), 0, 1, "viridis")),
    ])

    before_cleanup = dbg["semantic_map_before_cleanup"]
    num_classes = len(semantic_ids)
    class_colors = (plt.get_cmap("tab10")(np.arange(num_classes) % 10)[:, :3] * 255).astype(np.uint8)

    def class_view(class_map):
        small = cv2.resize(class_map, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
        colored = class_colors[np.clip(small.astype(int) - 1, 0, None)]
        return np.where((small > 0)[..., None], cv2.addWeighted(base, 0.45, colored, 0.55, 0), base)

    if pre_rescue["uncovered"]:
        after_rescue = np.rint(before_cleanup * num_classes).astype(np.uint8)
        uncovered_view = class_view(pre_rescue["map"])
        for class_id_, component in pre_rescue["uncovered"]:
            region = to_display(component.region.astype(np.uint8))
            uncovered_view = outline(uncovered_view, region, (255, 0, 0), 3)
            cx, cy = int((component.peak[0] + 0.5) * DISPLAY / grid[1]), int((component.peak[1] + 0.5) * DISPLAY / grid[0])
            cv2.circle(uncovered_view, (cx, cy), 7, (255, 255, 0), -1)
            cv2.circle(uncovered_view, (cx, cy), 7, (0, 0, 0), 1)
        added = (pre_rescue["map"] == 0) & (after_rescue > 0)
        added_view = overlay(base, cv2.resize(added.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), (255, 0, 0), 0.6)
        print("再推定した成分:", [(c, comp.peak, int(comp.region.sum())) for c, comp in pre_rescue["uncovered"]], " 足された画素:", int(added.sum()))
        save_panels(path("r08_rescue"), [
            ("再推定の前（クラスごとのマスクの和）", class_view(pre_rescue["map"])),
            ("覆われていない成分（赤枠・黄点: 最内点）", uncovered_view),
            ("再推定で足された画素", added_view),
            ("再推定の後", class_view(after_rescue)),
        ], ncols=4, size=3.2)

    final = refined.semantic_map
    final_small = cv2.resize(final, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
    save_panels(path("r07_final"), [
        ("最終結果の semantic_map（クラスごとの色）", class_view(final)),
        ("輪郭を元画像に重ねる", outline(cv2.resize(image_raw, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA), final_small > 0, (255, 255, 0), 2)),
    ], size=4.2)

    # ---- 全体像の 1 枚 ------------------------------------------------------
    save_panels(path("overview"), [
        ("入力", base),
        ("KMeans のクラスタ", to_display(cluster_map)),
        ("粗い前景マスク", overlay(base, fg_after, alpha=0.55)),
        ("EdgeTAM に渡す点", points_on(base, refined.point_coords, refined.point_labels, scale)),
        ("最終結果", outline(overlay(base, final_small > 0, alpha=0.4), final_small > 0, (255, 255, 0), 2)),
    ], size=3.0)

    # ---- 本体の segment（スレッドあり）と同じ結果か確認 -----------------------
    runner.workers = refiner.workers = threaded
    reference = runner.segment(image_raw)
    same = bool((reference.semantic_map == final).all())
    print("本体の segment と semantic_map が一致:", same, " 差のある画素:", int((reference.semantic_map != final).sum()))


if __name__ == "__main__":
    main()
