"""格子の検出方法の説明資料（docs/lattice_detection.md）用の画像を作る: シリンジの巣の画像をパイプラインに通し、格子の検出の各段階の中間結果を docs/images/ に保存する。

    python tools/docs/make_lattice_doc_images.py [--image PATH] [--out-dir docs/images] [--prefix latdet]

画像は JPEG（幅 1200px 以下。設定は shrink_images.py）で保存する。OpenVINO 版（configs/ov_config.yaml）で実行する。
本体の `core/common/lattice.py`・`LatticeSeparator.separate` と同じ手順を段階ごとに再現する（ピークの候補や位相のテンプレートなど、本体が返さない
中間値を取り出すため）。最後に、再現した結果（格子ベクトル・位相・セルの中心）が本体の出力と一致することを確かめる。
"""
import argparse
import glob
import sys
from pathlib import Path
from typing import Dict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.openvino.runtime_utils import make_compile_config  # noqa: F401  (import openvino より前に /opt/intel/openvino を外す)
from core.common.detail import lattice as L
from core.common.params import (
    LATTICE_DETECT_SIZE,
    LATTICE_MIN_CELLS,
    LATTICE_MIN_LAG,
    LATTICE_MIN_STRENGTH,
    LATTICE_PEAK_RADIUS,
    LATTICE_PEAK_RATIO,
    LATTICE_PHASE_BINS,
    LATTICE_PHASE_PROBES,
)
from core.common.segmentation_runner import SegmentationRunner
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_image
from tools.docs.make_doc_images import save_figure

SYRINGE_DIR = Path("/home/llm/datasets/02_object_detection/syringe")
DEFAULT_IMAGE = SYRINGE_DIR / "006_syringe.png"
C1, C2, C3 = "#e6194b", "#0082c8", "#3cb44b"   # v1・v2・補助の色（make_doc_images.py の色と同じ系統）


# =============================================================================
# 本体の呼び出しと、中間値の取り出し
# =============================================================================

def run_pipeline(runner: SegmentationRunner, path: Path) -> Dict:
    """画像を本体のパイプラインに通し、格子の分離に渡された値（作業解像度の前景・モデル入力の画像・ロジット・特徴）と、その結果を取り出す。"""
    separator = runner.refiner.lattice_separator
    captured: Dict = {}
    original = separator.separate

    def spy(semantic_map, content, logit, feats, foreground_map):
        captured.update(semantic_map=semantic_map.copy(), content=content.copy(), logit=logit.copy(), feats=feats, foreground_map=foreground_map.copy())
        captured["result"] = original(semantic_map, content, logit, feats, foreground_map)
        return captured["result"]

    separator.separate = spy
    try:
        image = load_image(path)
        runner.segment(image)
    finally:
        separator.separate = original
    captured["image"] = image
    return captured


def detection_steps(content: np.ndarray, region: np.ndarray) -> Dict:
    """`lattice.detect` の前半（小さい画像での判定）を段階ごとに再現する。"""
    steps: Dict = {}
    height, width = content.shape[:2]
    steps["scale"] = scale = LATTICE_DETECT_SIZE / max(height, width)
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    gray = cv2.cvtColor(cv2.resize(content, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    steps["gray"] = cv2.GaussianBlur(gray, (0, 0), 1.5 * LATTICE_DETECT_SIZE / 512)
    steps["gradient"], steps["mask"], _ = L._gradient_and_mask(content, region, LATTICE_DETECT_SIZE)
    steps["corr"], steps["valid"] = L._autocorrelation(steps["gradient"], steps["mask"])
    steps["found"] = L._lattice_vectors(steps["corr"], steps["valid"], LATTICE_MIN_LAG * scale)
    return steps


def corr_peaks(corr: np.ndarray, valid: np.ndarray, min_lag: float):
    """自己相関の極大（`_lattice_vectors` と同じ条件）。(遅れ (dx, dy), 値) と、候補（強い順に 16 個、最も強いピークの LATTICE_PEAK_RATIO 倍以上）。"""
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    candidate = valid & (corr > 0.05)
    local_max = corr == ndi.maximum_filter(corr, size=2 * LATTICE_PEAK_RADIUS + 1)
    coords = np.argwhere(local_max & candidate) - np.array([cy, cx])
    values = corr[local_max & candidate]
    far = np.hypot(coords[:, 0], coords[:, 1]) >= min_lag
    coords, values = coords[far], values[far]
    lags = np.stack([coords[:, 1], coords[:, 0]], axis=1).astype(float)   # (dx, dy)
    order = np.argsort(-values)[:16]
    strongest = values[order].max()
    chosen = [i for i in order if values[i] >= LATTICE_PEAK_RATIO * strongest]
    return lags, values, order, chosen


def refine_samples(corr: np.ndarray, valid: np.ndarray, v, max_multiple: int = 12):
    """`_refine_vector` と同じ手順で、k·v の近くのピークの位置 (k, 遅れ) を集める。"""
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    ks, lags = [], []
    for k in range(1, max_multiple + 1):
        x0, x1 = int(round(cx + k * v[0])) - 3, int(round(cx + k * v[0])) + 4
        y0, y1 = int(round(cy + k * v[1])) - 3, int(round(cy + k * v[1])) + 4
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            break
        window = corr[y0:y1, x0:x1]
        if window.max() < 0.25 or not valid[y0:y1, x0:x1].all():
            break
        iy, ix = np.unravel_index(np.argmax(window), window.shape)
        ks.append(k)
        lags.append((x0 + ix - cx, y0 + iy - cy))
    return np.array(ks, float), np.array(lags, float)


def phase_template(lattice: L.Lattice, values: np.ndarray, mask: np.ndarray, input_size):
    """`find_phase` と同じ手順で、セルのテンプレート・明るい側の重み・物体の中心（周期の中の位置 0〜1）を返す。"""
    grid_h, grid_w = values.shape
    ys, xs = np.nonzero(mask)
    points = np.stack([(xs + 0.5) * input_size[1] / grid_w, (ys + 0.5) * input_size[0] / grid_h], axis=1)
    basis = L.basis(lattice)
    coef = np.linalg.solve(basis, points.T).T
    frac = coef - np.floor(coef)
    bins = np.minimum((frac * LATTICE_PHASE_BINS).astype(int), LATTICE_PHASE_BINS - 1)
    total = np.zeros((LATTICE_PHASE_BINS, LATTICE_PHASE_BINS))
    count = np.zeros_like(total)
    np.add.at(total, (bins[:, 1], bins[:, 0]), values[ys, xs])
    np.add.at(count, (bins[:, 1], bins[:, 0]), 1)
    template = total / np.maximum(count, 1)
    weight = np.maximum(template - (template.max() + template.min()) / 2, 0)
    centres = (np.arange(LATTICE_PHASE_BINS) + 0.5) / LATTICE_PHASE_BINS
    centre_frac = []
    for axis_weight in (weight.sum(0), weight.sum(1)):
        angle = np.arctan2((axis_weight * np.sin(2 * np.pi * centres)).sum(), (axis_weight * np.cos(2 * np.pi * centres)).sum())
        centre_frac.append((angle / (2 * np.pi)) % 1.0)
    return template, weight, np.array(centre_frac), basis @ np.array(centre_frac)


def correct_phase(separator, lattice, phase, region, feats, input_hw):
    """`LatticeSeparator.separate` の位相の補正を再現する（一部のセルをデコードして、重心のずれの中央値で位相を直す）。"""
    half = np.array(L.cell_half_size(lattice))
    basis = L.basis(lattice)
    centers = L.cell_centers(lattice, phase, region, input_hw)
    probes = centers[np.linspace(0, len(centers) - 1, min(len(centers), LATTICE_PHASE_PROBES)).astype(int)]
    low, scores, origins = separator._decode(feats, probes, half)
    centroids = separator._centroids(low, scores, origins)
    found = ~np.isnan(centroids[:, 0])
    offset = np.zeros(2)
    if found.sum() >= 3:
        offset = np.linalg.solve(basis, (centroids[found] - probes[found]).T).T
        offset -= np.round(offset)
        offset = np.median(offset, axis=0)
        phase = phase + offset @ basis.T
    return phase, probes, centroids, found


# =============================================================================
# 図
# =============================================================================

def new_axes(ncols: int, size=(3.7, 3.7), nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows + 0.3), squeeze=False)
    for ax in axes.ravel():
        ax.set_anchor("N")   # 縦横比が違う画像でも、タイトルの高さをそろえる
    return fig, axes.ravel()


def show(ax, image, title, cmap=None, vmin=None, vmax=None, extent=None):
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
    ax.set_title(title, fontsize=10.5)


def finish(fig, path: Path, hide_axes=True, left=0.01, bottom=0.01, wspace=0.05):
    if hide_axes:
        for ax in fig.axes:
            if not getattr(ax, "_keep_axes", False):
                ax.axis("off")
    fig.subplots_adjust(left=left, right=0.99, top=0.88, bottom=bottom, wspace=wspace)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


def outline_mask(ax, mask, color="yellow", lw=1.4, extent=None):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw, extent=extent, origin="upper")


def fig_input(data, path):
    image, content, region = data["image"], data["content"], data["semantic_map"] > 0
    fig, axes = new_axes(3)
    show(axes[0], image, f"① 元画像（{image.shape[1]}×{image.shape[0]}）")
    show(axes[1], content, f"② モデル入力（{content.shape[1]}×{content.shape[0]}）。以降はこの座標")
    base = cv2.resize(content, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    shaded = base.copy()
    shaded[region] = 0.55 * shaded[region] + 0.45 * np.array([230, 25, 75])
    show(axes[2], shaded.astype(np.uint8), f"③ 最終の前景（semantic_map > 0）。{region.mean() * 100:.0f}% が前景\n巣が 1 つの塊になっている")
    finish(fig, path)


def fig_gradient(steps, path):
    fig, axes = new_axes(4, size=(3.0, 3.0))
    show(axes[0], steps["gray"], f"① 縮小（長辺 {LATTICE_DETECT_SIZE}px）+ グレー + ぼかし", cmap="gray")
    show(axes[1], steps["gradient"], "② 勾配の大きさ（Sobel）", cmap="magma")
    show(axes[2], steps["mask"], "③ 前景のマスク（256px）", cmap="gray")
    show(axes[3], steps["gradient"] * steps["mask"], "④ 前景の中の勾配（②×③）\n前景の外は無視する", cmap="magma")
    finish(fig, path)


def fig_autocorrelation(steps, path):
    corr = steps["corr"]
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    half = 110
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for ax in axes[:2]:
        ax.set_anchor("N")
    ext_full = (-cx, width - cx, height - cy, -cy)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under("0.85")
    im = axes[0].imshow(corr, cmap=cmap, vmin=-0.2, vmax=1.0, extent=ext_full, interpolation="nearest")
    axes[0].set_title("① 自己相関（全体）。灰色 = 重なりが足りない遅れ", fontsize=10)
    axes[0].add_patch(plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, ec="cyan", lw=1.5))
    window = corr[cy - half:cy + half, cx - half:cx + half]
    axes[1].imshow(window, cmap=cmap, vmin=-0.2, vmax=1.0, extent=(-half, half, half, -half), interpolation="nearest")
    axes[1].set_title("② 中央の拡大。白い点 = 格子点のピーク", fontsize=10)
    axes[1].plot([0], [0], "+", color="cyan", ms=12, mew=2)
    axes[1].text(-half + 6, -half + 10, "+ = 遅れ 0（自分自身。相関 1）", color="cyan", fontsize=8.5, bbox=dict(fc="black", ec="none", alpha=0.6, pad=2))
    row = corr[cy, cx - half:cx + half]
    axes[2].plot(np.arange(-half, half), row, color=C2, lw=1.6)
    axes[2].axhline(0, color="0.6", lw=0.8)
    axes[2].set_title("③ 横方向（dy = 0）の断面。v の整数倍で山", fontsize=10)
    axes[2].set_xlabel("遅れ dx（256px の画素）")
    axes[2].set_ylabel("自己相関")
    axes[2].spines[["top", "right"]].set_visible(False)
    for ax in axes[:2]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    cbar.set_label("自己相関", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.13, wspace=0.18)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


def fig_vectors(steps, content, path):
    corr, valid, scale = steps["corr"], steps["valid"], steps["scale"]
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    half = 110
    lags, values, order, chosen = corr_peaks(corr, valid, LATTICE_MIN_LAG * scale)
    v1, v2, strength = steps["found"]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under("0.85")
    fig, axes = new_axes(3, size=(3.9, 3.9))
    ax = axes[0]
    ax.imshow(corr[cy - half:cy + half, cx - half:cx + half], cmap=cmap, vmin=-0.2, vmax=1.0, extent=(-half, half, half, -half), interpolation="nearest")
    inside = (abs(lags[:, 0]) < half) & (abs(lags[:, 1]) < half)
    ax.plot(lags[inside, 0], lags[inside, 1], ".", color="white", ms=3.5, label="極大（相関 > 0.05）")
    pick = [i for i in chosen if inside[i]]
    ax.plot(lags[pick, 0], lags[pick, 1], "o", mfc="none", mec="cyan", ms=9, mew=1.5, label=f"候補（最強の {LATTICE_PEAK_RATIO} 倍以上）")
    for color, v, name in ((C1, v1, "v1"), (C2, v2, "v2")):
        ax.annotate("", (v[0], v[1]), (0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4))
        ax.text(v[0] * 1.0 + 4, v[1] * 1.0 - 2, f"{name}", color=color, fontsize=11, fontweight="bold", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1))
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    ax.set_title("① 候補のうち最も短い v1 と、\nv1 と平行でない最短の v2", fontsize=10.5)
    # 画像をベクトルだけずらして重ねる
    vx, vy = v1[0] / scale, v1[1] / scale

    def shifted(sx, sy):
        matrix = np.float32([[1, 0, sx], [0, 1, sy]])
        return cv2.warpAffine(content, matrix, (content.shape[1], content.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    show(axes[1], cv2.addWeighted(content, 0.5, shifted(vx, vy), 0.5, 0), f"② 画像を v1（{vx:.0f}px）ずらして重ねる\n物体が重なる = 自己相関が高い")
    show(axes[2], cv2.addWeighted(content, 0.5, shifted(vx / 2, vy / 2), 0.5, 0), f"③ v1 の半分（{vx / 2:.0f}px）ずらして重ねる\n二重像になる = 自己相関が低い")
    finish(fig, path)


def fig_gate(steps, per_image, path):
    corr, valid, scale = steps["corr"], steps["valid"], steps["scale"]
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    v1, v2, strength = steps["found"]

    def at(dx, dy):
        x, y = int(round(cx + dx)), int(round(cy + dy))
        return float(corr[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].max())

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), gridspec_kw=dict(width_ratios=[1, 1.25]))
    labels = ["v1", "2·v1", "v2", "2·v2"]
    vals = [at(*v1), at(2 * v1[0], 2 * v1[1]), at(*v2), at(2 * v2[0], 2 * v2[1])]
    ax = axes[0]
    bars = ax.bar(labels, vals, color=[C1, C1, C2, C2], width=0.6)
    ax.axhline(LATTICE_MIN_STRENGTH, color="k", ls="--", lw=1.2, label=f"下限 {LATTICE_MIN_STRENGTH}")
    ax.axhline(min(vals), color=C3, lw=1.6, label=f"強さ = 4 つの最小 = {min(vals):.2f}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5, framealpha=0.9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("自己相関")
    ax.set_title(f"① 006 の格子の強さ（v・2v の自己相関の小さいほう）\nセル数 {steps['cells']:.0f} 個（下限 {LATTICE_MIN_CELLS}）", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = [r["name"] for r in per_image]
    ys = np.arange(len(names))[::-1]
    colors = [C3 if r["strength"] is not None and r["strength"] >= LATTICE_MIN_STRENGTH else "0.7" for r in per_image]
    ax.barh(ys, [r["strength"] or 0 for r in per_image], color=colors, height=0.6)
    ax.axvline(LATTICE_MIN_STRENGTH, color="k", ls="--", lw=1.2)
    for y, r in zip(ys, per_image):
        text = "前景がなく、検出まで進まない" if r["strength"] is None else f"{r['strength']:.2f}（セル {r['cells']:.0f} 個。{r['note']}）"
        ax.text((r["strength"] or 0) + 0.015, y, text, va="center", fontsize=8.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.75)
    ax.set_xlabel("格子の強さ")
    ax.set_title("② シリンジの 6 枚の、格子の強さ（破線 = 下限 0.4）", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    for ax in axes:
        ax._keep_axes = True
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.14, wspace=0.28)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


def corner_cells(r):
    """右下の 3×3 セルの切り出し（画像・extent）と、最初のベクトル・精密化したベクトルが指すセルの中心。原点は左上のセルの中心。"""
    content = r["content"]
    sign = lambda v, axis: 1 if v[axis] > 0 else -1
    a, a2 = (r[k] / r["full_scale"] * sign(r[k], 0) for k in ("start1", "refined1"))
    b, b2 = (r[k] / r["full_scale"] * sign(r[k], 1) for k in ("start2", "refined2"))
    origin, nx, ny = r["origin"], r["nx"], r["ny"]
    cells = [(i, j) for i in range(nx - 3, nx) for j in range(ny - 3, ny)]
    start_pts = np.array([origin + i * a + j * b for i, j in cells])
    refined_pts = np.array([origin + i * a2 + j * b2 for i, j in cells])
    margin = 0.8 * np.array([abs(a2[0]), abs(b2[1])])
    x0, y0 = np.maximum(refined_pts.min(axis=0) - margin, 0)
    x1, y1 = np.minimum(refined_pts.max(axis=0) + margin, [content.shape[1], content.shape[0]])
    crop = content[int(y0):int(y1), int(x0):int(x1)]
    return crop, (int(x0), int(x1), int(y1), int(y0)), start_pts, refined_pts


def fig_refine(r, path):
    """精密化の図。① 最後のピーク（k が最大）の拡大、② ピークの実測位置と k·v のずれ、③ 右下のセルで、最初のベクトルと精密化したベクトルが指す位置。"""
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    # ① 最後のピーク（v1）の拡大
    ax = axes[0]
    start, refined, k_max = r["start1"], r["refined1"], int(r["k1"][-1])
    unit = start / np.hypot(*start)
    centre = k_max * np.hypot(*refined)
    t = np.linspace(centre - 14, centre + 14, 400)
    profile = ndi.map_coordinates(r["corr"], [r["cy"] + t * unit[1], r["cx"] + t * unit[0]], order=1, mode="nearest")
    profile[profile < -0.99] = np.nan   # 重なりが足りない遅れ
    ax.plot(t, profile, color="0.35", lw=1.5)
    measured = np.hypot(*r["lag1"][-1])
    ax.axvline(k_max * np.hypot(*start), color="k", ls=":", lw=1.6, label=f"最初の v1 の {k_max} 倍 = {k_max * np.hypot(*start):.1f}")
    ax.axvline(centre, color=C3, lw=1.8, label=f"精密化した v1 の {k_max} 倍 = {centre:.1f}")
    ax.plot([measured], [np.nanmax(profile)], "v", color=C1, ms=9, label=f"実測のピーク = {measured:.0f}")
    ax.set_xlabel(f"v1 の方向の遅れ（512px の画素）。{k_max} 番目のピーク付近")
    ax.set_ylabel("自己相関")
    ax.set_title(f"① 最も遠いピーク（k = {k_max}）の拡大\n最初の値は {abs(k_max * np.hypot(*start) - measured):.0f}px ずれる", fontsize=10)
    ax.legend(fontsize=7.8, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    # ② ずれ
    ax = axes[1]
    for color, key, name in ((C1, "1", "v1"), (C2, "2", "v2")):
        k, lag = r["k" + key], r["lag" + key]
        unit = r["refined" + key] / np.hypot(*r["refined" + key])
        measured = lag @ unit
        ax.plot(k, measured - k * (r["start" + key] @ unit), "o--", color=color, mfc="none", label=f"{name}: 最初の値（{np.hypot(*r['start' + key]):.2f}px）")
        ax.plot(k, measured - k * (r["refined" + key] @ unit), "o-", color=color, label=f"{name}: 精密化（{np.hypot(*r['refined' + key]):.2f}px）")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("k（最初のピークから数えた倍数）")
    ax.set_ylabel("実測のピーク − k·v（px）")
    ax.set_title("② 実測のピーク位置と k·v のずれ\n最初の値は k とともにずれ、最小二乗は ±1px", fontsize=10)
    ax.set_ylim(-4.6, 1.4)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    # ③ 右下の 3×3 セル
    ax = axes[2]
    crop, extent, start_pts, refined_pts = corner_cells(r)
    ax.imshow(crop, extent=extent)
    ax.plot(*start_pts.T, "o", mfc="none", mec="yellow", ms=13, mew=2.2, label="最初のベクトル")
    ax.plot(*refined_pts.T, "+", color=C3, ms=11, mew=2.4, label="精密化したベクトル")
    ax.set_title(f"③ 右下の 3×3 セル（左上のセルから {r['nx'] - 1}・{r['ny'] - 1} 個先）\n黄 = 最初の値は物体の中心から外れる", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.85)
    ax.axis("off")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.84, bottom=0.14, wspace=0.28)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


def fig_phase(p, path):
    content, logit = p["content"], p["logit"]
    fig, axes = new_axes(4, size=(2.95, 2.95))
    ax = axes[0]
    show(ax, logit, "① 物体らしさ\n（EdgeTAM のロジット。> 0 が前景）", cmap="coolwarm", vmin=-12, vmax=12)
    ax = axes[1]
    show(ax, np.tile(p["template"], (2, 2)), f"② 周期で重ねた平均（テンプレート）\n{LATTICE_PHASE_BINS}×{LATTICE_PHASE_BINS} を 2×2 に並べた", cmap="viridis", extent=(0, 2, 2, 0))
    centres = (np.arange(2 * LATTICE_PHASE_BINS) + 0.5) / LATTICE_PHASE_BINS
    ax.contour(centres, centres, np.tile((p["weight"] > 0).astype(float), (2, 2)), levels=[0.5], colors=["white"], linewidths=1.2)
    for dx in (0, 1):
        for dy in (0, 1):
            ax.plot(p["centre_frac"][0] + dx, p["centre_frac"][1] + dy, "x", color="red", ms=10, mew=2.4)
    ax.text(0.03, 0.06, "白線 = 明るい側 / 赤 × = 周期的な重心", color="white", fontsize=7.5, va="top", bbox=dict(fc="black", ec="none", alpha=0.5, pad=1.5))
    ax = axes[2]
    show(ax, content, f"③ 格子点（位相）\n白 = 重心、緑 = {len(p['probes'])} セルのデコードで補正")
    ax.plot(*p["points0"].T, ".", color="white", ms=4)
    ax.plot(*p["points1"].T, ".", color=C3, ms=3.2)
    # 補正の拡大（左上の 3×3 セル）
    ax = axes[3]
    origin = p["points1"][np.argmin(p["points1"].sum(axis=1))]
    half = 1.6 * np.hypot(*p["v1"])
    x0, y0 = max(origin[0] - half, 0), max(origin[1] - half, 0)
    x1, y1 = min(x0 + 2 * half, content.shape[1]), min(y0 + 2 * half, content.shape[0])
    show(ax, content[int(y0):int(y1), int(x0):int(x1)], "④ ③ の左上の拡大\n補正で物体の中心に合う", extent=(x0, x1, y1, y0))
    sel0 = (p["points0"][:, 0] > x0) & (p["points0"][:, 0] < x1) & (p["points0"][:, 1] > y0) & (p["points0"][:, 1] < y1)
    sel1 = (p["points1"][:, 0] > x0) & (p["points1"][:, 0] < x1) & (p["points1"][:, 1] > y0) & (p["points1"][:, 1] < y1)
    ax.plot(*p["points0"][sel0].T, "o", mfc="none", mec="white", ms=9, mew=1.8)
    ax.plot(*p["points1"][sel1].T, "+", color=C3, ms=10, mew=2.2)
    finish(fig, path)


def fig_cells(c, path):
    content = c["content"]
    fig, axes = new_axes(3, size=(3.9, 3.9))
    ax = axes[0]
    show(ax, content, f"① 格子点のうち、前景に入るもの = セルの中心\n（{len(c['centers'])} 個 = {c['nx']}×{c['ny']}）")
    ax.plot(*c["all_points"].T, ".", color="0.6", ms=2.5)
    ax.plot(*c["centers"].T, ".", color=C3, ms=5)
    half_w, half_h = c["half"]
    for x, y in c["centers"]:
        ax.add_patch(plt.Rectangle((x - half_w, y - half_h), 2 * half_w, 2 * half_h, fill=False, ec="yellow", lw=0.5))
    ax = axes[1]
    x0, y0 = c["centers"][np.argmin(c["centers"].sum(axis=1))]
    w, h = 3.4 * half_w, 3.4 * half_h
    x0, y0 = max(x0 - 1.2 * half_w, 0), max(y0 - 1.2 * half_h, 0)
    show(ax, content[int(y0):int(y0 + 2 * h), int(x0):int(x0 + 2 * w)], f"② 拡大。箱 = 格子ベクトル 1 つ分\n（{2 * half_w:.0f}×{2 * half_h:.0f}px。隣の物体までの距離）", extent=(x0, x0 + 2 * w, y0 + 2 * h, y0))
    sel = (c["centers"][:, 0] > x0) & (c["centers"][:, 0] < x0 + 2 * w) & (c["centers"][:, 1] > y0) & (c["centers"][:, 1] < y0 + 2 * h)
    for x, y in c["centers"][sel]:
        ax.add_patch(plt.Rectangle((x - half_w, y - half_h), 2 * half_w, 2 * half_h, fill=False, ec="yellow", lw=1.6))
        ax.plot(x, y, "+", color=C3, ms=11, mew=2.2)
    ax = axes[2]
    instance = c["instance_logit"] > 0
    base = cv2.resize(content, (instance.shape[1], instance.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    base[instance] = 0.5 * base[instance] + 0.5 * np.array([230, 25, 75])
    show(ax, base.astype(np.uint8), "③ 次の段階: セルの箱で 1 個ずつデコード\n（LatticeSeparator。detailed_segmentation.md）")
    finish(fig, path)


def fig_flow(d, path):
    """フロー全体の概観図: 各段階の画像を、矢印でつなぐ。"""
    from matplotlib.patches import FancyArrowPatch
    steps, content, r = d["steps"], d["content"], d["refine"]
    fig = plt.figure(figsize=(11.8, 6.5))
    w, h = 0.215, 0.355
    xs = [0.012, 0.262, 0.512, 0.762]
    ys = [0.56, 0.055]
    axes = []
    for row in range(2):
        for col in range(4):
            ax = fig.add_axes([xs[col], ys[row], w, h])
            axes.append(ax)
    titles = ["0. 入力と前景", "1. 勾配の大きさ", "2. 前景の中の自己相関", "3. ピークから v1・v2",
              "4. 判定（強さ・セル数）", "5. 精密化（512px）", "6. 位相（物体の中心）", "7. セルの中心と箱"]
    notes = ["前景の外は無視する", "縁の強さの繰り返しを見る", "v の整数倍でピーク", "最も強いものの 0.6 倍以上で最短",
             "強さ ≥ 0.4・セル ≥ 16 個で通過", "k·v のピークから最小二乗", "ロジットを周期で重ねて重心 + 補正", "箱 = ベクトル 1 つ分"]
    region = d["region"]
    # 0
    base = cv2.resize(content, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    base[region] = 0.55 * base[region] + 0.45 * np.array([230, 25, 75])
    axes[0].imshow(base.astype(np.uint8))
    # 1
    axes[1].imshow(steps["gradient"] * steps["mask"], cmap="magma")
    # 2・3
    corr = steps["corr"]
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    half = 110
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under("0.85")
    window = corr[cy - half:cy + half, cx - half:cx + half]
    axes[2].imshow(window, cmap=cmap, vmin=-0.2, vmax=1.0)
    axes[3].imshow(window, cmap=cmap, vmin=-0.2, vmax=1.0, extent=(-half, half, half, -half))
    v1, v2, _ = steps["found"]
    for color, v in ((C1, v1), (C2, v2)):
        axes[3].annotate("", (v[0], v[1]), (0, 0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6))
    axes[3].text(v1[0] + 4, v1[1] + 12, "v1", color=C1, fontsize=11, fontweight="bold", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1))
    axes[3].text(v2[0] + 5, v2[1] + 2, "v2", color=C2, fontsize=11, fontweight="bold", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1))
    # 4
    ax = axes[4]
    height, width = corr.shape
    at = lambda dx, dy: float(corr[max(int(round(height // 2 + dy)) - 2, 0):int(round(height // 2 + dy)) + 3, max(int(round(width // 2 + dx)) - 2, 0):int(round(width // 2 + dx)) + 3].max())
    vals = [at(*v1), at(2 * v1[0], 2 * v1[1]), at(*v2), at(2 * v2[0], 2 * v2[1])]
    ax.bar(["v1", "2v1", "v2", "2v2"], vals, color=[C1, C1, C2, C2], width=0.6)
    ax.axhline(LATTICE_MIN_STRENGTH, color="k", ls="--", lw=1.2)
    ax.axhline(min(vals), color=C3, lw=1.8)
    ax.text(3.45, 1.0, f"強さ = 最小 = {min(vals):.2f}", ha="right", va="top", color=C3, fontsize=9, fontweight="bold")
    ax.text(3.45, LATTICE_MIN_STRENGTH - 0.03, f"下限 {LATTICE_MIN_STRENGTH}", ha="right", va="top", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_ylim(0, 1.05)
    ax.set_position([xs[0] + 0.03, ys[1] + 0.045, w - 0.035, h - 0.045])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    # 5
    crop, extent, start_pts, refined_pts = corner_cells(r)
    axes[5].imshow(crop, extent=extent)
    axes[5].plot(*start_pts.T, "o", mfc="none", mec="yellow", ms=11, mew=2)
    axes[5].plot(*refined_pts.T, "+", color=C3, ms=10, mew=2.2)
    # 6
    p = d["phase"]
    axes[6].imshow(np.tile(p["template"], (2, 2)), cmap="viridis", extent=(0, 2, 2, 0))
    for dx in (0, 1):
        for dy in (0, 1):
            axes[6].plot(p["centre_frac"][0] + dx, p["centre_frac"][1] + dy, "x", color="red", ms=9, mew=2.2)
    # 7
    axes[7].imshow(content)
    half_w, half_h = d["half"]
    for x, y in d["centers"]:
        axes[7].add_patch(plt.Rectangle((x - half_w, y - half_h), 2 * half_w, 2 * half_h, fill=False, ec="yellow", lw=0.5))
    axes[7].plot(*d["centers"].T, ".", color=C3, ms=4)
    for i, (ax, title, note) in enumerate(zip(axes, titles, notes)):
        row, col = divmod(i, 4)
        if i != 4:
            ax.axis("off")
        fig.text(xs[col] + w / 2, ys[row] + h + 0.012, title, ha="center", va="bottom", fontsize=11, fontweight="bold")
        fig.text(xs[col] + w / 2, ys[row] - 0.012, note, ha="center", va="top", fontsize=9, color="0.25")
    # 矢印
    def arrow(a, b, **kw):
        fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=2, color="0.25", **kw))
    for row in range(2):
        for col in range(3):
            arrow((xs[col] + w + 0.004, ys[row] + h / 2), (xs[col + 1] - 0.004, ys[row] + h / 2))
    # 1 行目の右端から 2 行目の左端へ
    ymid = ys[0] - 0.065
    fig.add_artist(plt.Line2D([xs[3] + w / 2, xs[3] + w / 2, xs[0] + w / 2], [ys[0] - 0.045, ymid, ymid], transform=fig.transFigure, color="0.25", lw=2))
    arrow((xs[0] + w / 2, ymid), (xs[0] + w / 2, ys[1] + h + 0.045))
    fig.text(0.62, ys[0] + h / 2 + 0.205, "", fontsize=1)
    save_figure(fig, path)
    plt.close(fig)
    print("保存:", path)


# =============================================================================
# 本体
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--prefix", default="latdet")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config()
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    separator = runner.refiner.lattice_separator

    # ---- 006: 各段階 ----------------------------------------------------------
    data = run_pipeline(runner, Path(args.image))
    content, logit, feats = data["content"], data["logit"], data["feats"]
    region = data["semantic_map"] > 0
    input_hw = content.shape[:2]
    steps = detection_steps(content, region)
    v1, v2, strength = steps["found"]
    scale = steps["scale"]
    steps["cells"] = steps["mask"].sum() / abs(v1[0] * v2[1] - v1[1] * v2[0])
    lattice = L.detect(content, region)
    print(f"格子ベクトル（256px）v1={v1} v2={v2} 強さ={strength:.3f} セル数={steps['cells']:.1f}")
    print(f"精密化後（入力の画素）v1={lattice.v1} v2={lattice.v2}")

    fig_input(data, out / f"{args.prefix}_01_input.jpg")
    fig_gradient(steps, out / f"{args.prefix}_02_gradient.jpg")
    fig_autocorrelation(steps, out / f"{args.prefix}_03_autocorrelation.jpg")
    fig_vectors(steps, content, out / f"{args.prefix}_04_vectors.jpg")

    # ---- シリンジ 6 枚の強さ ----------------------------------------------------
    per_image = []
    for path in sorted(glob.glob(str(SYRINGE_DIR / "*_syringe.*"))):
        if "mask" in path:
            continue
        name = Path(path).name[:3]
        d = data if Path(path) == Path(args.image) else run_pipeline(runner, Path(path))
        if "semantic_map" not in d:
            per_image.append(dict(name=f"{name}_syringe", strength=None, cells=None, note=""))
            continue
        st = detection_steps(d["content"], d["semantic_map"] > 0)
        if st["found"] is None:
            per_image.append(dict(name=f"{name}_syringe", strength=None, cells=None, note=""))
            continue
        a, b, s_ = st["found"]
        cells = st["mask"].sum() / abs(a[0] * b[1] - a[1] * b[0])
        note = "分離が作動" if d["result"] is not None else "分離は何も変えない"
        per_image.append(dict(name=f"{name}_syringe", strength=s_, cells=cells, note=note))
        print(name, f"強さ {s_:.3f} セル {cells:.0f}", note)
    fig_gate(steps, per_image, out / f"{args.prefix}_05_gate.jpg")

    # ---- 精密化（フル解像度）------------------------------------------------------
    gradient, mask, full_scale = L._gradient_and_mask(content, region, max(content.shape[:2]))
    corr, valid = L._autocorrelation(gradient, mask)
    start1, start2 = (np.array(c) / scale * full_scale for c in (v1, v2))
    k1, lag1 = refine_samples(corr, valid, start1)
    k2, lag2 = refine_samples(corr, valid, start2)
    refined1 = np.array(L._refine_vector(corr, valid, tuple(start1)))
    refined2 = np.array(L._refine_vector(corr, valid, tuple(start2)))
    assert np.allclose(refined1 / full_scale, lattice.v1) and np.allclose(refined2 / full_scale, lattice.v2), "精密化が本体と一致しない"
    print(f"精密化: v1 {np.hypot(*start1):.2f} -> {np.hypot(*refined1):.2f}px, v2 {np.hypot(*start2):.2f} -> {np.hypot(*refined2):.2f}px（フル解像度 {full_scale * max(content.shape[:2]):.0f}px）")
    # ---- 位相 ------------------------------------------------------------------
    front = logit > 0
    template, weight, centre_frac, _ = phase_template(lattice, logit, front, input_hw)
    phase0 = L.find_phase(lattice, logit, front, input_hw)
    phase1, probes, centroids, found = correct_phase(separator, lattice, phase0, region, feats, input_hw)
    points0 = L.cell_centers(lattice, phase0, region, input_hw)
    points1 = L.cell_centers(lattice, phase1, region, input_hw)
    centers = data["result"]["centers"]
    assert np.allclose(points1, centers), "位相の補正が本体と一致しない"
    print(f"位相の補正: {np.median(np.linalg.norm(points1 - points0[: len(points1)], axis=1)):.1f}px（セル {len(points1)} 個）")
    fig_phase(dict(content=content, logit=logit, template=template, weight=weight, centre_frac=centre_frac, points0=points0, points1=points1,
                   probes=probes, v1=lattice.v1), out / f"{args.prefix}_07_phase.jpg")

    # ---- セルの中心と箱 ---------------------------------------------------------
    all_points = L.cell_centers(lattice, phase1, np.ones((8, 8), bool), input_hw)
    basis = L.basis(lattice)
    coef = np.linalg.solve(basis, (centers - phase1).T).T
    nx, ny = len(np.unique(np.round(coef[:, 0]))), len(np.unique(np.round(coef[:, 1])))
    fig_cells(dict(content=content, centers=centers, all_points=all_points, half=L.cell_half_size(lattice), nx=nx, ny=ny,
                   instance_logit=data["result"]["instance_logit"]), out / f"{args.prefix}_08_cells.jpg")

    # ---- 精密化の図（左上のセルの中心を原点に、最初のベクトルと精密化したベクトルが指す位置を比べる）-------------
    origin = centers[np.argmin(centers.sum(axis=1))]
    height, width = corr.shape
    refine_data = dict(corr=corr, cy=height // 2, cx=width // 2, start1=start1, start2=start2, refined1=refined1, refined2=refined2, k1=k1, lag1=lag1, k2=k2, lag2=lag2,
                       content=content, origin=origin, nx=nx, ny=ny, full_scale=full_scale)
    fig_refine(refine_data, out / f"{args.prefix}_06_refine.jpg")

    # ---- 概観図 -----------------------------------------------------------------
    fig_flow(dict(steps=steps, content=content, region=region, refine=refine_data, half=L.cell_half_size(lattice), centers=centers,
                  phase=dict(template=template, centre_frac=centre_frac)), out / f"{args.prefix}_00_flow.jpg")


if __name__ == "__main__":
    main()
