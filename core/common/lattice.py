"""規則的に並んだ同じ物体（トレイ・巣・ブリスターなど）の格子の検出と、格子点（物体の中心）の求め方（NumPy。torch / openvino に依存しない）。

物体の縁の強さ（画像の勾配の大きさ）の自己相関は、規則的に並んだ物体では格子のずれ（v, 2v, ...）で強いピークを持つ。
単一の物体や不規則な配置では、強いピークは出ない。自己相関は前景の領域の中だけで取る（領域の外は無視する）。
"""

from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

from core.common.params import (
    LATTICE_DETECT_SIZE,
    LATTICE_MIN_CELLS,
    LATTICE_MIN_LAG,
    LATTICE_MIN_STRENGTH,
    LATTICE_PEAK_RADIUS,
    LATTICE_PEAK_RATIO,
    LATTICE_PHASE_BINS,
)


class Lattice(NamedTuple):
    """格子ベクトルは入力画像の画素 (x, y)。"""

    v1: Tuple[float, float]
    v2: Tuple[float, float]
    strength: float   # 2 つのベクトルの v と 2v の自己相関の最小値（1 に近いほど規則的）


def _autocorrelation(gradient: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """領域内の勾配の大きさの正規化自己相関（中心が遅れ 0）と、重なりが十分な遅れのマスク。マスクの外は無視するので、重なった画素数で割る。"""
    x = ((gradient - (gradient * mask).sum() / mask.sum()) * mask).astype(np.float32)
    shape = (2 * x.shape[0], 2 * x.shape[1])
    spec = sfft.rfft2(x, s=shape)  # float32 で計算する（毎回実行する判定なので速くする）
    corr = sfft.irfft2(spec * np.conj(spec), s=shape)
    spec_mask = sfft.rfft2(mask.astype(np.float32), s=shape)
    overlap = sfft.irfft2(spec_mask * np.conj(spec_mask), s=shape)
    corr = sfft.fftshift(corr / np.maximum(overlap, 1))
    valid = sfft.fftshift(overlap) > 0.25 * mask.sum()
    corr = corr / corr[shape[0] // 2, shape[1] // 2]
    corr[~valid] = -1.0
    return corr, valid


def _gradient_and_mask(image: np.ndarray, region: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """画像を長辺 size に縮小して、勾配の大きさと領域のマスクを返す。3 つ目は縮小率。"""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    gray = cv2.cvtColor(cv2.resize(image, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5 * size / 512)
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    mask = cv2.resize(region.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return gradient, mask, scale


def _lattice_vectors(
    corr: np.ndarray, valid: np.ndarray, min_lag: float
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """自己相関のピークから、独立な 2 本の最短のベクトルと、強さ（v と 2v の小さいほう）を返す。"""
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    # ピークは、相関が正で、近傍（半径 LATTICE_PEAK_RADIUS）の最大のもの。相関が高い画素（全体の一部）に限って、最大値フィルタをかける
    size = 2 * LATTICE_PEAK_RADIUS + 1
    candidate = valid & (corr > 0.05)
    rows, cols = np.nonzero(candidate)
    if not len(rows):
        return None
    top, left = max(rows.min() - LATTICE_PEAK_RADIUS, 0), max(cols.min() - LATTICE_PEAK_RADIUS, 0)
    bottom, right = rows.max() + LATTICE_PEAK_RADIUS + 1, cols.max() + LATTICE_PEAK_RADIUS + 1
    crop = (slice(top, bottom), slice(left, right))
    local_max = np.zeros_like(candidate)
    local_max[crop] = corr[crop] == ndi.maximum_filter(corr[crop], size=size)
    peaks = local_max & candidate
    coords = np.argwhere(peaks) - np.array([cy, cx])
    values = corr[peaks]
    far = np.hypot(coords[:, 0], coords[:, 1]) >= min_lag
    coords, values = coords[far], values[far]
    if not len(values):
        return None

    def at(dx: float, dy: float) -> float:
        x, y = int(round(cx + dx)), int(round(cy + dy))
        if not (0 <= x < width and 0 <= y < height):
            return -1.0
        return float(corr[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].max())

    order = np.argsort(-values)[:16]
    candidates = [(float(coords[i, 1]), float(coords[i, 0]), float(values[i])) for i in order]
    # 最も強いピークの LATTICE_PEAK_RATIO 倍以上のピークのうち、短いものから選ぶ（強いピークが 2v・3v のときも、v を選ぶ）
    strongest = max(c[2] for c in candidates)
    candidates = sorted((c for c in candidates if c[2] >= LATTICE_PEAK_RATIO * strongest), key=lambda c: np.hypot(c[0], c[1]))
    first = candidates[0]
    second = next(
        (c for c in candidates[1:] if abs(first[0] * c[1] - first[1] * c[0]) > 0.3 * np.hypot(*first[:2]) * np.hypot(*c[:2])), None
    )
    if second is None:
        return None
    strength = min(min(at(c[0], c[1]), at(2 * c[0], 2 * c[1])) for c in (first, second))
    return first[:2], second[:2], strength


def _refine_vector(corr: np.ndarray, valid: np.ndarray, v: Tuple[float, float], max_multiple: int = 12) -> Tuple[float, float]:
    """格子ベクトルを、k·v（k 倍）のピーク位置から最小二乗で精密化する（k が大きいほど、整数画素の丸め誤差が 1/k になる）。"""
    height, width = corr.shape
    cy, cx = height // 2, width // 2
    multiples, lags = [], []
    for k in range(1, max_multiple + 1):
        x0, x1 = int(round(cx + k * v[0])) - 3, int(round(cx + k * v[0])) + 4
        y0, y1 = int(round(cy + k * v[1])) - 3, int(round(cy + k * v[1])) + 4
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            break
        window = corr[y0:y1, x0:x1]
        if window.max() < 0.25 or not valid[y0:y1, x0:x1].all():
            break
        iy, ix = np.unravel_index(np.argmax(window), window.shape)
        multiples.append(k)
        lags.append((x0 + ix - cx, y0 + iy - cy))
    if not multiples:
        return v
    k, lag = np.array(multiples, float), np.array(lags, float)
    result = (k[:, None] * lag).sum(0) / (k ** 2).sum()
    return float(result[0]), float(result[1])


def detect(image: np.ndarray, region: np.ndarray) -> Optional[Lattice]:
    """image（RGB, uint8）の region（bool。前景）の中に、規則的に並んだ物体の格子があれば返す。無ければ None。

    まず小さい画像（LATTICE_DETECT_SIZE）で、格子が強いか（LATTICE_MIN_STRENGTH）・前景に十分な数のセルが入るか（LATTICE_MIN_CELLS）を
    調べる（毎回実行する判定なので軽くする）。通ったときだけ、画像の大きさで格子ベクトルを精密化する。
    """
    if region.sum() == 0:
        return None
    gradient, mask, scale = _gradient_and_mask(image, region, LATTICE_DETECT_SIZE)
    if mask.sum() == 0:  # 縮小で領域が消えた（極小の前景）
        return None
    corr, valid = _autocorrelation(gradient, mask)
    found = _lattice_vectors(corr, valid, LATTICE_MIN_LAG * scale)
    if found is None or found[2] < LATTICE_MIN_STRENGTH:
        return None
    v1, v2, strength = found
    cell_area = abs(v1[0] * v2[1] - v1[1] * v2[0])
    if cell_area <= 0 or mask.sum() / cell_area < LATTICE_MIN_CELLS:
        return None

    # 通ったので、画像の大きさ（長辺）で精密化する
    full_size = max(image.shape[:2])
    gradient, mask, full_scale = _gradient_and_mask(image, region, full_size)
    corr, valid = _autocorrelation(gradient, mask)
    start = tuple(c / scale * full_scale for c in v1), tuple(c / scale * full_scale for c in v2)
    v1_full, v2_full = (_refine_vector(corr, valid, v) for v in start)
    return Lattice(
        (v1_full[0] / full_scale, v1_full[1] / full_scale), (v2_full[0] / full_scale, v2_full[1] / full_scale), strength
    )


def basis(lattice: Lattice) -> np.ndarray:
    """格子ベクトルを列に持つ 2x2 行列。格子の係数 (i, j) を入力画像の画素 (x, y) に直す。"""
    return np.array([[lattice.v1[0], lattice.v2[0]], [lattice.v1[1], lattice.v2[1]]], dtype=np.float64)


def find_phase(lattice: Lattice, values: np.ndarray, mask: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """物体の中心を通る格子点の位置（入力画像の画素 (x, y)）。

    values（物体らしさ。値が大きいほど物体の中心側。格子状の低解像度マップ）を格子の周期で重ねた平均（セルのテンプレート）の、
    明るい側（最大と最小の中点より上）の周期的な重心を、物体の中心とする。mask は values の中の前景（bool）。input_size: (高さ, 幅)。
    """
    grid_h, grid_w = values.shape
    ys, xs = np.nonzero(mask)
    points = np.stack([(xs + 0.5) * input_size[1] / grid_w, (ys + 0.5) * input_size[0] / grid_h], axis=1)
    grid_basis = basis(lattice)
    coef = np.linalg.solve(grid_basis, points.T).T
    frac = coef - np.floor(coef)
    bins = np.minimum((frac * LATTICE_PHASE_BINS).astype(int), LATTICE_PHASE_BINS - 1)
    total = np.zeros((LATTICE_PHASE_BINS, LATTICE_PHASE_BINS))
    count = np.zeros((LATTICE_PHASE_BINS, LATTICE_PHASE_BINS))
    np.add.at(total, (bins[:, 1], bins[:, 0]), values[ys, xs])
    np.add.at(count, (bins[:, 1], bins[:, 0]), 1)
    template = total / np.maximum(count, 1)
    weight = np.maximum(template - (template.max() + template.min()) / 2, 0)
    centres = (np.arange(LATTICE_PHASE_BINS) + 0.5) / LATTICE_PHASE_BINS
    centre_frac = []
    for axis_weight in (weight.sum(0), weight.sum(1)):  # 列方向（x）→ 行方向（y）
        angle = np.arctan2((axis_weight * np.sin(2 * np.pi * centres)).sum(), (axis_weight * np.cos(2 * np.pi * centres)).sum())
        centre_frac.append((angle / (2 * np.pi)) % 1.0)
    return grid_basis @ np.array(centre_frac)


def cell_centers(lattice: Lattice, phase: np.ndarray, region: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """phase を通る格子点のうち、画像の内側で region（bool。任意の大きさ）が真のもの。(N, 2)（入力画像の画素 (x, y)）。"""
    height, width = input_size
    grid_basis = basis(lattice)
    extent = int(np.ceil(max(height, width) / max(np.hypot(*lattice.v1), np.hypot(*lattice.v2)))) + 2
    ii, jj = np.meshgrid(np.arange(-extent, extent + 1), np.arange(-extent, extent + 1))
    points = phase[None] + (grid_basis @ np.stack([ii.ravel(), jj.ravel()])).T
    inside = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    points = points[inside]
    rows = np.minimum((points[:, 1] / height * region.shape[0]).astype(int), region.shape[0] - 1)
    cols = np.minimum((points[:, 0] / width * region.shape[1]).astype(int), region.shape[1] - 1)
    return points[region[rows, cols]]


def cell_half_size(lattice: Lattice) -> Tuple[float, float]:
    """セルの箱の半分の幅・高さ（入力画像の画素）。格子ベクトルの 1 つ分（隣の物体までの距離）が箱の一辺。"""
    return (
        max(abs(lattice.v1[0]), abs(lattice.v2[0])) / 2,
        max(abs(lattice.v1[1]), abs(lattice.v2[1])) / 2,
    )
