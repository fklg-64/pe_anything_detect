"""大きさ（長辺 L・短辺 S）が分かっている長方形を、前景に敷き詰める（学習なし）。

縮小した画像（RECT_TILE_HALF など）で、向きごとに長方形 (L, S, 向き) を各位置に置いたときの
  coverage = 長方形の中の前景の割合
  long_edge / short_edge = 長辺 2 本 / 短辺 2 本の縁の強さ（辺の法線方向の色の勾配の、辺に沿った平均）
の地図を、回した長方形のカーネルの相関で求める。点数 = coverage ^ RECT_COVERAGE_POWER × 縁の強さ（長辺・短辺を辺の長さで重みづけ）の極大から、
点数の高い順に、前景の割合が RECT_MIN_COVERAGE 以上で、すでに選んだ長方形との重なりが自分の面積の RECT_MAX_OVERLAP 以下のものを選ぶ（貪欲な敷き詰め）。
敷き詰めの質（tiling_quality）は、L の候補の中から正しい長さを選ぶのに使う。
"""

from typing import Dict, List, NamedTuple, Sequence

import cv2
import numpy as np

from core.common.params import (
    RECT_CANDIDATE_WINDOW,
    RECT_COVERAGE_POWER,
    RECT_EDGE_CLIP,
    RECT_EDGE_SCALE_PERCENTILE,
    RECT_GRADIENT_SIGMA,
    RECT_IMAGE_SIZE,
    RECT_MAX_OVERLAP,
    RECT_MIN_COVERAGE,
)

KERNEL_SUPERSAMPLE = 4   # カーネルを描くときの拡大（縁の線の太さは縮小後に 1 画素）


class RectCandidate(NamedTuple):
    """敷き詰めで選んだ長方形（作業解像度の画素。向きは度で、長辺が x 軸から時計回りに傾く角）。"""

    cx: float
    cy: float
    theta: float
    score: float
    coverage: float      # 長方形の中の前景の割合
    long_edge: float     # 長辺 2 本の縁の強さ（法線 = 短軸）
    short_edge: float    # 短辺 2 本の縁の強さ（法線 = 長軸）


class EdgeMaps:
    """縮小した画像の、長さ・向きに依らない地図（前景・色の勾配）。向きごとの縁の強さは覚えておいて、大きさの違う長方形でも使い回す。"""

    def __init__(self, image: np.ndarray, foreground: np.ndarray, half: int):
        self.half = half
        small = cv2.GaussianBlur(cv2.resize(image, (half, half), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), RECT_GRADIENT_SIGMA)
        self.gx = np.stack([cv2.Sobel(small[..., c], cv2.CV_32F, 1, 0) for c in range(3)])
        self.gy = np.stack([cv2.Sobel(small[..., c], cv2.CV_32F, 0, 1) for c in range(3)])
        self.foreground = cv2.resize(foreground.astype(np.float32), (half, half), interpolation=cv2.INTER_AREA)
        self.scale = float(np.percentile(np.sqrt((self.gx ** 2 + self.gy ** 2).sum(0)), RECT_EDGE_SCALE_PERCENTILE)) + 1e-6   # 画像全体の縁の強さ
        self._directional: Dict[float, np.ndarray] = {}

    def edge(self, angle: float) -> np.ndarray:
        """向き angle（度）の方向の色の勾配の強さ（チャンネルのノルム）。"""
        if angle not in self._directional:
            theta = np.deg2rad(angle)
            self._directional[angle] = np.sqrt(((np.cos(theta) * self.gx + np.sin(theta) * self.gy) ** 2).sum(0))
        return self._directional[angle]


def rect_corners(cx: float, cy: float, length: float, width: float, theta: float) -> np.ndarray:
    """長方形 (中心, 長辺, 短辺, 向き) の 4 隅 (4, 2)。"""
    t = np.deg2rad(theta)
    rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    local = np.array([[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]])
    return local @ rotation.T + [cx, cy]


def rect_kernel(length: float, width: float, theta: float, kind: str, half: int) -> np.ndarray:
    """縮小した画像（一辺 half）の座標での、奇数サイズ・総和 1 のカーネル。kind: 'rect'（塗りつぶし）/ 'short'（短辺 2 本）/ 'long'（長辺 2 本）。"""
    a, b = length * half / (2 * RECT_IMAGE_SIZE), width * half / (2 * RECT_IMAGE_SIZE)      # 半長・半幅
    r = int(np.ceil(np.hypot(a, b))) + 2
    n = 2 * r + 1
    scale = KERNEL_SUPERSAMPLE
    big = np.zeros((n * scale, n * scale), np.float32)
    c = np.array([r + 0.5, r + 0.5]) * scale
    u = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    v = np.array([-u[1], u[0]])
    a, b = a * scale, b * scale
    if kind == "rect":
        poly = np.array([c - u * a - v * b, c + u * a - v * b, c + u * a + v * b, c - u * a + v * b])
        cv2.fillPoly(big, [np.round(poly).astype(np.int32)], 1.0)
    else:
        for sign in (-1, 1):
            if kind == "short":
                p0, p1 = c + sign * u * a - v * b, c + sign * u * a + v * b
            else:
                p0, p1 = c + sign * v * b - u * a, c + sign * v * b + u * a
            cv2.line(big, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1.0, scale)
    kernel = cv2.resize(big, (n, n), interpolation=cv2.INTER_AREA)
    return kernel / max(kernel.sum(), 1e-6)


def tile(maps: EdgeMaps, length: float, width: float, angle_step: float, top: int) -> List[RectCandidate]:
    """長方形 (length, width) を前景に敷き詰めて、選んだ長方形のリスト（点数の高い順）を返す。top: 点数の上位から調べる候補の数。"""
    half = maps.half
    scale_back = RECT_IMAGE_SIZE / half
    weight_long, weight_short = length / (length + width), width / (length + width)
    window = np.ones((RECT_CANDIDATE_WINDOW, RECT_CANDIDATE_WINDOW), np.uint8)
    columns: List[np.ndarray] = []      # 向きごとの極大: [x, y, 向き, 点数, 前景の割合, 長辺の縁, 短辺の縁]
    for theta in np.arange(0, 180, angle_step):
        coverage = cv2.filter2D(maps.foreground, -1, rect_kernel(length, width, theta, "rect", half), borderType=cv2.BORDER_CONSTANT)
        short_edge = cv2.filter2D(maps.edge(theta), -1, rect_kernel(length, width, theta, "short", half), borderType=cv2.BORDER_REPLICATE)   # 短辺の法線は長軸
        long_edge = cv2.filter2D(maps.edge(theta + 90), -1, rect_kernel(length, width, theta, "long", half), borderType=cv2.BORDER_REPLICATE)
        long_edge, short_edge = np.minimum(long_edge / maps.scale, RECT_EDGE_CLIP), np.minimum(short_edge / maps.scale, RECT_EDGE_CLIP)
        score = np.clip(coverage, 0, 1) ** RECT_COVERAGE_POWER * (weight_long * long_edge + weight_short * short_edge)
        ys, xs = np.nonzero((score == cv2.dilate(score, window)) & (coverage >= RECT_MIN_COVERAGE))
        columns.append(np.stack([xs, ys, np.full(len(xs), theta), score[ys, xs], coverage[ys, xs], long_edge[ys, xs], short_edge[ys, xs]]).astype(np.float64))
    found = np.concatenate(columns, axis=1)[:, np.argsort(-np.concatenate([c[3] for c in columns]), kind="stable")[:top]]    # 点数の降順（同点は向き・位置の順）の上位 top 個
    centers = (found[:2].T + 0.5) * scale_back
    angles = np.deg2rad(found[2])
    local = np.array([[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]])
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    polygons = np.round(np.stack([local[:, 0] * cos - local[:, 1] * sin, local[:, 0] * sin + local[:, 1] * cos], axis=-1) + centers[:, None, :]).astype(np.int32)
    lows, highs = np.maximum(polygons.min(1), 0), np.minimum(polygons.max(1) + 1, RECT_IMAGE_SIZE)

    occupied = np.zeros((RECT_IMAGE_SIZE, RECT_IMAGE_SIZE), np.uint8)
    chosen: List[RectCandidate] = []
    for k in range(found.shape[1]):
        (x0, y0), (x1, y1) = lows[k], highs[k]
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [polygons[k] - [x0, y0]], 1)
        if (mask & occupied[y0:y1, x0:x1]).sum() / max(int(mask.sum()), 1) > RECT_MAX_OVERLAP:
            continue
        chosen.append(RectCandidate(float(centers[k, 0]), float(centers[k, 1]), float(found[2, k]), float(found[3, k]), float(found[4, k]), float(found[5, k]), float(found[6, k])))
        occupied[y0:y1, x0:x1] |= mask
    return chosen


def tiling_quality(chosen: Sequence[RectCandidate], length: float, width: float, foreground_area: int) -> float:
    """敷き詰めの質 = 覆う前景の割合 × 選んだ長方形の「弱い方の辺（長辺・短辺のうち弱い方）の縁の強さ」の平均。大きさが正しいほど高い。"""
    explained = sum(c.coverage * length * width for c in chosen) / max(foreground_area, 1)
    return explained * float(np.mean([min(c.long_edge, c.short_edge) for c in chosen]))
