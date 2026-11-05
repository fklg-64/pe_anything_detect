"""長方形の物体の大きさ（長辺 L・短辺 S）を、隣り合う平行な縁の間隔のヒストグラムから求める（学習なし）。

画像を回し、向きごとに「縦に続く縁」の行の上の極大を探して、隣り合う縁の間隔を数える（縁の強さの小さい方で重みづけ）。向きに依らず、
物体の辺に当たる向きでだけ間隔が同じ値に揃うので、ヒストグラムの極大が物体の辺の長さになる。隣り合う縁だけを数えるので、2 個分の間隔（S の 2 倍）は現れない。
  S = 縦 RECT_SHORT_EDGE_LINE 画素続く縁のヒストグラムの基本周期（最大の RECT_PEAK_RATIO 倍以上の極大のうち最小の間隔）
  L の候補 = 縦 RECT_LONG_EDGE_LINE 画素続く縁（短辺の端まで数える）のヒストグラムの、S より十分長い間隔の極大
作業解像度 image_size = 512 のまま数える（256 に縮小すると、継ぎ目の弱い縁が消えて S の当たりが落ちる）。
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.common.params import (
    RECT_EDGE_NMS,
    RECT_FOREGROUND_DILATE,
    RECT_GAP_ANGLE_STEP,
    RECT_GRADIENT_SIGMA,
    RECT_HISTOGRAM_SMOOTH_SIGMA,
    RECT_LONG_EDGE_LINE,
    RECT_LONG_GAP_POWER,
    RECT_LONG_MIN_RATIO,
    RECT_MAX_GAP,
    RECT_MIN_GAP,
    RECT_PEAK_RATIO,
    RECT_SHORT_EDGE_LINE,
)


def gap_histograms(image: np.ndarray, foreground: np.ndarray, lines: Tuple[int, ...]) -> List[np.ndarray]:
    """隣り合う平行な縁の間隔 d（画素。2〜RECT_MAX_GAP）の重み付きヒストグラムを、縦に続く縁の長さ lines の各値について作る。

    image: 作業解像度の画像（RGB, uint8）。foreground: 前景（bool, 同じ大きさ）。重み = 2 つの縁の強さの小さい方。
    縁の強さが前景の中央値未満の極大と、間隔の中央が前景の外にある間隔は数えない。
    縁は向き a ごとに「回した画像の縦の縁」= cos a · Gx + sin a · Gy を回した地図（勾配は 1 回だけ求める）。
    返り値: lines の順の、長さ RECT_MAX_GAP + 1 の配列のリスト。
    """
    height, width = image.shape[:2]
    histograms = [np.zeros(RECT_MAX_GAP + 1, np.float64) for _ in lines]
    region = cv2.dilate(foreground.astype(np.uint8), np.ones((RECT_FOREGROUND_DILATE, RECT_FOREGROUND_DILATE), np.uint8))
    if not region.any():
        return histograms
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), RECT_GRADIENT_SIGMA)
    gx = [cv2.Sobel(blurred[..., c], cv2.CV_32F, 1, 0) for c in range(blurred.shape[2])]
    gy = [cv2.Sobel(blurred[..., c], cv2.CV_32F, 0, 1) for c in range(blurred.shape[2])]
    magnitude = np.sqrt(sum(a * a + b * b for a, b in zip(gx, gy)))
    floor = float(np.median(magnitude[region > 0]))   # 前景の中の縁の強さの中央値。これ未満は物体の縁ではない
    center = (width / 2 - 0.5, height / 2 - 0.5)
    across = np.ones((1, RECT_EDGE_NMS), np.uint8)
    along = [np.ones((line, 1), np.uint8) for line in lines]

    for angle in np.arange(0, 180, RECT_GAP_ANGLE_STEP):
        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        derivative = [cv2.addWeighted(a, cos_a, b, sin_a, 0) for a, b in zip(gx, gy)]
        edge = np.sqrt(sum(d * d for d in derivative))
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        edge = cv2.warpAffine(edge, rotation, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rotated_region = cv2.warpAffine(region, rotation, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        for histogram, kernel in zip(histograms, along):
            vertical = cv2.erode(edge, kernel)    # 縦に続く縁だけ残す
            peak = (vertical == cv2.dilate(vertical, across)) & (vertical >= floor) & (rotated_region > 0)
            peak[:, 0] = peak[:, -1] = False
            ys, xs = np.nonzero(peak)
            if len(xs) < 2:
                continue
            strength = vertical[ys, xs]
            gap = xs[1:] - xs[:-1]
            valid = (ys[1:] == ys[:-1]) & (gap >= 2) & (gap <= RECT_MAX_GAP)   # 同じ行の隣り合う極大
            valid &= rotated_region[ys[1:], (xs[1:] + xs[:-1]) // 2] > 0
            histogram += np.bincount(gap[valid], weights=np.minimum(strength[1:], strength[:-1])[valid], minlength=RECT_MAX_GAP + 1)
    return histograms


def pick_short(histogram: np.ndarray) -> Optional[float]:
    """ヒストグラムの極大のうち、最大の RECT_PEAK_RATIO 倍以上のものの中で最も小さい間隔（基本周期）。極大が無ければ None。"""
    peaks = [i for i in range(RECT_MIN_GAP, len(histogram) - 1) if histogram[i] >= histogram[i - 1] and histogram[i] > histogram[i + 1]]
    if not peaks:
        return None
    top = max(histogram[i] for i in peaks)
    return float(min(i for i in peaks if histogram[i] >= RECT_PEAK_RATIO * top))


def pick_long(histogram: np.ndarray, short: float) -> Optional[float]:
    """S の RECT_LONG_MIN_RATIO 倍より長い間隔の極大のうち、重み（間隔 ^ RECT_LONG_GAP_POWER × 値）が最大のもの。無ければ None。"""
    peaks = [i for i in range(int(RECT_LONG_MIN_RATIO * short), len(histogram) - 1) if histogram[i] >= histogram[i - 1] and histogram[i] > histogram[i + 1]]
    return float(max(peaks, key=lambda i: histogram[i] * i ** RECT_LONG_GAP_POWER)) if peaks else None


def estimate_gaps(image: np.ndarray, foreground: np.ndarray) -> Optional[Tuple[float, float]]:
    """縁の間隔のヒストグラムから (L の候補, S)（作業解像度の画素）を求める。S か L の候補が見つからなければ None。

    L の候補は、端と端が接して並ぶ配置では 2 個分の長さになることがある（RectSeparator が敷き詰めの質で直す）。
    """
    short_hist, long_hist = (
        cv2.GaussianBlur(h.astype(np.float32).reshape(1, -1), (0, 0), RECT_HISTOGRAM_SMOOTH_SIGMA).ravel()
        for h in gap_histograms(image, foreground, (RECT_SHORT_EDGE_LINE, RECT_LONG_EDGE_LINE))
    )
    short = pick_short(short_hist)
    if short is None:
        return None
    long = pick_long(long_hist, short)
    return None if long is None else (long, short)
