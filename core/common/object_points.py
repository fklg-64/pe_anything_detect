"""前景のプロンプト点を、EdgeTAM の高解像度特徴量の画素から選ぶ（点を物体の上に合わせる）。

粗いマスクはパッチ単位（1 パッチ = 16 画素）で、縁のパッチは物体と背景の両方を含む。パッチの中心に点を置くと、縁や細い物体では
点が背景に落ちたり、物体の一部（歯ブラシの毛だけ・ヘッドの縁や柄が欠ける）に偏ったりする。そこで、点をパッチの中心ではなく、
EdgeTAM の高解像度特徴量（入力の 1/4）の画素から選ぶ。

  物体らしさ = 物体の特徴（クラスの core の平均）とのコサイン類似度 − 背景の特徴（粗い前景から離れた領域の平均）とのコサイン類似度
               （EdgeTAM の特徴量 3 層で求めて、最も細かい層の大きさで平均する）
  候補       = 粗いマスク全体（縁のパッチも含む）のうち、物体らしさが正の画素
  選び方     = 各連結成分の最内点（そのパッチの中で物体らしさが最大の画素）を必ず含め、残りは「選択済みの点からの距離 × 物体らしさ」が最大の画素を順に足す

距離だけで選ぶと（物体らしさで重みづけしないと）縁の背景寄りの画素に点が寄り、合成データで下がる。パラメータは増やさない。
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from core.common.mask_utils import patch_structure
from core.common.params import BACKGROUND_POINT_MARGIN


def normalize_features(feats: Sequence[np.ndarray]) -> List[np.ndarray]:
    """EdgeTAM の特徴量（high_res_feats0 / high_res_feats1 / image_embed）を、画素ごとに L2 正規化した (C, H, W) の配列にする。"""
    levels = []
    for feat in feats:
        array = np.asarray(feat, dtype=np.float32)
        array = array[0] if array.ndim == 4 else array
        levels.append(array / (np.linalg.norm(array, axis=0, keepdims=True) + 1e-8))
    return levels


def mean_feature(level: np.ndarray, patch_mask: np.ndarray) -> Optional[np.ndarray]:
    """パッチ格子のマスクの内側にある特徴の平均（L2 正規化）。マスクが空なら None。level は (C, H, W)。"""
    _, height, width = level.shape
    mask = cv2.resize(patch_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    if not mask.any():
        return None
    mean = level[:, mask].mean(axis=1)
    return mean / (np.linalg.norm(mean) + 1e-8)


def background_features(levels: Sequence[np.ndarray], foreground: np.ndarray) -> Optional[List[np.ndarray]]:
    """背景の特徴（層ごとの平均）。背景 = 粗い前景から BACKGROUND_POINT_MARGIN パッチ以上離れた領域（無ければ前景以外）。作れなければ None。

    foreground: 粗い前景マスク（パッチ格子、全クラス）。画像に 1 回求めれば、全クラスで共通に使える。
    """
    background = ~binary_dilation(foreground, patch_structure(), iterations=BACKGROUND_POINT_MARGIN)
    if not background.any():
        background = ~foreground
    means = [mean_feature(level, background) for level in levels]
    return None if any(m is None for m in means) else means


def objectness_map(levels: Sequence[np.ndarray], background: Sequence[np.ndarray], core: np.ndarray) -> Optional[np.ndarray]:
    """物体らしさのマップ（最も細かい層の大きさ）。正なら、背景の特徴より core の特徴に近い。core が空なら None。

    levels: normalize_features の結果。background: background_features の結果。core: クラスの core（パッチ格子）。
    """
    height, width = levels[0].shape[1:]
    maps = []
    for level, background_mean in zip(levels, background):
        foreground_mean = mean_feature(level, core)
        if foreground_mean is None:
            return None
        score = np.tensordot(foreground_mean - background_mean, level, axes=(0, 0))
        maps.append(score if score.shape == (height, width) else cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR))
    return np.mean(maps, axis=0)


def object_prompt_points(
    score: np.ndarray, region: np.ndarray, peaks: List[Tuple[int, int]], k: int, grid: Tuple[int, int]
) -> np.ndarray:
    """正例点（パッチ座標 (x, y)。小数で、整数がパッチの中心）を返す。各連結成分の最内点を必ず含め、残りは k 点まで補う。

    score: objectness_map の結果。region: 粗いマスク（パッチ格子。縁のパッチを含む全体）。peaks: 各連結成分の最内点（パッチ座標）。grid: パッチ格子 (高さ, 幅)。
    """
    height, width = score.shape
    grid_h, grid_w = grid

    # 最内点は、そのパッチの中で物体らしさが最大の画素へ動かす
    selected = []
    for x, y in peaks:
        u0, u1 = int(np.floor(x * width / grid_w)), min(width, int(np.ceil((x + 1) * width / grid_w)))
        v0, v1 = int(np.floor(y * height / grid_h)), min(height, int(np.ceil((y + 1) * height / grid_h)))
        j, i = np.unravel_index(np.argmax(score[v0:v1, u0:u1]), (v1 - v0, u1 - u0))
        selected.append((u0 + i, v0 + j))

    candidate = (cv2.resize(region.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0) & (score > 0)
    if candidate.any() and len(selected) < k:
        pixels = np.argwhere(candidate)[:, ::-1].astype(np.float64)  # (u, v)
        weight = score[candidate]
        d2 = np.min(((pixels[:, None, :] - np.asarray(selected, dtype=np.float64)[None]) ** 2).sum(-1), axis=1)
        while len(selected) < k and d2.max() > 0:
            idx = int(np.argmax(np.sqrt(d2) * weight))
            selected.append(tuple(pixels[idx]))
            d2 = np.minimum(d2, ((pixels - pixels[idx]) ** 2).sum(-1))

    # 画素の中心 -> パッチ座標
    return np.array([((u + 0.5) * grid_w / width - 0.5, (v + 0.5) * grid_h / height - 0.5) for u, v in selected], dtype=np.float64).reshape(-1, 2)
