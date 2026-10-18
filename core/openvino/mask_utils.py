"""マスク・座標・点サンプリングの純粋なヘルパー関数（状態を持たない）。"""

from typing import Optional, List, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as ndi_label

from core.openvino.params import (
    BACKGROUND_POINT_MARGIN,
    MEDIAN_BLUR_KSIZE,
    PATCH_NEIGHBORHOOD,
    PATCH_STRUCTURE,
)

Component = Tuple[Tuple[int, int], np.ndarray, np.ndarray]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# =============================================================================
# 点のサンプリング
# =============================================================================

def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """
    points: (M, 2) の座標配列。各行は (x, y)
    k:      抽出する点の数

    return: (k, 2) の座標配列
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")

    m = len(points)

    if not 0 <= k <= m:
        raise ValueError("k must be between 0 and len(points)")

    if k == 0:
        return points[:0].copy()

    if k == m:
        return points.copy()

    # 距離計算用
    pts = points.astype(np.float64, copy=False)

    # 最初は重心に最も近い点を選ぶ
    center = pts.mean(axis=0)
    first = np.argmin(np.sum((pts - center) ** 2, axis=1))

    selected = np.empty(k, dtype=np.intp)
    selected[0] = first

    # 各点から、選択済みの点までの最小二乗距離
    min_dist2 = np.sum((pts - pts[first]) ** 2, axis=1)
    min_dist2[first] = -1

    for i in range(1, k):
        # 選択済みの点から最も遠い候補を選ぶ
        idx = np.argmax(min_dist2)
        selected[i] = idx

        # 新しく選んだ点との距離で最小距離を更新
        dist2 = np.sum((pts - pts[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

        # 重複選択を防ぐ
        min_dist2[selected[:i + 1]] = -1

    return points[selected]


def sample_points(candidate: np.ndarray, k: int) -> np.ndarray:
    """True の画素から最遠点サンプリングで最大 k 点を抽出する。座標は (x, y)。"""
    points = np.argwhere(candidate)[:, ::-1]
    return farthest_point_sampling(points, min(k, len(points)))


def prompt_points(core: np.ndarray, peaks: List[Tuple[int, int]], k: int) -> np.ndarray:
    """正例点を返す。各連結成分の最内点(peaks)を必ず含め、残りは core 上の最遠点で k 点まで補う。

    最内点を必ず含めることで、細い物体や小さい物体にも漏れなく点が置かれる。
    """
    selected = [np.asarray(p, dtype=np.float64) for p in peaks]
    points = np.argwhere(core)[:, ::-1].astype(np.float64)
    if len(points) and len(selected) < k:
        min_d2 = np.min(((points[:, None, :] - np.asarray(selected)[None]) ** 2).sum(-1), axis=1)
        while len(selected) < k and min_d2.max() > 0:
            idx = int(np.argmax(min_d2))
            selected.append(points[idx])
            min_d2 = np.minimum(min_d2, ((points - points[idx]) ** 2).sum(-1))
    return np.asarray(selected).reshape(-1, 2).astype(int)


def select_background_points(
    foreground_binary: np.ndarray, count: int, foreground_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """画像端につながる背景のうち、前景から十分離れた位置から負例点を count 点抽出する。

    物体内部の穴や影・縁の近くに負例を置くと、物体が欠けるため避ける。
    foreground_map を渡すと、背景と判定された中でも前景確信度が最も低い（背景だと最も確信できる）
    パッチから選ぶ。確信の弱い「背景」は、実際は物体（例: 濃い布）のことがあり、そこに負例を置くと
    物体を打ち消してしまうため。
    """
    background = ~foreground_binary
    labeled, _ = ndi_label(background)
    border_ids = np.unique(np.concatenate([labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]]))
    outer = np.isin(labeled, border_ids[border_ids > 0])
    if not outer.any():
        outer = background
    dist = distance_transform_edt(outer)
    for margin in (BACKGROUND_POINT_MARGIN, 1):
        candidate = outer & (dist >= margin)
        if candidate.any():
            break
    if foreground_map is not None:
        # 確信度の低い（値の小さい）パッチから count 個以上そろうところまでを候補にする
        for level in np.unique(foreground_map[candidate]):
            confident = candidate & (foreground_map <= level)
            if confident.sum() >= count:
                candidate = confident
                break
    return sample_points(candidate, count)


# =============================================================================
# マスクの解析・整形
# =============================================================================

def connected_components(mask: np.ndarray) -> List[Component]:
    """マスクの連結成分ごとに (最内点(x, y), core, 成分マスク) を返す。

    最内点は距離変換の最大点。core は成分を1パッチ内側へ縮めたもので、
    縮めると消える細い成分では成分そのものを core とする。
    """
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    result = []
    for i in range(1, n):
        region = comp == i
        edt = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(edt), edt.shape)
        core = binary_erosion(region, structure=PATCH_STRUCTURE, border_value=1)
        result.append(((int(x), int(y)), core if core.any() else region, region))
    return result


def clean_mask(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    mask: 2次元の二値マスク（0/1 または 0/255）
    min_area: 除去する連結成分の最小面積（ピクセル数）

    return: 0/1 の二値マスク
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)

    # 小さな独立成分を除去
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # 面積が足りる連結成分だけを残す（背景ラベル0は常に除外）
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    cleaned = keep[labels].astype(np.uint8)

    # 輪郭の細かなノイズを平滑化
    cleaned = cv2.medianBlur(cleaned, MEDIAN_BLUR_KSIZE)

    return cleaned


def remove_small_regions(semantic_map: np.ndarray, min_area: int) -> None:
    """セマンティックマップ上の小領域ノイズを除去する（in-place）。

    1. クラスごとに小さな連結成分を背景へ戻す。
    2. 背景に残った小さな穴を、周囲で最も多いクラスで埋める。
    """
    height, width = semantic_map.shape
    for class_id in np.unique(semantic_map):
        if class_id == 0:
            continue
        region = semantic_map == class_id
        keep = clean_mask(region, min_area).astype(bool)
        semantic_map[region & ~keep] = 0

    n, comp, stats, _ = cv2.connectedComponentsWithStats(
        (semantic_map == 0).astype(np.uint8), connectivity=8
    )
    ring_kernel = np.ones((3, 3), dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            continue
        x, y, w, h = stats[i, :4]
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1, y1 = min(x + w + 1, width), min(y + h + 1, height)
        hole = comp[y0:y1, x0:x1] == i
        ring = cv2.dilate(hole.astype(np.uint8), ring_kernel).astype(bool) & ~hole
        neighbors = semantic_map[y0:y1, x0:x1][ring]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size:
            semantic_map[y0:y1, x0:x1][hole] = np.bincount(neighbors).argmax()


# =============================================================================
# パッチ格子 <-> 作業解像度の変換
# =============================================================================

def patch_mean(patch_map: np.ndarray) -> np.ndarray:
    """パッチ近傍(PATCH_NEIGHBORHOOD四方)の平均。境界は scipy.ndimage の既定(reflect)と同じ。"""
    return cv2.blur(
        patch_map.astype(np.float32), (PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD),
        borderType=cv2.BORDER_REFLECT,
    )


def resize_map(patch_map: np.ndarray, size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """パッチ格子のマップを size=(幅, 高さ) へ拡大する。"""
    return cv2.resize(patch_map.astype(np.float32), size, interpolation=interpolation)
