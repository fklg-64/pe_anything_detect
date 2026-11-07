"""パッチ格子とマスクの基本演算（粗い段階・詳細段階で共通。状態を持たない純粋関数）。

段階に固有の処理は、それぞれのパッケージにある: 粗い段階 = coarse/、詳細段階（点の選び方・マスクの整形など）= detail/。
"""

from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from core.common.params import PATCH_NEIGHBORHOOD


# =============================================================================
# 型
# =============================================================================

class Component(NamedTuple):
    """マスクの連結成分1つ分（タプルとしても (peak, core, region) の順に取り出せる）。"""

    peak: Tuple[int, int]   # 最内点 (x, y)。距離変換が最大の位置
    core: np.ndarray        # 成分を1パッチ内側へ縮めたマスク（縮めると消える細い成分では成分そのもの）
    region: np.ndarray      # 成分のマスク


def as_uint8(mask: np.ndarray) -> np.ndarray:
    """bool（または 0/非 0 の整数）のマスクを、コピーせずに（bool なら）uint8 の 0/1 として読む。cv2 の関数は bool を受け付けない。"""
    return np.ascontiguousarray(mask, dtype=bool).view(np.uint8)


# =============================================================================
# パッチ近傍の演算
# =============================================================================

def patch_kernel() -> np.ndarray:
    """パッチ近傍（PATCH_NEIGHBORHOOD 四方）の構造要素（uint8）。erode / dilate に渡す。"""
    return np.ones((PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD), dtype=np.uint8)


def erode(mask: np.ndarray) -> np.ndarray:
    """マスクをパッチ近傍で 1 回縮める（bool）。画像の外は前景として扱うので、画像の端では縮まない。"""
    return cv2.erode(as_uint8(mask), patch_kernel()).view(bool)


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """マスクをパッチ近傍で膨らませる（bool）。画像の外は背景として扱う。"""
    return cv2.dilate(as_uint8(mask), patch_kernel(), iterations=iterations).view(bool)


def patch_mean(patch_map: np.ndarray) -> np.ndarray:
    """パッチ近傍(PATCH_NEIGHBORHOOD四方)の平均。境界は端の画素を折り返す(reflect)。"""
    return cv2.blur(
        patch_map.astype(np.float32), (PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD),
        borderType=cv2.BORDER_REFLECT,
    )


# =============================================================================
# 連結成分と距離
# =============================================================================

def squared_distance_to_background(mask: np.ndarray) -> np.ndarray:
    """前景の各画素から、最も近い背景（False）の画素までのユークリッド距離の二乗（int32。画像の外は背景ではない）。

    cv2 の距離は float32 で、等しい距離でも最後の桁がずれて同点の位置が変わる（scipy と最内点が変わる）ので、二乗して整数に丸める。
    """
    distance = cv2.distanceTransform(as_uint8(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.rint(distance * distance).astype(np.int32)


def connected_components(mask: np.ndarray) -> List[Component]:
    """マスクの連結成分ごとに (最内点(x, y), core, 成分マスク) を返す。

    最内点は距離変換の最大点。core は成分を1パッチ内側へ縮めたもので、
    縮めると消える細い成分では成分そのものを core とする。
    """
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    result = []
    for i in range(1, n):
        region = comp == i
        edt = squared_distance_to_background(np.pad(region, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(edt), edt.shape)
        core = erode(region)
        result.append(Component((int(x), int(y)), core if core.any() else region, region))
    return result


def bounding_slices(mask: np.ndarray, margin: int = 1) -> Optional[Tuple[slice, slice]]:
    """マスクの外接矩形（上下左右に margin 画素の余白つき、画像内に収める）を (行, 列) のスライスで返す。空なら None。"""
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    height, width = mask.shape
    return (
        slice(max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, height)),
        slice(max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, width)),
    )


# =============================================================================
# パッチ格子 <-> 作業解像度
# =============================================================================

def resize_map(patch_map: np.ndarray, size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """パッチ格子のマップを size=(幅, 高さ) へ拡大する。"""
    return cv2.resize(patch_map.astype(np.float32), size, interpolation=interpolation)


def resize_mask(patch_mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """パッチ格子の bool マスクを size=(幅, 高さ) へ最近傍で拡大する（`resize_map(..., INTER_NEAREST) > 0` と同じ結果。uint8 のまま拡大するので速い）。"""
    return cv2.resize(patch_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).view(bool)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """ガイデッドフィルタ（He ら 2010）の輝度（1チャンネル）ガイド版。ボックスフィルタと四則演算だけ。

    src を、guide（同じ大きさの float32、0〜1）の縁に沿うように整える。窓（半径 radius）の中で src ≒ a × guide + b と
    当てはめ、窓ごとの (a, b) を平均して出力する。guide が平坦な所は平滑化、縁のある所は縁を保つ。
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)

    mean_i, mean_p = box(guide), box(src)
    a = (box(guide * src) - mean_i * mean_p) / (box(guide * guide) - mean_i * mean_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)
