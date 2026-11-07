"""EdgeTAM が切り出したマスクの整形（作業解像度の画素単位）。粗い（パッチ格子の）マスクではなく、精緻化の後のマスクが対象。

  clean_mask           1 領域の 2 値マスクのノイズ除去（1 パッチ未満の成分を除き、輪郭をメディアンフィルタでならす）  RegionEstimator.estimate が使う
  remove_small_regions 全クラスを重ね合わせたセマンティックマップの小領域ノイズの除去                          SemanticRefiner.refine の後処理
  fill_narrow_gaps     クラス境界の割れ（前景内部の細い隙間）を埋める                                          同上

`min_area` は、粗いマスクの信頼限界（1 パッチ）に当たる作業解像度の画素数。1 パッチ未満の領域はノイズとして扱う。
"""

from functools import partial
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from core.common.mask_utils import bounding_slices, erode
from core.common.parallel import Workers
from core.common.params import CLEAN_MASK_MAX_PER_COMPONENT, GAP_FILL_PIXELS_PER_PATCH, MEDIAN_BLUR_KSIZE


# =============================================================================
# 1 領域の 2 値マスク
# =============================================================================

def clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    mask: 2次元の二値マスク（bool、または 0/1・0/255 の整数）
    min_area: 除去する連結成分の最小面積（ピクセル数）

    return: 0/1 の二値マスク（uint8。値は 0/1 だけなので、`.view(bool)` で bool として読める）

    処理は前景の外接矩形（余白1画素）の中だけで行う（外は0なので、連結成分もメディアンフィルタ（3x3）も画像全体で行うのと同じ結果になる）。
    """
    mask = np.asarray(mask)
    foreground = mask if mask.dtype == bool else mask > 0  # bool のまま扱う（整数との比較は、bool を int64 に広げるので遅い）
    box = bounding_slices(foreground)
    cleaned = np.zeros(foreground.shape, dtype=np.uint8)
    if box is None:
        return cleaned
    crop = foreground.view(np.uint8)[box]  # bool の 0/1 をコピーせず uint8 として読む

    # 小さな独立成分を除去
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        crop, connectivity=8
    )

    # 面積が足りない連結成分（背景ラベル0は除く）だけを消す。クロップ内の前景は、どれかの成分に属する
    small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_area) + 1
    if len(small) == 0:
        kept = crop
    elif len(small) <= CLEAN_MASK_MAX_PER_COMPONENT:
        kept = crop.copy()
        for i in small:  # 成分の外接矩形の中だけで消す
            x, y, w, h = stats[i, :4]
            kept[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i] = 0
    else:
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        kept = keep.astype(np.uint8)[labels]  # 成分数分の表を先に uint8 にしてから全画素へ引く（全画素の型変換を省く）

    # 輪郭の細かなノイズを平滑化
    cleaned[box] = cv2.medianBlur(kept, MEDIAN_BLUR_KSIZE)

    return cleaned


# =============================================================================
# セマンティックマップ（全クラスの重ね合わせの後）
# =============================================================================

def _small_pieces(semantic_map: np.ndarray, class_id: int, min_area: int) -> Tuple[Tuple[slice, slice], np.ndarray]:
    """クラス class_id の小さな連結成分（clean_mask が除く画素）を、外接矩形と、その中のマスクで返す。semantic_map は読むだけ。"""
    region = semantic_map == class_id
    box = bounding_slices(region)  # 外接矩形の中だけで処理する（clean_mask の結果は外接矩形の外で常に 0）
    crop = region[box]
    return box, crop & ~clean_mask(crop, min_area).view(bool)


def remove_small_regions(semantic_map: np.ndarray, min_area: int, workers: Workers) -> None:
    """セマンティックマップ上の小領域ノイズを除去する（in-place）。

    1. クラスごとに小さな連結成分を背景へ戻す。
    2. 背景に残った小さな穴を、周囲で最も多いクラスで埋める。

    クラスは互いに素な画素なので、1 のクラスごとの整形は独立（スレッドがあれば並行して求め、反映は順に行う。結果は同じ）。
    """
    height, width = semantic_map.shape
    present_class_ids = [int(c) for c in np.flatnonzero(np.bincount(semantic_map.ravel())) if c != 0]
    jobs = [workers.submit(partial(_small_pieces, semantic_map, class_id, min_area)) for class_id in present_class_ids]
    for job in jobs:
        box, small = job.result()
        semantic_map[box][small] = 0

    n, comp, stats, _ = cv2.connectedComponentsWithStats(
        (semantic_map == 0).view(np.uint8), connectivity=8
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


def fill_narrow_gaps(semantic_map: np.ndarray, foreground: np.ndarray) -> None:
    """粗い前景の内部にある、幅が 1 パッチ未満の背景の隙間を前景にする（semantic_map を in-place で更新）。

    クラスごとに EdgeTAM を実行する（他クラスの前景は背景扱い）ので、1 つの物体が複数クラスに分かれ、その境界が画像の強い縁
    （ハイライト線など）に重なると、どのクラスのマスクにも入らない細い隙間が残る。隙間が幅 1 パッチ未満（粗い前景の分解能以下）で、
    しかも粗い前景を 1 パッチ縮めた内側（core）にあり、両側が 2 クラス以上に接していれば、背景ではないとみなして、最寄りのクラスで埋める。
    1 クラスの物体の凹みや穴（両側が同じクラス）は、本当の背景のことがあるので埋めない。
    事前マスクで他クラスの前景を「不明」にしても割れは消えるが、クラスのマスクが周囲へ広がってシャープでなくなるので、
    マスクは変えず、後から隙間だけ埋める。

    semantic_map: 作業解像度のセマンティックマップ（EdgeTAM の結果。埋める対象）。
    foreground: 粗い前景マスク（パッチ格子、全クラス）。隙間が「前景の内部」かどうかを決める基準にだけ使う。
    """
    grid_h, grid_w = foreground.shape
    core = erode(foreground)
    if not core.any():
        return
    # 1 パッチ = GAP_FILL_PIXELS_PER_PATCH 画素に縮小したマスクで、閉処理で埋まる画素（隙間）を求める
    small_size = (grid_w * GAP_FILL_PIXELS_PER_PATCH, grid_h * GAP_FILL_PIXELS_PER_PATCH)
    height, width = semantic_map.shape
    union = cv2.resize((semantic_map > 0).view(np.uint8) * 255, small_size, interpolation=cv2.INTER_AREA) >= 128
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_FILL_PIXELS_PER_PATCH + 1,) * 2)
    closed = cv2.morphologyEx(union.view(np.uint8), cv2.MORPH_CLOSE, kernel).view(bool)
    core_small = np.kron(core, np.ones((GAP_FILL_PIXELS_PER_PATCH,) * 2, dtype=bool))
    gap_small = cv2.dilate((closed & ~union).view(np.uint8), np.ones((3, 3), np.uint8)).view(bool) & core_small  # 1 画素広げて、縮小による取りこぼしを防ぐ
    if not gap_small.any():
        return
    gap = (cv2.resize(gap_small.view(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0) & (semantic_map == 0)
    if not gap.any():
        return
    # 隙間ごとに、外接矩形（最寄りのクラスが入るよう 1 パッチ分広げる）の中で、最寄りの前景画素のクラスを与える
    margin = int(np.ceil(max(height / grid_h, width / grid_w)))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin | 1, margin | 1))  # 半径 約 半パッチ
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gap.view(np.uint8), connectivity=8)
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        y0, y1, x0, x1 = max(y - margin, 0), min(y + h + margin, height), max(x - margin, 0), min(x + w + margin, width)
        crop = semantic_map[y0:y1, x0:x1]
        fill = labels[y0:y1, x0:x1] == i
        # 隙間の両側（半パッチ以内）が 2 クラス以上に接しているときだけ埋める（物体の凹みや、1 クラスの物体の中の穴は埋めない）
        ring = (cv2.dilate(fill.view(np.uint8), ring_kernel) > 0) & ~fill
        if len(np.unique(crop[ring & (crop > 0)])) < 2:
            continue
        _, (iy, ix) = distance_transform_edt(crop == 0, return_indices=True)
        crop[fill] = crop[iy, ix][fill]
