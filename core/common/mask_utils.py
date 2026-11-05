"""マスク・座標・点サンプリング・kNN グラフ上の伝播の、純粋なヘルパー関数（状態を持たない）。"""

from functools import partial
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt

from core.common.parallel import Workers
from core.common.params import (
    BACKGROUND_POINT_MARGIN,
    CLEAN_MASK_MAX_PER_COMPONENT,
    GAP_FILL_PIXELS_PER_PATCH,
    MEDIAN_BLUR_KSIZE,
    PATCH_NEIGHBORHOOD,
)


# =============================================================================
# 型と数値のヘルパー
# =============================================================================

class Component(NamedTuple):
    """マスクの連結成分1つ分（タプルとしても (peak, core, region) の順に取り出せる）。"""

    peak: Tuple[int, int]   # 最内点 (x, y)。距離変換が最大の位置
    core: np.ndarray        # 成分を1パッチ内側へ縮めたマスク（縮めると消える細い成分では成分そのもの）
    region: np.ndarray      # 成分のマスク


def _as_uint8(mask: np.ndarray) -> np.ndarray:
    """bool（または 0/非 0 の整数）のマスクを、コピーせずに（bool なら）uint8 の 0/1 として読む。cv2 の関数は bool を受け付けない。"""
    return np.ascontiguousarray(mask, dtype=bool).view(np.uint8)


def patch_kernel() -> np.ndarray:
    """パッチ近傍（PATCH_NEIGHBORHOOD 四方）の構造要素（uint8）。erode / dilate に渡す。"""
    return np.ones((PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD), dtype=np.uint8)


def erode(mask: np.ndarray) -> np.ndarray:
    """マスクをパッチ近傍で 1 回縮める（bool）。画像の外は前景として扱うので、画像の端では縮まない。"""
    return cv2.erode(_as_uint8(mask), patch_kernel()).view(bool)


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """マスクをパッチ近傍で膨らませる（bool）。画像の外は背景として扱う。"""
    return cv2.dilate(_as_uint8(mask), patch_kernel(), iterations=iterations).view(bool)


def squared_distance_to_background(mask: np.ndarray) -> np.ndarray:
    """前景の各画素から、最も近い背景（False）の画素までのユークリッド距離の二乗（int32。画像の外は背景ではない）。

    cv2 の距離は float32 で、等しい距離でも最後の桁がずれて同点の位置が変わる（scipy と最内点が変わる）ので、二乗して整数に丸める。
    """
    distance = cv2.distanceTransform(_as_uint8(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.rint(distance * distance).astype(np.int32)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    """特徴 (N, C) を行ごとに L2 正規化する。"""
    return features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# =============================================================================
# 点のサンプリング
# =============================================================================

def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """最遠点サンプリングで k 点を選ぶ。最初は重心に最も近い点で、以降は選択済みの点から最も遠い点を足していく。

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
    確信度が連続値だと、候補は最も低い count パッチそのものになる（点が隅に寄る）。広げても精度は上がらない（docs/method_history.md の「負例点をばらす」）。
    """
    background = ~foreground_binary
    _, labeled = cv2.connectedComponents(_as_uint8(background), connectivity=4)
    border_ids = np.unique(np.concatenate([labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]]))
    outer = np.isin(labeled, border_ids[border_ids > 0])
    if not outer.any():
        outer = background
    squared = squared_distance_to_background(outer)
    for margin in (BACKGROUND_POINT_MARGIN, 1):
        candidate = outer & (squared >= margin * margin)
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
# 連結成分とマスクの整形
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

    foreground: 粗い前景マスク（パッチ格子、全クラス）。
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


# =============================================================================
# パッチ格子 <-> 作業解像度
# =============================================================================

def patch_mean(patch_map: np.ndarray) -> np.ndarray:
    """パッチ近傍(PATCH_NEIGHBORHOOD四方)の平均。境界は端の画素を折り返す(reflect)。"""
    return cv2.blur(
        patch_map.astype(np.float32), (PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD),
        borderType=cv2.BORDER_REFLECT,
    )


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


# =============================================================================
# 前景確信度の伝播（kNN グラフ）
# =============================================================================

def knn_graph(features: np.ndarray, k: int) -> sparse.csr_matrix:
    """特徴 (N, C)（L2 正規化済み）のコサイン類似度が上位 k 個の近傍で、対称の kNN グラフを作る（行を正規化した疎行列）。"""
    n = len(features)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(similarity, -k, axis=1)[:, -k:]
    directed = sparse.csr_matrix((np.ones(n * k, dtype=np.float32), (np.repeat(np.arange(n), k), neighbors.ravel())), shape=(n, n))
    graph = directed.maximum(directed.T)
    return sparse.diags(1.0 / np.asarray(graph.sum(axis=1)).ravel()) @ graph


def propagate(values: np.ndarray, graph: sparse.csr_matrix, alpha: float, iterations: int) -> np.ndarray:
    """values (N,) をグラフ上で value ← (1-α)·values0 + α·(近傍の平均) と iterations 回伝播する。"""
    propagated = values.astype(np.float32)
    for _ in range(iterations):
        propagated = (1.0 - alpha) * values + alpha * (graph @ propagated)
    return propagated


# =============================================================================
# インスタンス分離の幾何（距離変換の watershed）
# =============================================================================

def peak_point(mask: np.ndarray) -> Tuple[int, int]:
    """マスクの最も内側の点 (x, y)（距離変換の最大）。外接矩形の中だけで計算する（全体と同じ結果で、速い）。"""
    ys, xs = np.nonzero(mask)
    y0, x0 = ys.min(), xs.min()
    crop = mask[y0:ys.max() + 1, x0:xs.max() + 1]
    dist = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x + x0), int(y + y0)


def h_maxima_markers(dist: np.ndarray, h: float) -> np.ndarray:
    """h-maxima（高さ h 未満の山を無視した極大）を、ラベル画像（int32）にする。"""
    kernel = np.ones((3, 3), np.uint8)
    rec = np.maximum(dist - h, 0).astype(np.float32)
    dist = dist.astype(np.float32)
    for _ in range(400):
        new = np.minimum(cv2.dilate(rec, kernel), dist)
        if np.array_equal(new, rec):
            break
        rec = new
    peaks = (rec == cv2.dilate(rec, kernel)) & (rec > 0) & (dist > 0)
    _, labels = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def watershed_from_markers(landscape: np.ndarray, markers: np.ndarray, region: np.ndarray) -> np.ndarray:
    """landscape（小さいほど先に冠水。0〜1）と markers（> 0 の ID）で region の中を分割する。region の外は背景として扱う。"""
    n = int(markers.max())
    image = np.dstack([(255 * np.clip(landscape, 0, 1)).astype(np.uint8)] * 3)
    seeds = markers.copy()
    seeds[~region] = n + 1          # 領域の外を 1 つの標識にして、領域の外へ広がらないようにする
    out = cv2.watershed(image, seeds)
    out[out == n + 1] = 0
    boundary = out < 0               # 境界（-1）は、最も近い物体に付ける
    if boundary.any() and (out > 0).any():
        _, (iy, ix) = distance_transform_edt(~(out > 0), return_indices=True)
        out[boundary] = out[iy, ix][boundary]
    out[~region] = 0
    out[out < 0] = 0
    return out


def upsample_labels(labels: np.ndarray, region: np.ndarray) -> np.ndarray:
    """小さい解像度のラベル画像を region の大きさへ拡大する。region の中でラベルが 0 の画素は、最も近いラベルで埋める。"""
    full = cv2.resize(labels.astype(np.int32), (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    full[~region] = 0
    missing = region & (full == 0)
    if missing.any() and (full > 0).any():
        _, (iy, ix) = distance_transform_edt(full == 0, return_indices=True)
        full[missing] = full[iy, ix][missing]
    return full


def split_by_distance(region: np.ndarray, h_frac: float, scale: float) -> np.ndarray:
    """距離変換の山（h-maxima。高さは連結成分の最大半径の h_frac 倍）をシードにした watershed。ラベル画像（0 = 領域外）を返す。"""
    height, width = region.shape
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    small = cv2.resize(region.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    dist = distance_transform_edt(small)
    n, comp = cv2.connectedComponents(small.astype(np.uint8), connectivity=8)
    markers = np.zeros(small.shape, np.int32)
    offset = 0
    for i in range(1, n):
        part = comp == i
        d = np.where(part, dist, 0)
        top = float(d.max())
        if top <= 0:
            continue
        found = h_maxima_markers(d, max(h_frac * top, 1.0))
        found[~part] = 0
        markers[found > 0] = found[found > 0] + offset
        offset = int(markers.max())
    if offset == 0:
        return np.zeros(region.shape, np.int32)
    landscape = 1.0 - dist / max(float(dist.max()), 1e-6)
    return upsample_labels(watershed_from_markers(landscape, markers, small), region)


def fragment_adjacency(labels: np.ndarray) -> dict:
    """隣り合う断片のペア (a, b)（a < b）と、境界の長さ（接する画素数）。"""
    pairs = {}
    kernel = np.ones((3, 3), np.uint8)
    for value in np.unique(labels):
        if not value:
            continue
        grown = cv2.dilate((labels == value).astype(np.uint8), kernel).astype(bool)
        touching = labels[grown & (labels != value)]
        for other, count in zip(*np.unique(touching[touching > 0], return_counts=True)):
            if other > value:
                key = (int(value), int(other))
                pairs[key] = max(pairs.get(key, 0), int(count))
    return pairs


def fragment_stats(labels: np.ndarray) -> dict:
    """断片ごとの (最内点 (x, y), 内接円の直径)。"""
    stats = {}
    for value in np.unique(labels):
        if not value:
            continue
        body = labels == value
        ys, xs = np.nonzero(body)
        y0, x0 = ys.min(), xs.min()
        crop = body[y0:ys.max() + 1, x0:xs.max() + 1]
        dist = distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
        py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
        stats[int(value)] = ((int(px + x0), int(py + y0)), float(2 * dist.max()))
    return stats
