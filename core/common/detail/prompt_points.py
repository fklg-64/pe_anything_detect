"""EdgeTAM に与えるプロンプト点の選び方（粗いマスクはパッチ単位なので、点は粗いマスクから決める）。

  負例点（背景）    select_background_points   画像の端につながる背景のうち、前景から十分離れた位置。全領域に共通なので画像に 1 回求める
  正例点（前景）    object_prompt_points       EdgeTAM の高解像度特徴量の画素から、物体の上に選ぶ（下の説明）
                    prompt_points              物体らしさが作れないときの代わり。パッチの中心から選ぶ

正例点の選び方（物体らしさ）:
粗いマスクはパッチ単位（1 パッチ = 16 画素）で、縁のパッチは物体と背景の両方を含む。パッチの中心に点を置くと、縁や細い物体では
点が背景に落ちたり、物体の一部（歯ブラシの毛だけ・ヘッドの縁や柄が欠ける）に偏ったりする。そこで、点をパッチの中心ではなく、
EdgeTAM の高解像度特徴量（入力の 1/4）の画素から選ぶ。

  物体らしさ = 物体の特徴（クラスの core の平均）とのコサイン類似度 − 背景の特徴（粗い前景から離れた領域の平均）とのコサイン類似度
               （EdgeTAM の特徴量 3 層で求めて、最も細かい層の大きさで平均する）
  候補       = 粗いマスク全体（縁のパッチも含む）のうち、物体らしさが正の画素
  選び方     = 各連結成分の最内点（そのパッチの中で物体らしさが最大の画素）を必ず含め、残りは「選択済みの点からの距離 × 物体らしさ」が最大の画素を順に足す
               候補が 1 つも無い領域（粗い判定の誤りで、背景のパッチ）には、点を置かない

距離だけで選ぶと（物体らしさで重みづけしないと）縁の背景寄りの画素に点が寄り、合成データで下がる。パラメータは増やさない。
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.common.mask_utils import as_uint8, dilate, squared_distance_to_background
from core.common.params import BACKGROUND_POINT_MARGIN


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


# =============================================================================
# 負例点（背景）
# =============================================================================

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
    _, labeled = cv2.connectedComponents(as_uint8(background), connectivity=4)
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
# 正例点（前景）: EdgeTAM の高解像度特徴量の画素から、物体の上に選ぶ
# =============================================================================

def normalize_features(feats: Sequence[np.ndarray]) -> List[np.ndarray]:
    """EdgeTAM の特徴量（high_res_feats0 / high_res_feats1 / image_embed）を、画素ごとに L2 正規化した (C, H, W) の配列にする。"""
    levels = []
    for feat in feats:
        # 連続した (C, H, W) にそろえる（OpenVINO の特徴量は、NHWC で持つものを NCHW に見せたビューのことがある。以降の和の順序を変えないため）
        array = np.ascontiguousarray(feat, dtype=np.float32)
        array = array[0] if array.ndim == 4 else array
        levels.append(array / (np.linalg.norm(array, axis=0, keepdims=True) + 1e-8))
    return levels


def mean_feature(level: np.ndarray, patch_mask: np.ndarray) -> Optional[np.ndarray]:
    """パッチ格子のマスクの内側にある特徴の平均（L2 正規化）。マスクが空なら None。level は (C, H, W)。"""
    _, height, width = level.shape
    mask = cv2.resize(patch_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    index = np.flatnonzero(mask)
    if not index.size:
        return None
    # level[:, mask].mean(axis=1) と同じ値（その和は、画素を順に足す）を、チャンネルごとの集めなおしで速く求める。np.take(...).mean() は足す順序が違い、値が少しずれる
    picked = np.take(level.reshape(level.shape[0], -1), index, axis=1)
    mean = np.add.accumulate(picked, axis=1)[:, -1] / index.size
    return mean / (np.linalg.norm(mean) + 1e-8)


def background_features(levels: Sequence[np.ndarray], foreground: np.ndarray) -> Optional[List[np.ndarray]]:
    """背景の特徴（層ごとの平均）。背景 = 粗い前景から BACKGROUND_POINT_MARGIN パッチ以上離れた領域（無ければ前景以外）。作れなければ None。

    foreground: 粗い前景マスク（パッチ格子、全クラス）。画像に 1 回求めれば、全クラスで共通に使える。
    """
    background = ~dilate(foreground, iterations=BACKGROUND_POINT_MARGIN)
    if not background.any():
        background = ~foreground
    means = [mean_feature(level, background) for level in levels]
    return None if any(m is None for m in means) else means


def objectness_map(levels: Sequence[np.ndarray], background: Sequence[np.ndarray], core: np.ndarray) -> Optional[np.ndarray]:
    """物体らしさのマップ（最も細かい層の大きさ）。正なら、背景の特徴より core の特徴に近い。core が空なら None。

    levels: normalize_features の結果。background: background_features の結果。core: 物体の特徴を代表するパッチ（パッチ格子）。
    領域自身の core だと、領域が背景のパッチのとき基準も背景になり、ほぼ全画素が正になる。極小の成分は、クラスの本体の core を渡す。
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
    領域に物体らしい画素（物体らしさが正）が 1 つも無いときは、点を置かず空の配列を返す（粗い判定の誤り: 背景のパッチ）。
    """
    height, width = score.shape
    grid_h, grid_w = grid

    candidate = (cv2.resize(region.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0) & (score > 0)
    if not candidate.any():
        return np.empty((0, 2), dtype=np.float64)

    # 最内点は、そのパッチの中で物体らしさが最大の画素へ動かす
    selected = []
    for x, y in peaks:
        u0, u1 = int(np.floor(x * width / grid_w)), min(width, int(np.ceil((x + 1) * width / grid_w)))
        v0, v1 = int(np.floor(y * height / grid_h)), min(height, int(np.ceil((y + 1) * height / grid_h)))
        j, i = np.unravel_index(np.argmax(score[v0:v1, u0:u1]), (v1 - v0, u1 - u0))
        selected.append((u0 + i, v0 + j))

    if len(selected) < k:
        pixels = np.argwhere(candidate)[:, ::-1].astype(np.float64)  # (u, v)
        weight = score[candidate]
        d2 = np.min(((pixels[:, None, :] - np.asarray(selected, dtype=np.float64)[None]) ** 2).sum(-1), axis=1)
        while len(selected) < k and d2.max() > 0:
            idx = int(np.argmax(np.sqrt(d2) * weight))
            selected.append(tuple(pixels[idx]))
            d2 = np.minimum(d2, ((pixels - pixels[idx]) ** 2).sum(-1))

    # 画素の中心 -> パッチ座標
    return np.array([((u + 0.5) * grid_w / width - 0.5, (v + 0.5) * grid_h / height - 0.5) for u, v in selected], dtype=np.float64).reshape(-1, 2)


# =============================================================================
# 正例点（前景）: パッチの中心から選ぶ（物体らしさが作れないときの代わり）
# =============================================================================

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
