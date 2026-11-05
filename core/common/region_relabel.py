"""前景のパッチのラベル（クラス）の付け直し。粗い判定のクラスタ（画像全体の K 個）の付けたラベルを、前景だけのブロックの根拠で整える。

粗い判定は、画像全体を K 個にクラスタリングして、クラスタごとにプロンプトを選ぶ。クラスタは背景を含み、代表トークンは画像全体の CLS に引っ張られるので、
(a) 1 つの物体が、似たプロンプト（tube と transparent tubes など）の別クラスに割れる、(b) 種類の違う物体が同じクラスになる、
(c) ふたとカップのように、見た目も言葉も近い部品が隣り合うと、境目のクラスタが両方のパッチを含み、パッチ単位でクラスが入り混じる、が起こる。
ここでは前景のパッチだけを部品ごとのブロックに分け直し（CLS を含まない extract_region_tokens で前景プロンプトと照合する）、3 段で整える。

  1. rejudge_regions       （物体, クラス）の領域ごとに、PE クラスタの根拠とブロックの根拠を集計して、名前を判定し直す（種類の違う物体を分ける）
  2. merge_weak_classes    画像全体で、ブロックの根拠が弱い小さいクラスを、大きいクラスへまるごと統合する（割れを戻す。ふたとカップのような違う部品は残る）
  3. relabel_foreground    同じ物体の中で、現状のラベルとブロックの照合を等しい重みで混ぜ、EdgeTAM の特徴が似たパッチどうしで伝播して、パッチごとに選ぶ（境目を整える）

根拠はすべて CLIP のロジット（確率の対数。CLIP_LOGIT_SCALE 込み）の単位で足し引きするので、しきい値は無い。領域が 1 つしか無い画像は何も変えない。

精緻化の後には、最終マップ（EdgeTAM が切り出した画素単位の物体）の領域ごとに、名前をもう一度決め直す（name_final_regions）。
粗い段階のブロックは物体の縁で背景と混ざり、小さな物体は 1 ブロックより小さいので判定し直せない。最終マップの物体は EdgeTAM が物体と認めた
画素なので、その画素だけから作った代表トークンは背景を含まず、物体ごとに名前を付けられる（種類の違う物体を分ける）。前景のマスクは変えない。
"""

from typing import Callable, List, NamedTuple, Tuple

import cv2
import numpy as np
from scipy import sparse

from core.common.mask_utils import propagate, softmax
from core.common.params import GAP_FILL_PIXELS_PER_PATCH


def object_components(foreground: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """前景 (N,) の 8 連結成分（物体）の番号 (N,)。背景は 0、物体は 1~。"""
    _, components = cv2.connectedComponents(foreground.reshape(grid).astype(np.uint8), connectivity=8)
    return components.reshape(-1)


def regions_of(prompt_index: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """前景のパッチ（groups > 0）を、（グループ, クラス）が同じものどうしの領域に分ける。
    (前景のパッチの位置, 領域の鍵 = グループ * stride + クラス, パッチごとの領域番号, stride) を返す。"""
    foreground = np.flatnonzero(groups > 0)
    stride = int(prompt_index.max()) + 1
    keys, inverse = np.unique(groups[foreground] * stride + prompt_index[foreground], return_inverse=True)
    return foreground, keys, inverse.reshape(-1), stride


def needs_relabel(prompt_index: np.ndarray, components: np.ndarray) -> bool:
    """領域（物体 × クラス）が 1 つしか無ければ、比べる相手が無いので何も変わらない。"""
    return len(regions_of(prompt_index, components)[1]) >= 2


def has_split_object(prompt_index: np.ndarray, components: np.ndarray) -> bool:
    """物体のどれかに、2 つ以上のクラスがあるか。無ければ、パッチ単位の付け直し（relabel_foreground）をしても変わらない。"""
    keys = regions_of(prompt_index, components)[1]
    return len(keys) > len(np.unique(keys // (int(prompt_index.max()) + 1)))


def rejudge_regions(prompt_index: np.ndarray, components: np.ndarray, evidence: np.ndarray, min_region_divisor: int) -> np.ndarray:
    """（物体, クラス）の領域ごとに、領域のパッチで平均した根拠が最も強い前景プロンプトへ付け直す（新しい名前も許す）。

    evidence: パッチごとの前景プロンプトの根拠 (N, 前景プロンプト数)（ロジット。大きいほどそのプロンプトらしい）。
    前景のパッチ数を min_region_divisor で割った面積（本体は k_clusters = 8 なので前景の 1/8。ブロック数 RELABEL_BLOCK_COUNT とは別）より小さい領域は、独立した根拠を持てない（根拠のノイズに引きずられる）ので、今のラベルのまま。
    """
    foreground, keys, inverse, _ = regions_of(prompt_index, components)
    area = np.bincount(inverse).astype(np.float64)
    total = np.zeros((len(keys), evidence.shape[1]))
    np.add.at(total, inverse, evidence[foreground])
    judged = np.where(area >= len(foreground) / min_region_divisor, np.argmax(total / area[:, None], axis=1), keys % (int(prompt_index.max()) + 1))
    result = prompt_index.copy()
    result[foreground] = judged[inverse]
    return result


def merge_weak_classes(prompt_index: np.ndarray, groups: np.ndarray, blobs: np.ndarray, blob_logits: np.ndarray) -> np.ndarray:
    """グループ（物体や画像全体）の中の小さいクラスを、根拠が弱ければ大きいクラスへまるごと統合する。ロジットが僅差のクラスを割れたままにしない。

    面積が小さいクラス b から順に、領域のブロックのロジットを平均した対数オッズ  mean_i[L_i(c) - L_i(b)] + log(w_c / w_b)  が正で最大の、
    より大きいクラス c へ移す。w はグループの中の今のクラスの面積（事前分布）。ロジットは CLIP の確率の単位（対数）なので、差の許容はこの式に含まれる。
    違う部品は、領域のブロックのロジットが自分のクラスを大きく上回るので（ふたとカップ）、面積の差を超えて残る。
    """
    foreground, keys, inverse, stride = regions_of(prompt_index, groups)
    area = np.bincount(inverse).astype(np.float64)
    total = np.zeros((len(keys), blob_logits.shape[1]))
    np.add.at(total, inverse, blob_logits[blobs[foreground]])

    target = keys % stride
    classes = {}  # グループ -> {クラス: [面積, ロジットの合計]}
    for key, count, logits in zip(keys, area, total):
        classes.setdefault(int(key // stride), {})[int(key % stride)] = [count, logits]
    moved = {}    # (グループ, クラス) -> 移し先のクラス
    for group, members in classes.items():
        merged = True
        while merged and len(members) > 1:
            merged = False
            for small in sorted(members, key=lambda cls: members[cls][0])[:-1]:
                small_area, small_logits = members[small]
                best, best_score = None, 0.0
                for big, (big_area, _) in members.items():
                    if big_area > small_area:
                        score = (small_logits[big] - small_logits[small]) / small_area + np.log(big_area / small_area)
                        if score > best_score:
                            best, best_score = big, score
                if best is not None:
                    members[best][0] += small_area
                    members[best][1] = members[best][1] + small_logits
                    del members[small]
                    moved[(group, small)] = best
                    merged = True
                    break
    for index, key in enumerate(keys):
        group, cls = int(key // stride), int(key % stride)
        while (group, cls) in moved:
            cls = moved[(group, cls)]
        target[index] = cls
    result = prompt_index.copy()
    result[foreground] = target[inverse]
    return result


def same_object_graph(graph: sparse.csr_matrix, components: np.ndarray) -> sparse.csr_matrix:
    """kNN グラフ（N, N）の辺のうち、同じ物体の前景のパッチどうしだけを残して、行を正規化する（近傍が無い前景のパッチは自分自身を平均する）。"""
    coo = graph.tocoo()
    keep = (components[coo.row] == components[coo.col]) & (components[coo.row] > 0)
    kept = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape)
    degree = np.asarray(kept.sum(axis=1)).ravel()
    isolated = np.flatnonzero((degree == 0) & (components > 0))
    kept = sparse.diags(1.0 / np.maximum(degree, 1e-9)) @ kept
    return (kept + sparse.csr_matrix((np.ones(len(isolated)), (isolated, isolated)), shape=graph.shape)).tocsr()


def relabel_foreground(
    prompt_index: np.ndarray,
    components: np.ndarray,
    blob_labels: np.ndarray,
    blob_logits: np.ndarray,
    graph: sparse.csr_matrix,
    alpha: float,
    iterations: int,
) -> np.ndarray:
    """前景のパッチのプロンプト番号を付け直した配列を返す（背景のパッチと prompt_index は変えない）。

    prompt_index: パッチごとのプロンプト番号 (N,)。前景のパッチは前景プロンプト（< 前景プロンプト数）。
    components: object_components の結果 (N,)。blob_labels: 前景のパッチのブロック番号 (N,)（背景は無視）。
    blob_logits: ブロックごとの前景プロンプトのロジット (ブロック数, 前景プロンプト数)。graph: EdgeTAM の特徴の kNN グラフ。
    """
    foreground = np.flatnonzero(components > 0)
    present = np.unique(prompt_index[foreground])               # 現状に出てくるクラス（列）
    column = np.zeros(int(prompt_index.max()) + 1, dtype=np.int64)
    column[present] = np.arange(len(present))
    old_column = column[prompt_index[foreground]]

    # 1. 現状のラベル
    belief_old = np.zeros((len(foreground), len(present)), dtype=np.float32)
    belief_old[np.arange(len(foreground)), old_column] = 1.0

    # 2. ブロックの代表トークンの照合。候補は、現状のラベルが同じ物体の中にあるクラスだけ
    candidates = np.zeros((int(components.max()) + 1, len(present)), dtype=bool)
    candidates[components[foreground], old_column] = True
    logits = blob_logits[blob_labels[foreground]][:, present]
    logits = np.where(candidates[components[foreground]], logits, -np.inf)
    belief_new = softmax(logits, axis=1).astype(np.float32)

    # 同じ物体の中で、EdgeTAM の特徴が似たパッチどうしに伝播して、最も強いクラスを選ぶ
    belief = np.zeros((len(prompt_index), len(present)), dtype=np.float32)
    belief[foreground] = 0.5 * (belief_old + belief_new)
    belief = propagate(belief, same_object_graph(graph, components), alpha, iterations)

    result = prompt_index.copy()
    result[foreground] = present[np.argmax(belief[foreground], axis=1)]
    return result


# =============================================================================
# 精緻化の後: 最終マップの領域ごとの名前付け
# =============================================================================

class FinalRegions(NamedTuple):
    """最終マップの（物体, クラス）の領域。物体は前景の 8 連結成分（画素単位）。"""

    objects: np.ndarray       # (H, W) int32。画素ごとの物体番号（0 = 背景）
    keys: np.ndarray          # (R+1,) 領域の鍵 = 物体番号 * stride + クラス番号（最終マップの 1~）。0 番は背景
    stride: int               # 鍵のクラス番号の桁（= 最大のクラス番号 + 1）
    sample_area: np.ndarray   # (R+1,) 領域の標本の数（面積に比例する）
    patch_labels: np.ndarray  # (N,) パッチごとの領域番号。パッチの面積の半分以上を占める領域（無ければ 0）。
                              # 半分以上を占めるパッチが無い小さな領域には、その領域が最も多くを占める、まだどの領域にも属さないパッチを割り当てる


def final_regions(semantic_map: np.ndarray, grid: Tuple[int, int]) -> FinalRegions:
    """最終マップ (H, W)（0 = 背景、1~ = クラス）を、（物体, クラス）の領域に分け、パッチ格子 grid = (高さ, 幅) の各パッチを領域に割り当てる。

    領域がパッチに占める割合は、1 パッチを GAP_FILL_PIXELS_PER_PATCH 四方の点で標本にして数える（全画素で数えると 1024 四方で約 8ms）。
    標本に 1 点も入らない（パッチの 1/64 未満の）領域は、ここでは領域にならない（name_final_regions が同じクラスの最大の領域に合わせる）。
    """
    n_objects, objects = cv2.connectedComponents((semantic_map > 0).astype(np.uint8), connectivity=8)
    stride = int(semantic_map.max()) + 1
    grid_h, grid_w = grid
    height, width = semantic_map.shape
    step = GAP_FILL_PIXELS_PER_PATCH
    rows = np.minimum(((np.arange(grid_h * step) + 0.5) * height / (grid_h * step)).astype(np.int64), height - 1)
    cols = np.minimum(((np.arange(grid_w * step) + 0.5) * width / (grid_w * step)).astype(np.int64), width - 1)
    key = objects[np.ix_(rows, cols)].astype(np.int64) * stride + semantic_map[np.ix_(rows, cols)]
    sample_area = np.bincount(key.ravel(), minlength=n_objects * stride)
    keys = np.flatnonzero(sample_area)
    if keys[0] != 0:  # 背景の標本が無い画像でも、0 番を背景にする
        keys = np.concatenate([[0], keys])
    lookup = np.zeros(n_objects * stride, dtype=np.int64)
    lookup[keys] = np.arange(len(keys))
    region = lookup[key]

    # 各パッチを、面積の半分以上を占める領域に割り当てる
    num_patches = grid_h * grid_w
    patch = (np.arange(grid_h * step) // step)[:, None] * grid_w + (np.arange(grid_w * step) // step)[None, :]
    counts = np.bincount((region * num_patches + patch).ravel(), minlength=len(keys) * num_patches)
    fraction = counts.reshape(len(keys), num_patches) / float(step * step)
    owner = np.argmax(fraction[1:], axis=0) + 1
    labels = np.where(fraction[1:].max(axis=0) >= 0.5, owner, 0)
    for r in np.setdiff1d(np.arange(1, len(keys)), labels):
        p = int(np.argmax(fraction[r]))
        if labels[p] == 0 or fraction[labels[p], p] < fraction[r, p]:  # 半分以上を占める領域のパッチは取らない（割合が 0.5 未満なので）
            labels[p] = r
    return FinalRegions(objects, keys, stride, sample_area[keys], labels.astype(np.int64))


def name_final_regions(
    semantic_map: np.ndarray,
    prompt_indices: List[int],
    region_logits: Callable[[np.ndarray], np.ndarray],
    grid: Tuple[int, int],
    min_region_divisor: int,
) -> Tuple[np.ndarray, List[int]]:
    """最終マップの（物体, クラス）の領域ごとに名前（前景プロンプト）を決め直し、(新しい最終マップ, クラス 1~ のプロンプト番号) を返す。前景のマスクは変えない。

    region_logits: パッチごとの領域番号 (N,) -> 番号 0.. ごとの前景プロンプトのロジット (最大番号+1, 前景プロンプト数)。
    1. 判定し直し: 領域のパッチの代表トークンの照合が最も強いプロンプトにする。前景のパッチ数 / min_region_divisor（粗い段階の rejudge_regions と同じ
       k_clusters）より小さい領域は、独立した根拠を持てない（1〜2 パッチのトークンの照合はノイズが大きい）ので、今の名前のまま。
    2. 統合: 画像全体で、根拠が弱い小さいクラスを大きいクラスへまとめる（merge_weak_classes。対数オッズ + log 面積比）。
    名前ごとに 1 クラスにする。クラスの順は、今のクラスの順（残った名前）→ 新しい名前（出てきた順）。何も変わらなければ、入力をそのまま返す。
    """
    if not len(prompt_indices) or semantic_map.max() == 0:
        return semantic_map, prompt_indices
    regions = final_regions(semantic_map, grid)
    labels, stride = regions.patch_labels, regions.stride
    region_class = regions.keys % stride
    num_regions = len(regions.keys)
    foreground = labels > 0
    if not foreground.any():
        return semantic_map, prompt_indices
    logits = region_logits(labels)
    if len(logits) < num_regions:  # パッチを持たない（最大番号より大きい）領域の行
        logits = np.vstack([logits, np.zeros((num_regions - len(logits), logits.shape[1]), logits.dtype)])

    # 1. 判定し直し（小さな領域は今の名前のまま）
    prompts = np.concatenate([[-1], np.asarray(prompt_indices, dtype=np.int64)])  # クラス番号 -> プロンプト番号
    current = prompts[region_class]
    area = np.bincount(labels, minlength=num_regions)
    rejudge = area >= foreground.sum() / min_region_divisor
    rejudge[0] = False
    name = np.where(rejudge, np.argmax(logits, axis=1), current)

    # 2. 統合（パッチ単位の配列にして渡す。グループ = 画像全体、ブロック = 領域）
    merged = merge_weak_classes(np.where(foreground, name[labels], 0), foreground.astype(np.int64), labels, logits)
    moved = {int(name[r]): int(merged[np.argmax(labels == r)]) for r in np.unique(labels[foreground])}
    name = np.array([moved.get(int(n), int(n)) for n in name], dtype=np.int64)

    # 標本に入らなかった小さな領域は、同じクラスの最大の領域の名前にする（クラスに標本が無ければ、今の名前のまま）
    class_name = prompts.copy()
    largest = np.zeros(stride, dtype=np.int64)
    for r in range(1, num_regions):
        c = region_class[r]
        if regions.sample_area[r] > largest[c]:
            largest[c], class_name[c] = regions.sample_area[r], name[r]
    histogram = cv2.calcHist([semantic_map.astype(np.uint8)], [0], None, [stride], [0, stride]).ravel()  # 全画素の bincount より速い
    present_classes = np.flatnonzero(histogram[1:]) + 1
    names = [int(n) for n in name[1:]] + [int(class_name[c]) for c in present_classes]
    order = [p for p in dict.fromkeys(prompt_indices) if p in names] + [p for p in dict.fromkeys(names) if p not in prompt_indices]
    number = {p: k + 1 for k, p in enumerate(order)}
    class_lookup = np.zeros(stride, dtype=semantic_map.dtype)
    class_lookup[present_classes] = [number[int(class_name[c])] for c in present_classes]
    region_number = np.array([0] + [number[int(n)] for n in name[1:]])
    if order == list(prompt_indices) and np.array_equal(class_lookup[present_classes], present_classes):
        if np.array_equal(region_number[1:], class_lookup[region_class[1:]]):
            return semantic_map, prompt_indices  # 何も変わらない
    if np.array_equal(region_number[1:], class_lookup[region_class[1:]]):
        return class_lookup[semantic_map], order  # 同じクラスの物体はすべて同じ名前: クラスの表で書き換える
    key_lookup = class_lookup[np.arange(int(regions.objects.max() + 1) * stride) % stride]
    key_lookup[regions.keys[1:]] = region_number[1:]
    return key_lookup[regions.objects * stride + semantic_map], order  # 同じクラスの物体に違う名前: 物体ごとに書き換える
