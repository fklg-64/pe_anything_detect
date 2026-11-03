"""前景のパッチのラベル（クラス）の付け直し。粗い判定のクラスタ（画像全体の K 個）の付けたラベルを、前景だけのブロックの根拠で整える。

粗い判定は、画像全体を K 個にクラスタリングして、クラスタごとにプロンプトを選ぶ。クラスタは背景を含み、代表トークンは画像全体の CLS に引っ張られるので、
(a) 1 つの物体が、似たプロンプト（tube と transparent tubes など）の別クラスに割れる、(b) 種類の違う物体が同じクラスになる、
(c) ふたとカップのように、見た目も言葉も近い部品が隣り合うと、境目のクラスタが両方のパッチを含み、パッチ単位でクラスが入り混じる、が起こる。
ここでは前景のパッチだけを部品ごとのブロックに分け直し（CLS を含まない extract_region_tokens で前景プロンプトと照合する）、3 段で整える。

  1. rejudge_regions       （物体, クラス）の領域ごとに、PE クラスタの根拠とブロックの根拠を集計して、名前を判定し直す（種類の違う物体を分ける）
  2. merge_weak_classes    画像全体で、ブロックの根拠が弱い小さいクラスを、大きいクラスへまるごと統合する（割れを戻す。ふたとカップのような違う部品は残る）
  3. relabel_foreground    同じ物体の中で、現状のラベルとブロックの照合を等しい重みで混ぜ、EdgeTAM の特徴が似たパッチどうしで伝播して、パッチごとに選ぶ（境目を整える）

根拠はすべて CLIP のロジット（確率の対数。CLIP_LOGIT_SCALE 込み）の単位で足し引きするので、しきい値は無い。領域が 1 つしか無い画像は何も変えない。
"""

from typing import Tuple

import cv2
import numpy as np
from scipy import sparse

from core.common.mask_utils import propagate, softmax


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


def rejudge_regions(prompt_index: np.ndarray, components: np.ndarray, evidence: np.ndarray, blob_count: int) -> np.ndarray:
    """（物体, クラス）の領域ごとに、領域のパッチで平均した根拠が最も強い前景プロンプトへ付け直す（新しい名前も許す）。

    evidence: パッチごとの前景プロンプトの根拠 (N, 前景プロンプト数)（ロジット。大きいほどそのプロンプトらしい）。
    前景を blob_count 個のブロックに分けたときの 1 ブロックの平均面積より小さい領域は、独立した根拠を持てない（根拠のノイズに引きずられる）ので、今のラベルのまま。
    """
    foreground, keys, inverse, _ = regions_of(prompt_index, components)
    area = np.bincount(inverse).astype(np.float64)
    total = np.zeros((len(keys), evidence.shape[1]))
    np.add.at(total, inverse, evidence[foreground])
    judged = np.where(area >= len(foreground) / blob_count, np.argmax(total / area[:, None], axis=1), keys % (int(prompt_index.max()) + 1))
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
