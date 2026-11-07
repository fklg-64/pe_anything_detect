"""④ 前景のパッチのラベル（クラス）の付け直し。粗い判定のクラスタ（画像全体の K 個）の付けたラベルを、前景だけのブロックの根拠で整える。

粗い判定（clusters.py）は、画像全体を K 個にクラスタリングして、クラスタごとにプロンプトを選ぶ。クラスタは背景を含み、代表トークンは画像全体の CLS に引っ張られるので、
(a) 1 つの物体が、似たプロンプト（tube と transparent tubes など）の別クラスに割れる、(b) 種類の違う物体が同じクラスになる、
(c) ふたとカップのように、見た目も言葉も近い部品が隣り合うと、境目のクラスタが両方のパッチを含み、パッチ単位でクラスが入り混じる、が起こる。
ここでは前景のパッチだけを部品ごとのブロックに分け直し（CLS を含まない extract_region_tokens で前景プロンプトと照合する）、3 段で整える。

  1. rejudge_regions       （物体, クラス）の領域ごとに、PE クラスタの根拠とブロックの根拠を集計して、名前を判定し直す（種類の違う物体を分ける）
  2. merge_weak_classes    画像全体で、ブロックの根拠が弱い小さいクラスを、大きいクラスへまるごと統合する（割れを戻す。ふたとカップのような違う部品は残る）
  3. relabel_foreground    同じ物体の中で、現状のラベルとブロックの照合を等しい重みで混ぜ、EdgeTAM の特徴が似たパッチどうしで伝播して、パッチごとに選ぶ（境目を整える）

根拠はすべて CLIP のロジット（確率の対数。CLIP_LOGIT_SCALE 込み）の単位で足し引きするので、しきい値は無い。領域が 1 つしか無い画像は何も変えない。
全体の進行は ForegroundRelabeler.relabel。

merge_weak_classes と regions_of は、精緻化の後の最終の名前付け（detail/naming.py）も使う。
"""

from typing import Tuple

import cv2
import numpy as np
from scipy import sparse

from core.common.backend import CLIPBackend
from core.common.coarse.clusters import kmeans_labels
from core.common.coarse.propagation import edge_features_of, propagate
from core.common.params import EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS, RELABEL_BLOCK_COUNT
from core.common.prompts import PromptBank
from core.utils import softmax


# =============================================================================
# 全体の進行
# =============================================================================

class ForegroundRelabeler:
    """前景のパッチのプロンプト番号を付け直す。"""

    def __init__(self, extractor: CLIPBackend, prompts: PromptBank, grid: Tuple[int, int], k_clusters: int, niter: int):
        self.extractor = extractor
        self.prompts = prompts
        self.grid = grid                # パッチ格子 (高さ, 幅)
        self.k_clusters = k_clusters    # 粗い判定のクラスタ数。「小さすぎる領域」の基準（前景の 1/k_clusters）にも使う
        self.niter = niter              # KMeans の反復回数

    def relabel(
        self,
        prompt_index: np.ndarray,
        foreground: np.ndarray,
        patch_features: np.ndarray,
        patch_tokens: np.ndarray,
        edge_embed: np.ndarray,
        graph: sparse.csr_matrix,
        pe_labels: np.ndarray,
        cluster_logits: np.ndarray,
    ) -> np.ndarray:
        """前景のパッチのプロンプト番号を付け直す。領域（物体 × クラス）が 1 つしか無ければ、何もしない。

        前景のパッチだけを、PE（fused）と EdgeTAM の特徴を並べた空間で KMeans して部品ごとのブロックにする（ブロック数は RELABEL_BLOCK_COUNT = 32。画像全体の k_clusters = 8 より細かい）。
        ブロックごとの代表トークンは extract_region_tokens（CLS を含まない value の平均）で求めるので、数パッチの小さなブロックでも画像全体の CLS に
        引っ張られず、それらしい前景プロンプトに紐づく。その根拠で、次の 3 段で付け直す。
          1. 領域ごとに、PE のクラスタのロジット（今のラベルの根拠）とブロックのロジットを集計して、名前を判定し直す
          2. 画像全体で、ブロックの根拠が弱い小さいクラスを、大きいクラスへ統合する
          3. 同じ物体の中で、今のラベルとブロックの照合を混ぜ、EdgeTAM の特徴が似たパッチどうしで伝播して、パッチごとに選ぶ

        prompt_index: パッチごとのプロンプト番号 (N,)。foreground: 前景のパッチ (N,) bool。patch_tokens: PE の fused パッチ特徴 (N, C)。
        pe_labels / cluster_logits: ① のクラスタ番号 (N,) と、クラスタごとの前景プロンプトのロジット（clusters.judge_clusters）。graph: EdgeTAM の kNN グラフ。
        """
        components = object_components(foreground, self.grid)
        if not needs_relabel(prompt_index, components):
            return prompt_index
        blobs, blob_logits = self.foreground_blocks(foreground, patch_features, patch_tokens, edge_embed)

        # 小さすぎる領域（前景の 1/k_clusters 未満）は判定し直さない。この基準はブロック数（RELABEL_BLOCK_COUNT）と別
        prompt_index = rejudge_regions(prompt_index, components, cluster_logits[pe_labels] + blob_logits[blobs], self.k_clusters)
        prompt_index = merge_weak_classes(prompt_index, foreground.astype(np.int64), blobs, blob_logits)
        if not has_split_object(prompt_index, components):
            return prompt_index
        return relabel_foreground(
            prompt_index, components, blobs, blob_logits, graph, EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS
        )

    def foreground_blocks(
        self, foreground: np.ndarray, patch_features: np.ndarray, patch_tokens: np.ndarray, edge_embed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前景のパッチだけを、部品ごとのブロックに分け、(パッチごとのブロック番号 (N,)（前景でないパッチは 0）, ブロックごとの前景プロンプトのロジット) を返す。

        ブロック番号は、前景のパッチで 0~、ロジットの行は同じ番号（blob_logits[blobs] でパッチごとの根拠になる）。
        """
        # PE の fused 特徴はモデルの中で L2 正規化済み。EdgeTAM の特徴と並べても、2 つの特徴が同じ重みになる
        features = np.hstack([patch_tokens[foreground], edge_features_of(edge_embed)[foreground]])
        blobs = np.zeros(len(foreground), dtype=np.int64)
        blobs[foreground] = kmeans_labels(features, k=min(RELABEL_BLOCK_COUNT, len(features)), niter=self.niter)
        # 番号 0 は前景でないパッチ。ブロックは 1~
        tokens = self.extractor.extract_region_tokens(patch_features, np.where(foreground, blobs + 1, 0))[1:]
        return blobs, self.prompts.foreground_logits(tokens)


# =============================================================================
# 領域（物体 × クラス）
# =============================================================================

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


# =============================================================================
# 1. 名前の判定し直し
# =============================================================================

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


# =============================================================================
# 2. 根拠の弱い小さいクラスの統合
# =============================================================================

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


# =============================================================================
# 3. 同じ物体の中での、パッチ単位の付け直し
# =============================================================================

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
