"""③ 前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える。

PE のクラスタは意味は近いが空間的にばらけ、背景の斑点・織り目の一部が前景のクラスタに混ざる（誤検出）。
EdgeTAM の特徴は物体の輪郭・領域に沿ってまとまるので、その類似度の kNN グラフで確信度を平均して、物体の構造に沿って揃える。
前景を消す（AND のように）のではなく、確信度をならすだけなので、大きな物体は崩れにくい。
kNN グラフは、④ の付け直しでも使う。
"""

from typing import Tuple

import numpy as np
from scipy import sparse

from core.common.params import EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS, EDGE_PROPAGATION_NEIGHBORS
from core.utils import normalize_rows


def edge_features_of(edge_embed: np.ndarray) -> np.ndarray:
    """EdgeTAM の image_embed (1, C, h, w) -> パッチごとの L2 正規化した特徴 (h*w, C)。"""
    return normalize_rows(edge_embed[0].reshape(edge_embed.shape[1], -1).T)


def knn_graph(features: np.ndarray, k: int) -> sparse.csr_matrix:
    """特徴 (N, C)（L2 正規化済み）のコサイン類似度が上位 k 個の近傍で、対称の kNN グラフを作る（行を正規化した疎行列）。"""
    n = len(features)
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(similarity, -k, axis=1)[:, -k:]
    directed = sparse.csr_matrix((np.ones(n * k, dtype=np.float32), (np.repeat(np.arange(n), k), neighbors.ravel())), shape=(n, n))
    graph = directed.maximum(directed.T)
    return sparse.diags(1.0 / np.asarray(graph.sum(axis=1)).ravel()) @ graph


def edge_graph_of(edge_embed: np.ndarray) -> sparse.csr_matrix:
    """EdgeTAM の image_embed (1, C, h, w) のコサイン類似度の kNN グラフ（パッチ同士）。PE の推論と重ねて別スレッドで求める。"""
    return knn_graph(edge_features_of(edge_embed), EDGE_PROPAGATION_NEIGHBORS)


def propagate(values: np.ndarray, graph: sparse.csr_matrix, alpha: float, iterations: int) -> np.ndarray:
    """values (N,) をグラフ上で value ← (1-α)·values0 + α·(近傍の平均) と iterations 回伝播する。"""
    propagated = values.astype(np.float32)
    for _ in range(iterations):
        propagated = (1.0 - alpha) * values + alpha * (graph @ propagated)
    return propagated


def propagate_confidence(
    confidence: np.ndarray,
    prompt_index: np.ndarray,
    graph: sparse.csr_matrix,
    num_foreground_prompts: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える。(前景確信度, プロンプト番号) を返す。

    判定が反転したパッチ（確信度がしきい値をまたいだパッチ）のプロンプトは、反転後と同じ側に判定されていた近傍のプロンプトの最頻値にする
    （無ければ画像全体でその側の最頻値）。
    """
    propagated = propagate(confidence, graph, EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS)

    old_foreground, new_foreground = confidence > threshold, propagated > threshold
    old_prompt_index = prompt_index
    prompt_index = prompt_index.copy()
    for patch in np.nonzero(old_foreground != new_foreground)[0]:
        neighbors = graph.indices[graph.indptr[patch]:graph.indptr[patch + 1]]
        # 近傍のプロンプトは、反転する前の判定のもの（先に反転した近傍の新しいプロンプトを読むと、前景のパッチに背景のプロンプトが付くことがある）
        candidates = old_prompt_index[neighbors[old_foreground[neighbors] == new_foreground[patch]]]
        if not len(candidates):
            candidates = old_prompt_index[old_foreground == new_foreground[patch]]
        if len(candidates):
            prompt_index[patch] = np.bincount(candidates).argmax()
        else:
            prompt_index[patch] = 0 if new_foreground[patch] else num_foreground_prompts
    return propagated, prompt_index
