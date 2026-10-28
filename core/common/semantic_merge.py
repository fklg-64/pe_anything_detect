"""セマンティック（クラス）のまとめ方。1 つの物体が、似たプロンプトの別クラスに割れるのを防ぐ。

粗い判定は、クラスタごとに最も近い前景プロンプトを選ぶ。物体の中の別のクラスタが、ロジットが僅差の似たプロンプト（例: tube と
transparent tubes）をそれぞれ選ぶと、1 つの物体が複数のクラスに割れる。クラスタのロジットの推定にはノイズがあるので、僅差は「違う」とは言えない。
"""

from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def merge_near_ties(
    prompt_index: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_logits: np.ndarray,
    num_foreground_prompts: int,
    tolerance: float,
) -> np.ndarray:
    """前景のパッチのプロンプトを、ロジットの僅差の範囲でより大きなクラスにまとめる。

    各（PE のクラスタ、プロンプト）の組を、面積の小さい順に見る。その PE クラスタのロジットで、今のプロンプトとの差が tolerance 以内にあり、
    しかも今のクラスより大きい（前景のパッチ数が多い）前景プロンプトがあれば、その中で最も大きいものへ移す。差が大きい（別のものだと言える）
    プロンプトへは移さない。違うクラスのままになるのは、ロジットの差がノイズで説明できないものだけ。

    prompt_index: パッチごとのプロンプト番号（前景プロンプト < num_foreground_prompts <= 背景プロンプト）。変更せず、新しい配列を返す。
    cluster_labels: パッチごとの PE のクラスタ番号。cluster_logits: (クラスタ数, 前景プロンプト数) のロジット（正規化したトークンとテキストの類似度 × CLIP_LOGIT_SCALE）。
    tolerance が負なら何もしない（実験で、まとめない基準と比べるため）。
    """
    result = prompt_index.copy()
    foreground = result < num_foreground_prompts
    if tolerance < 0 or not foreground.any():
        return result
    weights = np.bincount(result[foreground], minlength=num_foreground_prompts).astype(np.float64)
    pairs: Dict[Tuple[int, int], int] = {}
    for cluster, prompt in zip(cluster_labels[foreground], result[foreground]):
        pairs[(int(cluster), int(prompt))] = pairs.get((int(cluster), int(prompt)), 0) + 1
    for (cluster, prompt), area in sorted(pairs.items(), key=lambda item: item[1]):
        logits = cluster_logits[cluster, :num_foreground_prompts]
        candidates = np.flatnonzero((logits >= logits[prompt] - tolerance) & (weights > weights[prompt]))
        if not len(candidates):
            continue
        target = int(candidates[np.argmax(weights[candidates])])
        result[(cluster_labels == cluster) & (result == prompt) & foreground] = target
        weights[target] += area
        weights[prompt] -= area
    return result


def synonym_representatives(text_features: np.ndarray, distance: float) -> np.ndarray:
    """前景プロンプトを、テキスト埋め込みのコサイン距離（平均連結の階層クラスタリング）が distance 未満の同義語の組にまとめ、
    プロンプトごとに代表プロンプトの番号を返す（代表は、組の中で他のメンバーとの平均類似度が最も高いもの。同点は yaml の順で先）。

    text_features: (前景プロンプト数, D)（L2 正規化済み）。組になるのは、tube と transparent tubes（類似度 0.94）・pill と capsule（0.93）・
    cable と wire（0.95）のように、同じ物を言い換えただけのもの。white plastic lid と round white food container（0.92）・noodle と package（0.85）のような、別の部品を指すものは組にしない。
    """
    count = len(text_features)
    representative = np.arange(count)
    if count < 2:
        return representative
    groups = fcluster(linkage(text_features, "average", metric="cosine"), distance, "distance")
    similarity = text_features @ text_features.T
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        if len(members) > 1:
            representative[members] = members[np.argmax(similarity[np.ix_(members, members)].mean(axis=1))]
    return representative


def merge_synonyms_in_objects(
    prompt_index: np.ndarray, grid: Tuple[int, int], num_foreground_prompts: int, representative: np.ndarray
) -> np.ndarray:
    """1 つの物体（前景の連結成分）の中にある、同義語（representative が同じ）のクラスを、その物体の中で最も大きいクラスにまとめる。

    ロジットの差が僅差でなくても（ノイズより大きくても）、同じ物の言い換えなら、1 つの物体を 2 つのクラスにする意味は無い。
    別の物体（連結していない領域）は触らない。カップ麺の麺・包装・ふたのような、別の部品を指すクラスは、同義語ではないのでそのまま。
    """
    result = prompt_index.copy()
    foreground = (result < num_foreground_prompts).reshape(grid)
    count, components = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    components = components.reshape(-1)
    for component in range(1, count):
        mask = components == component
        present, areas = np.unique(result[mask], return_counts=True)
        if len(present) < 2:
            continue
        for group in np.unique(representative[present]):
            members = present[representative[present] == group]
            if len(members) > 1:
                top = members[np.argmax(areas[np.isin(present, members)])]
                result[mask & np.isin(result, members)] = top
    return result
