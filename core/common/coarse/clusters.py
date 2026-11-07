"""① KMeans でパッチをクラスタにまとめ、② クラスタごとに代表トークンをテキストと照合して、前景/背景とプロンプトを決める。

パッチごとにテキストと照合すると斑点だらけになるので、似たパッチを K 個にまとめて、判定を K 回にする。
クラスタ単位で判定する代償は、クラスタに前景と背景が混ざると全体が誤判定されること（③ の伝播がこれを減らす）。
"""

from typing import List, Tuple

import faiss
import numpy as np

from core.common.backend import CLIPBackend
from core.common.params import FAISS_NUM_THREADS, KMEANS_SEED
from core.common.prompts import PromptBank
from core.utils import softmax


def kmeans_labels(features: np.ndarray, k: int, niter: int) -> np.ndarray:
    """features (N, D) を k 個に KMeans でクラスタ分けし、N 個のクラスタ番号（平坦）を返す。"""
    _, feature_dim = features.shape
    features = np.ascontiguousarray(features, dtype=np.float32)  # faiss は C 連続の float32 を要求する
    faiss.omp_set_num_threads(FAISS_NUM_THREADS)
    kmeans = faiss.Kmeans(
        d=feature_dim, k=k, niter=niter, spherical=False, verbose=False, seed=KMEANS_SEED
    )
    kmeans.train(features)
    _, cluster_labels = kmeans.index.search(features, 1)
    return cluster_labels.reshape(-1)


def classify_logits(prompt_logits: np.ndarray, num_foreground_prompts: int, threshold: float) -> Tuple[float, int]:
    """プロンプトごとのロジット（前景 -> 背景の順）から、(前景確信度, 最も近いプロンプト番号) を返す。

    プロンプト番号は 前景プロンプト -> 背景プロンプト の通し番号。前景確信度が threshold を超えれば前景のプロンプト、でなければ背景のプロンプト。
    """
    logits = prompt_logits[None, :]
    fg_logits = logits[:, :num_foreground_prompts]
    bg_logits = logits[:, num_foreground_prompts:]

    # グループスコアは最も近いプロンプトの類似度。プロンプトは「言い換え」ではなく「選択肢」の
    # 列挙なので、平均ではなく最大を使う（プロンプト数を増減しても判定が偏らない）
    fg_score = np.max(fg_logits, axis=-1)
    bg_score = np.max(bg_logits, axis=-1)

    # FG vs BG の2クラスsoftmax
    group_logits = np.concatenate([fg_score[:, None], bg_score[:, None]], axis=-1)
    foreground_confidence = float(softmax(group_logits, axis=-1)[0, 0])

    # 前景の場合（前景の度合い、前景のプロンプトのインデックス）
    if foreground_confidence > threshold:
        return foreground_confidence, int(np.argmax(fg_logits, axis=1)[0])
    # 背景の場合（前景の度合い、背景のプロンプトのインデックス）
    else:
        return foreground_confidence, num_foreground_prompts + int(np.argmax(bg_logits, axis=1)[0])


def judge_clusters(
    extractor: CLIPBackend,
    prompts: PromptBank,
    threshold: float,
    patch_features: np.ndarray,
    cluster_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """クラスタごとのトークンをテキストと照合し、(パッチごとの前景確信度, パッチごとのプロンプト番号, クラスタごとの前景プロンプトのロジット) を返す。

    ロジットは (クラスタ数, 前景プロンプト数)。前景のクラスの付け直し（relabel.py）で、領域の名前の判定し直しの根拠に使う。
    """
    cluster_tokens: List[np.ndarray] = extractor.extract_cluster_tokens(patch_features, cluster_labels)
    confidences = np.zeros((len(cluster_tokens),), dtype=np.float32)
    prompt_indices = np.zeros((len(cluster_tokens),), dtype=np.int64)
    cluster_logits = np.zeros((len(cluster_tokens), prompts.num_foreground), dtype=np.float32)
    for label_idx, cluster_feature in enumerate(cluster_tokens):
        prompt_logits = prompts.logits(cluster_feature)
        confidences[label_idx], prompt_indices[label_idx] = classify_logits(prompt_logits, prompts.num_foreground, threshold)
        cluster_logits[label_idx] = prompt_logits[:prompts.num_foreground]
    return confidences[cluster_labels], prompt_indices[cluster_labels], cluster_logits
