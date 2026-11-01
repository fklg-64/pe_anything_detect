"""CLIP特徴量とKMeansによる、パッチ単位の粗い前景/セマンティック推定。"""

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import faiss
import numpy as np
import yaml
from scipy import sparse

from core.utils import Timer, normalize_image
from core.common.backend import CLIPBackend
from core.common.mask_utils import knn_graph, propagate, softmax
from core.common.params import (
    CLIP_LOGIT_SCALE,
    EDGE_PROPAGATION_ALPHA,
    EDGE_PROPAGATION_ITERATIONS,
    EDGE_PROPAGATION_NEIGHBORS,
    FAISS_NUM_THREADS,
    KMEANS_SEED,
    PROMPT_SYNONYM_DISTANCE,
    PROMPT_TIE_TOLERANCE,
)
from core.common.semantic_merge import merge_near_ties, merge_synonyms_in_objects, synonym_representatives
from core.common.result import ClusterView


def load_prompts(path: Path) -> List[str]:
    """プロンプトの yaml から、base（あれば）と custom を連結して返す。"""
    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("base", []) + data["custom"]


class CoarseForeground(NamedTuple):
    """generate_foreground_map の結果（パッチ格子上）。タプルとして取り出すこともできる。"""

    foreground_map: np.ndarray      # (格子の高さ, 格子の幅) 前景確信度
    foreground_mask: np.ndarray     # 確信度が閾値を超えるパッチを1とした2値マスク
    cluster_labels: np.ndarray      # パッチごとのクラスタ番号（平坦）。EdgeTAM 併用時は PE のクラスタ × 伝播後のプロンプトの組
    cluster_index_list: List[int]   # cluster_labels のクラスタごとの対応プロンプト番号（前景プロンプト -> 背景プロンプトの通し番号）
    pe_clusters: Optional[ClusterView] = None    # 伝播前の PE だけのクラスタ。EdgeTAM を併用しないときは None


class CoarseSegmenter:
    """画像をパッチ格子上の「前景らしさ」と「粗いセマンティックラベル」に分ける。"""

    def __init__(self, extractor: CLIPBackend, config: dict):
        self.extractor = extractor
        self.prompts_dir = config["prompts_dir"]
        self.k_clusters = config["k_clusters"]
        self.niter = config["n_iter"]
        self.foreground_threshold = config["foreground_threshold"]
        self.feature_size: int = extractor.feature_size
        # 処理する画像のパッチ格子 (高さ, 幅)
        self.grid: Tuple[int, int] = (self.feature_size, self.feature_size)

        # プロンプトとそのテキスト特徴量（前景プロンプト -> 背景プロンプトの順に並べる）
        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: np.ndarray
        self._register_text_features()

    def _register_text_features(self) -> None:
        prompts_dir = Path(self.prompts_dir)
        self.foreground_prompts = load_prompts(prompts_dir / "foreground.yaml")
        self.background_prompts = load_prompts(prompts_dir / "background.yaml")

        fg_feats = self.extractor.extract_text_features(self.foreground_prompts)
        bg_feats = self.extractor.extract_text_features(self.background_prompts)
        self.fgbg_text_features = np.vstack([fg_feats, bg_feats])
        # 同義語の前景プロンプトの代表（1 つの物体の中で、同義語のクラスをまとめるのに使う）
        self._synonym_representative = synonym_representatives(fg_feats, PROMPT_SYNONYM_DISTANCE)

    # =========================================================================
    # CLIP特徴量
    # =========================================================================

    def extract_features(self, resized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """リサイズ済み画像（正規化前）を正規化し、CLIP（PE）のパッチ特徴と、クラスタリングに使う層を連結した（fused）パッチ特徴を返す。"""
        image_tensor = normalize_image(resized, self.extractor.mean, self.extractor.std)
        return self.extractor.extract_image_features(image_tensor)

    # =========================================================================
    # 前景マップ（KMeans -> クラスタごとにテキストと照合）
    # =========================================================================

    def fit_predict_kmeans(self, features: np.ndarray, k: int, niter: int) -> np.ndarray:
        """features (N, D) を k 個にKMeansでクラスタ分けし、N 個のクラスタ番号（平坦）を返す。"""
        _, feature_dim = features.shape
        features = np.ascontiguousarray(features, dtype=np.float32)  # faiss は C 連続の float32 を要求する
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)
        kmeans = faiss.Kmeans(
            d=feature_dim, k=k, niter=niter, spherical=False, verbose=False, seed=KMEANS_SEED
        )
        kmeans.train(features)
        _, cluster_labels = kmeans.index.search(features, 1)
        return cluster_labels.reshape(-1)

    def _prompt_logits(self, cluster_feature: np.ndarray) -> np.ndarray:
        """クラスタの代表トークンと全プロンプト（前景 -> 背景の順）との類似度 × CLIP_LOGIT_SCALE。"""
        feature = cluster_feature.reshape(1, -1)
        feature = feature / (np.linalg.norm(feature, axis=-1, keepdims=True) + 1e-8)
        return (CLIP_LOGIT_SCALE * feature @ self.fgbg_text_features.T)[0]

    def _classify_cluster(self, cluster_feature: np.ndarray) -> Tuple[float, int]:
        """クラスタの代表トークンを前景/背景プロンプトと照合し、(前景確信度, 最も近いプロンプト番号) を返す。

        プロンプト番号は 前景プロンプト -> 背景プロンプト の通し番号。
        """
        return self._classify_logits(self._prompt_logits(cluster_feature))

    def _classify_logits(self, prompt_logits: np.ndarray) -> Tuple[float, int]:
        """プロンプトごとのロジット（前景 -> 背景の順）から、(前景確信度, 最も近いプロンプト番号) を返す。"""
        num_foreground_prompts = len(self.foreground_prompts)

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

        if foreground_confidence > self.foreground_threshold:
            return foreground_confidence, int(np.argmax(fg_logits, axis=1)[0])
        return foreground_confidence, num_foreground_prompts + int(np.argmax(bg_logits, axis=1)[0])

    def _judge_clusters(
        self, patch_features: np.ndarray, cluster_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """クラスタごとのトークンをテキストと照合し、(パッチごとの前景確信度, パッチごとのプロンプト番号, クラスタごとの前景プロンプトのロジット) を返す。

        ロジットは (クラスタ数, 前景プロンプト数)。僅差のクラスのまとめ（merge_near_ties）で使う。
        """
        cluster_tokens: List[np.ndarray] = self.extractor.extract_cluster_tokens(patch_features, cluster_labels)
        confidences = np.zeros((len(cluster_tokens),), dtype=np.float32)
        prompt_indices = np.zeros((len(cluster_tokens),), dtype=np.int64)
        num_foreground_prompts = len(self.foreground_prompts)
        cluster_logits = np.zeros((len(cluster_tokens), num_foreground_prompts), dtype=np.float32)
        for label_idx, cluster_feature in enumerate(cluster_tokens):
            prompt_logits = self._prompt_logits(cluster_feature)
            confidences[label_idx], prompt_indices[label_idx] = self._classify_logits(prompt_logits)
            cluster_logits[label_idx] = prompt_logits[:num_foreground_prompts]
        return confidences[cluster_labels], prompt_indices[cluster_labels], cluster_logits

    def generate_foreground_map(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray,
        edge_embed: Optional[np.ndarray] = None,
        edge_graph: Optional[sparse.csr_matrix] = None,
    ) -> CoarseForeground:
        """パッチ特徴をKMeansでクラスタ分けし、クラスタごとの前景確信度と対応プロンプトを求める。

        edge_embed（EdgeTAMエンコーダの image_embed (1, C, h, w)）を渡すと、PE の判定で得た前景確信度を、EdgeTAM の特徴の
        類似度が高いパッチどうしで伝播して揃える（_propagate_with_edgetam）。
        edge_graph: edge_embed の kNN グラフ（edge_graph_of の結果）。PE の実行と重ねて先に求めてあれば渡す（None なら、ここで求める）。

        Returns: CoarseForeground（各フィールドの意味はそちら）。
        """
        patch_tokens = fused_patch_features[0, 1:, :]

        # 1. PE の特徴を KMeans でクラスタ分けし、クラスタごとにテキストと照合して、パッチごとの前景確信度とプロンプトを得る
        with Timer("    ┌ Kmeansクラスタ処理時間"):
            pe_labels = self.fit_predict_kmeans(patch_tokens, k=self.k_clusters, niter=self.niter)
        with Timer("    ┌ クラスタごとのCLSトークン抽出処理時間"):
            confidence, prompt_index, cluster_logits = self._judge_clusters(patch_features, pe_labels)

        # 2. EdgeTAM の特徴が似たパッチどうしで、前景確信度を伝播して揃える
        pe_clusters = None
        if edge_embed is not None:
            pe_clusters = ClusterView(pe_labels, self._prompt_per_cluster(pe_labels, prompt_index))
            with Timer("    ┌ EdgeTAM特徴での確信度の伝播処理時間"):
                confidence, prompt_index = self._propagate_with_edgetam(confidence, prompt_index, edge_embed, edge_graph)

        # 3. 1 つの物体が、似たプロンプトの別クラスに割れるのを防ぐ
        prompt_index = self._merge_similar_classes(prompt_index, pe_labels, cluster_logits)

        # 4. クラスタ番号を付け直す。伝播で判定が変わったパッチは、同じ PE のクラスタの中でもプロンプトが違うので、(PE のクラスタ, プロンプト) の組を新しいクラスタにする
        cluster_labels = pe_labels
        if edge_embed is not None:
            num_prompts = len(self.foreground_prompts) + len(self.background_prompts)
            _, cluster_labels = np.unique(pe_labels * num_prompts + prompt_index, return_inverse=True)
            cluster_labels = cluster_labels.reshape(-1)
        cluster_index_list = self._prompt_per_cluster(cluster_labels, prompt_index)

        foreground_map = confidence.reshape(self.grid).astype(np.float32)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)
        return CoarseForeground(foreground_map, foreground_mask, cluster_labels, cluster_index_list, pe_clusters)

    def _merge_similar_classes(self, prompt_index: np.ndarray, pe_labels: np.ndarray, cluster_logits: np.ndarray) -> np.ndarray:
        """1 つの物体の別のクラスタが、似たプロンプトを選んで別クラスに割れるのを防ぐ（semantic_merge.py）。

        ロジットが僅差のプロンプトは、より大きいクラスへ統合する。僅差でなくても、同じ物体（連結成分）の中の同義語のクラスは 1 つにする。
        """
        num_foreground_prompts = len(self.foreground_prompts)
        prompt_index = merge_near_ties(prompt_index, pe_labels, cluster_logits, num_foreground_prompts, PROMPT_TIE_TOLERANCE)
        return merge_synonyms_in_objects(prompt_index, self.grid, num_foreground_prompts, self._synonym_representative)

    @staticmethod
    def _prompt_per_cluster(cluster_labels: np.ndarray, prompt_index: np.ndarray) -> List[int]:
        """クラスタ番号 0~ごとの対応プロンプト番号。プロンプトはクラスタの中では一定なので、先頭のパッチから取る。"""
        return [int(prompt_index[np.argmax(cluster_labels == label_idx)]) for label_idx in range(int(cluster_labels.max()) + 1)]

    @staticmethod
    def edge_graph_of(edge_embed: np.ndarray) -> sparse.csr_matrix:
        """EdgeTAM の image_embed (1, C, h, w) のコサイン類似度の kNN グラフ（パッチ同士）。PE の結果に依らないので、PE の実行中に求められる。"""
        edge_features = edge_embed[0].reshape(edge_embed.shape[1], -1).T
        edge_features = edge_features / (np.linalg.norm(edge_features, axis=-1, keepdims=True) + 1e-8)
        return knn_graph(edge_features, EDGE_PROPAGATION_NEIGHBORS)

    def _propagate_with_edgetam(
        self, confidence: np.ndarray, prompt_index: np.ndarray, edge_embed: np.ndarray, graph: Optional[sparse.csr_matrix] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える。(前景確信度, プロンプト番号) を返す。

        PE のクラスタは意味は近いが空間的にばらけ、背景の斑点・織り目の一部が前景のクラスタに混ざる（誤検出）。
        EdgeTAM の特徴は物体の輪郭・領域に沿ってまとまるので、その類似度の kNN グラフで確信度を平均して、物体の構造に沿って揃える。
        前景を消す（AND のように）のではなく、確信度をならすだけなので、大きな物体は崩れにくい。判定が反転したパッチの
        プロンプトは、反転後と同じ側に判定されていた近傍のプロンプトの最頻値にする（無ければ画像全体でその側の最頻値）。
        """
        if graph is None:
            graph = self.edge_graph_of(edge_embed)
        propagated = propagate(confidence, graph, EDGE_PROPAGATION_ALPHA, EDGE_PROPAGATION_ITERATIONS)

        num_foreground_prompts = len(self.foreground_prompts)
        threshold = self.foreground_threshold
        old_foreground, new_foreground = confidence > threshold, propagated > threshold
        prompt_index = prompt_index.copy()
        for patch in np.nonzero(old_foreground != new_foreground)[0]:
            neighbors = graph.indices[graph.indptr[patch]:graph.indptr[patch + 1]]
            candidates = prompt_index[neighbors[old_foreground[neighbors] == new_foreground[patch]]]
            if not len(candidates):
                candidates = prompt_index[old_foreground == new_foreground[patch]]
            if len(candidates):
                prompt_index[patch] = np.bincount(candidates).argmax()
            else:
                prompt_index[patch] = 0 if new_foreground[patch] else num_foreground_prompts
        return propagated, prompt_index

    # =========================================================================
    # 粗いセマンティックラベル
    # =========================================================================

    def assign_semantic_labels(
        self,
        cluster_labels: np.ndarray,
        cluster_index_list: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """パッチごとの対応プロンプト番号を連番に直す。

        Returns:
            labels_list: 連番 -> プロンプト番号。
            semantic_labels: (格子の高さ, 格子の幅) のパッチごとの連番ラベル。
        """
        semantic_indices = np.asarray(cluster_index_list)[cluster_labels]
        labels_list, semantic_labels = np.unique(semantic_indices, return_inverse=True)
        return labels_list, semantic_labels.reshape(self.grid)
