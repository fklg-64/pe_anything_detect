"""① 粗い段階の進行役。CLIP（PE）の特徴量から、パッチ格子の上の「前景らしさ」と「粗いセマンティックラベル」を求める。

流れ（generate_foreground_map。各段階の実装は同じパッケージの各ファイル）:
  ① KMeans             PE の特徴を K 個のクラスタにまとめる                                    clusters.kmeans_labels
  ② クラスタの判定     クラスタごとに代表トークンをテキストと照合し、前景確信度とプロンプトを決める   clusters.judge_clusters
  ③ 確信度の伝播       EdgeTAM の特徴が似たパッチどうしで、前景確信度を揃える                    propagation.propagate_confidence
  ④ クラスの付け直し   前景のパッチだけのブロックの根拠で、名前を判定し直し・統合し・境目を整える   relabel.ForegroundRelabeler
  ⑤ ラベルの付与       クラスタ番号とプロンプトの組をクラスタにし、パッチごとの連番ラベルにする    assign_semantic_labels
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import sparse

from core.utils import Timer, normalize_image
from core.common.backend import CLIPBackend
from core.common.coarse.clusters import judge_clusters, kmeans_labels
from core.common.coarse.propagation import edge_graph_of, propagate_confidence
from core.common.coarse.relabel import ForegroundRelabeler
from core.common.prompts import PromptBank
from core.common.result import ClusterView


class CoarseForeground(NamedTuple):
    """generate_foreground_map の結果（パッチ格子上）。タプルとして取り出すこともできる。"""

    foreground_map: np.ndarray      # (格子の高さ, 格子の幅) 前景確信度
    foreground_mask: np.ndarray     # 確信度が閾値を超えるパッチを1とした2値マスク
    cluster_labels: np.ndarray      # パッチごとのクラスタ番号（平坦）。PE のクラスタ × 伝播後のプロンプトの組
    cluster_index_list: List[int]   # cluster_labels のクラスタごとの対応プロンプト番号（前景プロンプト -> 背景プロンプトの通し番号）
    pe_clusters: ClusterView        # 伝播前の PE だけのクラスタ


class CoarseSegmenter:
    """画像をパッチ格子上の「前景らしさ」と「粗いセマンティックラベル」に分ける。"""

    def __init__(self, extractor: CLIPBackend, config: dict):
        self.extractor = extractor
        self.prompts = PromptBank(extractor, config["prompts_dir"])
        self.k_clusters = config["k_clusters"]
        self.niter = config["n_iter"]
        self.foreground_threshold = config["foreground_threshold"]
        self.feature_size: int = extractor.feature_size
        # 処理する画像のパッチ格子 (高さ, 幅)
        self.grid: Tuple[int, int] = (self.feature_size, self.feature_size)
        self.relabeler = ForegroundRelabeler(extractor, self.prompts, self.grid, self.k_clusters, self.niter)

    # =========================================================================
    # CLIP特徴量
    # =========================================================================

    def extract_features(self, resized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """リサイズ済み画像（正規化前）を正規化し、CLIP（PE）のパッチ特徴と、クラスタリングに使う層を連結した（fused）パッチ特徴を返す。"""
        image_tensor = normalize_image(resized, self.extractor.mean, self.extractor.std)
        return self.extractor.extract_image_features(image_tensor)

    # =========================================================================
    # 前景マップ（① KMeans -> ② 照合 -> ③ 伝播 -> ④ 付け直し）
    # =========================================================================

    def generate_foreground_map(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray,
        edge_embed: np.ndarray,
        edge_graph: Optional[sparse.csr_matrix] = None,
    ) -> CoarseForeground:
        """パッチ特徴をKMeansでクラスタ分けし、クラスタごとの前景確信度と対応プロンプトを求める。

        PE の判定で得た前景確信度は、edge_embed（EdgeTAMエンコーダの image_embed (1, C, h, w)）の特徴の
        類似度が高いパッチどうしで伝播して揃える（propagation.py）。
        edge_graph: edge_embed の kNN グラフ（propagation.edge_graph_of の結果）。PE の実行と重ねて先に求めてあれば渡す（None なら、ここで求める）。

        Returns: CoarseForeground（各フィールドの意味はそちら）。
        """
        patch_tokens = fused_patch_features[0, 1:, :]
        if edge_graph is None:  # 並列化をしない場合はここでグラフを計算する
            edge_graph = edge_graph_of(edge_embed)

        # ① PE の特徴を KMeans でクラスタ分けし、② クラスタごとにテキストと照合して、パッチごとの前景確信度とプロンプトを得る
        with Timer("    ┌ Kmeansクラスタ処理時間"):
            pe_labels = kmeans_labels(patch_tokens, k=self.k_clusters, niter=self.niter)
        with Timer("    ┌ クラスタごとのCLSトークン抽出処理時間"):
            confidence, prompt_index, cluster_logits = judge_clusters(
                self.extractor, self.prompts, self.foreground_threshold, patch_features, pe_labels
            )

        # ③ EdgeTAM の特徴が似たパッチどうしで、前景確信度を伝播して揃える
        pe_clusters = ClusterView(pe_labels, self._prompt_per_cluster(pe_labels, prompt_index))
        with Timer("    ┌ EdgeTAM特徴での確信度の伝播処理時間"):
            confidence, prompt_index = propagate_confidence(
                confidence, prompt_index, edge_graph, self.prompts.num_foreground, self.foreground_threshold
            )

        # ④ 前景のパッチのクラスを、前景だけのブロックの代表トークンの照合と EdgeTAM の特徴での伝播で付け直す
        #    （種類の違う物体を分け、1 つの物体が似たプロンプトの別クラスに割れるのを戻し、部品の境目の混ざりを整える）
        with Timer("    ┌ 前景のクラスの付け直し処理時間"):
            prompt_index = self.relabeler.relabel(
                prompt_index, confidence > self.foreground_threshold, patch_features, patch_tokens, edge_embed, edge_graph,
                pe_labels, cluster_logits,
            )

        # クラスタ番号を付け直す。伝播で判定が変わったパッチは、同じ PE のクラスタの中でもプロンプトが違うので、(PE のクラスタ, プロンプト) の組を新しいクラスタにする
        num_prompts = self.prompts.num_foreground + len(self.prompts.background)
        _, cluster_labels = np.unique(pe_labels * num_prompts + prompt_index, return_inverse=True)
        cluster_labels = cluster_labels.reshape(-1)
        cluster_index_list = self._prompt_per_cluster(cluster_labels, prompt_index)

        foreground_map = confidence.reshape(self.grid).astype(np.float32)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)
        return CoarseForeground(foreground_map, foreground_mask, cluster_labels, cluster_index_list, pe_clusters)

    @staticmethod
    def _prompt_per_cluster(cluster_labels: np.ndarray, prompt_index: np.ndarray) -> List[int]:
        """クラスタ番号 0~ごとの対応プロンプト番号。プロンプトはクラスタの中では一定なので、先頭のパッチから取る。"""
        return [int(prompt_index[np.argmax(cluster_labels == label_idx)]) for label_idx in range(int(cluster_labels.max()) + 1)]

    # =========================================================================
    # ⑤ 粗いセマンティックラベル
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
