"""CLIP特徴量とKMeansによる、パッチ単位の粗い前景/セマンティック推定。"""

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import cv2
import faiss
import numpy as np
import yaml
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from core.utils import Timer
from core.common.backend import CLIPBackend
from core.common.mask_utils import knn_graph, propagate, softmax
from core.common.params import (
    CLIP_LOGIT_SCALE,
    EDGE_PROPAGATION_ALPHA,
    EDGE_PROPAGATION_ITERATIONS,
    EDGE_PROPAGATION_NEIGHBORS,
    FAISS_NUM_THREADS,
    KMEANS_SEED,
    PATCH_NEIGHBORHOOD,
    PROMPT_SYNONYM_DISTANCE,
    PROMPT_TIE_TOLERANCE,
)
from core.common.semantic_merge import merge_near_ties, merge_synonyms_in_objects, synonym_representatives
from core.common.result import ClusterView


def load_plain_prompts(path: Path) -> List[str]:
    """base と custom のプロンプトをそのまま連結して返す。"""
    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["base"] + data["custom"]


def load_color_shape_prompts(path: Path) -> List[str]:
    """color x shape の全組み合わせから "a photo of a <color> <shape> object." を作り、custom を足して返す。"""
    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    prompts = []
    for color in data["color"]:
        for shape in data["shape"]:
            article = "an" if color[0].lower() in "aeiou" else "a"
            prompts += [f"a photo of {article} {color} {shape} object."]
    return prompts + data["custom"]


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
        # 処理する画像のパッチ格子 (高さ, 幅)。アスペクト比を維持するモードでは画像ごとに変わるので、segment の前に runner が設定する
        self.grid: Tuple[int, int] = (self.feature_size, self.feature_size)

        # プロンプトとそのテキスト特徴量（前景プロンプト -> 背景プロンプトの順に並べる）
        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: np.ndarray
        self._register_text_features()

    def _register_text_features(self) -> None:
        prompts_dir = Path(self.prompts_dir)
        self.foreground_prompts = load_color_shape_prompts(prompts_dir / "foreground.yaml")
        self.background_prompts = load_plain_prompts(prompts_dir / "background.yaml")

        fg_feats = self.extractor.extract_text_features(self.foreground_prompts)
        bg_feats = self.extractor.extract_text_features(self.background_prompts)
        self.fgbg_text_features = np.vstack([fg_feats, bg_feats])
        # 同義語の前景プロンプトの代表（1 つの物体の中で、同義語のクラスをまとめるのに使う）
        self.synonym_representative = synonym_representatives(fg_feats, PROMPT_SYNONYM_DISTANCE)
        self.tie_tolerance = PROMPT_TIE_TOLERANCE   # 実験で runner ごとに変えられるよう、インスタンスの属性にしてある（yaml には出さない）

    # =========================================================================
    # CLIP特徴量
    # =========================================================================

    def extract_features(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 画像全体の特徴量は使わないので正規化を省く
        image_features, patch_features, fused_patch_features = (
            self.extractor.extract_image_features(image_tensor, normalize=False)
        )
        return image_features, patch_features, fused_patch_features

    # =========================================================================
    # 前景マップ（KMeans -> クラスタごとにテキストと照合）
    # =========================================================================

    def fit_predict_kmeans(self, features: np.ndarray, k: int = 12, niter: int = 10) -> np.ndarray:
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
        num_foreground_prompts = len(self.foreground_prompts)

        logits = self._prompt_logits(cluster_feature)[None, :]
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """クラスタごとのトークンをテキストと照合し、パッチごとの (前景確信度, プロンプト番号) を返す。

        クラスタごとの前景プロンプトのロジットは self.cluster_logits (クラスタ数, 前景プロンプト数) に残す（僅差のクラスのまとめで使う）。
        """
        cluster_tokens: List[np.ndarray] = self.extractor.extract_cluster_tokens(patch_features, cluster_labels)
        confidences = np.zeros((len(cluster_tokens),), dtype=np.float32)
        prompt_indices = np.zeros((len(cluster_tokens),), dtype=np.int64)
        num_foreground_prompts = len(self.foreground_prompts)
        self.cluster_logits = np.zeros((len(cluster_tokens), num_foreground_prompts), dtype=np.float32)
        for label_idx, cluster_feature in enumerate(cluster_tokens):
            confidences[label_idx], prompt_indices[label_idx] = self._classify_cluster(cluster_feature)
            self.cluster_logits[label_idx] = self._prompt_logits(cluster_feature)[:num_foreground_prompts]
        return confidences[cluster_labels], prompt_indices[cluster_labels]

    def generate_foreground_map(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray,
        edge_embed: Optional[np.ndarray] = None,
    ) -> CoarseForeground:
        """パッチ特徴をKMeansでクラスタ分けし、クラスタごとの前景確信度と対応プロンプトを求める。

        edge_embed（EdgeTAMエンコーダの image_embed (1, C, h, w)）を渡すと、PE の判定で得た前景確信度を、EdgeTAM の特徴の
        類似度が高いパッチどうしで伝播して揃える（_propagate_with_edgetam）。

        Returns: CoarseForeground（各フィールドの意味はそちら）。
        """
        patch_tokens = fused_patch_features[0, 1:, :]

        with Timer("    ┌ Kmeansクラスタ処理時間"):
            cluster_labels = self.fit_predict_kmeans(patch_tokens, k=self.k_clusters, niter=self.niter)

        with Timer("    ┌ クラスタごとのCLSトークン抽出処理時間"):
            confidence, prompt_index = self._judge_clusters(patch_features, cluster_labels)

        pe_clusters = None
        pe_labels = cluster_labels
        if edge_embed is not None:
            pe_clusters = ClusterView(cluster_labels, self._prompt_per_cluster(cluster_labels, prompt_index))
            with Timer("    ┌ EdgeTAM特徴での確信度の伝播処理時間"):
                confidence, prompt_index = self._propagate_with_edgetam(confidence, prompt_index, edge_embed)
        # 1 つの物体の別のクラスタが、僅差の似たプロンプトを選んで別クラスに割れるのを防ぐ
        prompt_index = merge_near_ties(prompt_index, pe_labels, self.cluster_logits, len(self.foreground_prompts), self.tie_tolerance)
        # ロジットの差が僅差でなくても、同じ物体の中の同義語のクラスは 1 つにする
        prompt_index = merge_synonyms_in_objects(prompt_index, self.grid, len(self.foreground_prompts), self.synonym_representative)
        if edge_embed is not None:
            # PE のクラスタの中でも、伝播で判定が変わったパッチはプロンプトが違うので、(PE のクラスタ, プロンプト) の組を新しいクラスタにする
            num_prompts = len(self.foreground_prompts) + len(self.background_prompts)
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

    def _propagate_with_edgetam(
        self, confidence: np.ndarray, prompt_index: np.ndarray, edge_embed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える。(前景確信度, プロンプト番号) を返す。

        PE のクラスタは意味は近いが空間的にばらけ、背景の斑点・織り目の一部が前景のクラスタに混ざる（誤検出）。
        EdgeTAM の特徴は物体の輪郭・領域に沿ってまとまるので、その類似度の kNN グラフで確信度を平均して、物体の構造に沿って揃える。
        前景を消す（AND のように）のではなく、確信度をならすだけなので、大きな物体は崩れにくい。判定が反転したパッチの
        プロンプトは、反転後と同じ側に判定されていた近傍のプロンプトの最頻値にする（無ければ画像全体でその側の最頻値）。
        """
        edge_features = edge_embed[0].reshape(edge_embed.shape[1], -1).T
        edge_features = edge_features / (np.linalg.norm(edge_features, axis=-1, keepdims=True) + 1e-8)
        graph = knn_graph(edge_features, EDGE_PROPAGATION_NEIGHBORS)
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

    def refine_coarse_segmentation(
        self, semantic_labels: np.ndarray, foreground_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """パッチ単位の粗い結果を整える。返り値は (前景マスク, セマンティックラベル)。

        1. 前景に囲まれた背景（反射・透明部などの穴）は前景とし、最寄りの前景ラベルを与える。
           最終的な形はEdgeTAMが決めるため、穴埋めは粗いマスクを物体単位でつなぐ目的に限る。
        2. 縁・影由来の細いクラス領域を取り除く。

        物体の縁や影のパッチは背景と物体のトークンが混ざり、別クラスタ（別プロンプト）になりやすい。
        幅が2パッチ以下（3x3で縮めると消える）のクラス領域が他クラスの前景に接していれば、
        最も長く接するクラスへ統合する。小さい領域から順に処理する。
        """
        original_fg = foreground_mask != 0
        fg = original_fg.copy()
        labels = semantic_labels.copy()

        # fg = self._fill_holes(fg, labels, original_fg)
        # self._merge_thin_regions(fg, labels)

        return fg.astype(foreground_mask.dtype), labels

    @staticmethod
    def _fill_holes(fg: np.ndarray, labels: np.ndarray, original_fg: np.ndarray) -> np.ndarray:
        """画像端に接しない背景の連結成分（穴）を前景にし、labels（in-place）には最寄りの前景ラベルを与える。"""
        n, comp = cv2.connectedComponents((~fg).astype(np.uint8), connectivity=4)
        border_ids = np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))
        holes = (comp > 0) & ~np.isin(comp, border_ids)
        if holes.any() and original_fg.any():
            _, (iy, ix) = distance_transform_edt(~original_fg, return_indices=True)
            labels[holes] = labels[iy, ix][holes]
            fg = fg | holes
        return fg

    @staticmethod
    def _merge_thin_regions(fg: np.ndarray, labels: np.ndarray) -> None:
        """細いクラス領域を、最も長く接する他クラスへ統合する（labels を in-place で更新）。"""
        regions = []
        for class_id in np.unique(labels[fg]):
            n, comp, stats, _ = cv2.connectedComponentsWithStats(
                (fg & (labels == class_id)).astype(np.uint8), connectivity=4
            )
            regions += [(stats[i, cv2.CC_STAT_AREA], class_id, comp == i) for i in range(1, n)]

        for _, class_id, region in sorted(regions, key=lambda r: r[0]):
            region &= labels == class_id
            if not region.any() or binary_erosion(region, structure=np.ones((PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD), dtype=bool), border_value=1).any():
                continue
            ring = binary_dilation(region, structure=np.ones((PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD), dtype=bool)) & ~region
            neighbors = labels[ring & fg]
            neighbors = neighbors[neighbors != class_id]
            if neighbors.size:
                labels[region] = np.bincount(neighbors).argmax()
