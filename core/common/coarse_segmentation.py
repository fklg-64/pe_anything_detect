"""CLIP特徴量とKMeansによる、パッチ単位の粗い前景/セマンティック推定。"""

from pathlib import Path
from typing import List, Tuple

import cv2
import faiss
import numpy as np
import yaml
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from core.utils import Timer
from core.common.backend import CLIPBackend
from core.common.mask_utils import softmax
from core.common.params import (
    CLIP_LOGIT_SCALE,
    FAISS_NUM_THREADS,
    KMEANS_SEED,
    PATCH_STRUCTURE,
)


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


class CoarseSegmenter:
    """画像をパッチ格子上の「前景らしさ」と「粗いセマンティックラベル」に分ける。"""

    def __init__(self, extractor: CLIPBackend, config: dict):
        self.extractor = extractor
        self.prompts_dir = config["prompts_dir"]
        self.k_clusters = config["k_clusters"]
        self.niter = config["n_iter"]
        self.foreground_threshold = config["foreground_threshold"]
        self.feature_size: int = extractor.feature_size

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
        N, D = features.shape
        features_contig = (
            np.ascontiguousarray(features, dtype=np.float32)
            if not features.flags["C_CONTIGUOUS"]
            else features
        )
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)
        kmeans = faiss.Kmeans(
            d=D, k=k, niter=niter, spherical=False, verbose=False, seed=KMEANS_SEED
        )
        kmeans.train(features_contig)
        _, cluster_labels = kmeans.index.search(features_contig, 1)
        return cluster_labels.reshape(-1)

    def _classify_cluster(self, cluster_feature: np.ndarray) -> Tuple[float, int]:
        """クラスタの代表トークンを前景/背景プロンプトと照合し、(前景確信度, 最も近いプロンプト番号) を返す。

        プロンプト番号は 前景プロンプト -> 背景プロンプト の通し番号。
        """
        fg_prompts_num = len(self.foreground_prompts)

        # 正規化してテキストとの類似度(logits)を得る
        feature = cluster_feature.reshape(1, -1)
        feature = feature / (np.linalg.norm(feature, axis=-1, keepdims=True) + 1e-8)
        logits = CLIP_LOGIT_SCALE * feature @ self.fgbg_text_features.T
        fg_logits = logits[:, :fg_prompts_num]
        bg_logits = logits[:, fg_prompts_num:]

        # グループスコアは最も近いプロンプトの類似度。プロンプトは「言い換え」ではなく「選択肢」の
        # 列挙なので、平均ではなく最大を使う（プロンプト数を増減しても判定が偏らない）
        fg_score = np.max(fg_logits, axis=-1)
        bg_score = np.max(bg_logits, axis=-1)

        # FG vs BG の2クラスsoftmax
        group_logits = np.concatenate([fg_score[:, None], bg_score[:, None]], axis=-1)
        foreground_confidence = float(softmax(group_logits, axis=-1)[0, 0])

        if foreground_confidence > self.foreground_threshold:
            return foreground_confidence, int(np.argmax(fg_logits, axis=1)[0])
        return foreground_confidence, fg_prompts_num + int(np.argmax(bg_logits, axis=1)[0])

    def generate_foreground_map(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """パッチ特徴をKMeansでクラスタ分けし、クラスタごとの前景確信度と対応プロンプトを求める。

        Returns:
            foreground_map: (feature_size, feature_size) 前景確信度。
            foreground_mask: 確信度が閾値を超えるパッチを1とした2値マスク。
            cluster_labels: パッチごとのクラスタ番号（平坦）。
            cluster_index_list: クラスタごとの対応プロンプト番号。
        """
        patch_tokens = fused_patch_features[0, 1:, :]

        with Timer("    ┌ Kmeansクラスタ処理時間"):
            cluster_labels = self.fit_predict_kmeans(patch_tokens, k=self.k_clusters, niter=self.niter)

        with Timer("    ┌ クラスタごとのCLSトークン抽出処理時間"):
            cluster_tokens: List[np.ndarray] = self.extractor.extract_cluster_tokens(patch_features, cluster_labels)

        foreground_map = np.zeros((self.feature_size * self.feature_size,), dtype=np.float32)
        cluster_index_list: List[int] = []
        for label_idx, cluster_feature in enumerate(cluster_tokens):
            confidence, prompt_index = self._classify_cluster(cluster_feature)
            cluster_index_list.append(prompt_index)
            # クラスタ内の全patchに確信度を付与
            foreground_map[cluster_labels == label_idx] += confidence

        foreground_map = foreground_map.reshape(self.feature_size, self.feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)
        return foreground_map, foreground_mask, cluster_labels, cluster_index_list

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
            semantic_labels: (feature_size, feature_size) のパッチごとの連番ラベル。
        """
        semantic_indices = np.asarray(cluster_index_list)[cluster_labels]
        labels_list, semantic_labels = np.unique(semantic_indices, return_inverse=True)
        return labels_list, semantic_labels.reshape(self.feature_size, self.feature_size)

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

        fg = self._fill_holes(fg, labels, original_fg)
        self._merge_thin_regions(fg, labels)

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
            if not region.any() or binary_erosion(region, structure=PATCH_STRUCTURE, border_value=1).any():
                continue
            ring = binary_dilation(region, structure=PATCH_STRUCTURE) & ~region
            neighbors = labels[ring & fg]
            neighbors = neighbors[neighbors != class_id]
            if neighbors.size:
                labels[region] = np.bincount(neighbors).argmax()
