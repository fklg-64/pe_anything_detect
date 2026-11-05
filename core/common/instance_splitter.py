"""セマンティックマップの同じクラスの領域を、物体 1 個ずつ（インスタンス）に分ける。

semantic_map は同じ種類の物体が接していると 1 つの塊になる。そこで、クラスごとの前景領域の中を、
EdgeTAM のデコーダー（点を 1 つ指したときに、どこまでが同じ物体か）を手がかりに分ける。分け方は 2 つ（instance_method）。

proposals: 領域に点を置いて 1 点ずつデコードし、領域に収まり・新しい面積があり・既に採った物体と重ならない提案を、確信の高い順に採る
           （SAM の自動マスク生成と同じ形）。接触した同種の物体で最も安定して分けられる。
merge:     距離変換の watershed で過分割し、隣り合う断片の一方を指したマスクがもう一方を覆い（互いに）、かつ境界にくびれが無いときだけ結合する。
           離れた物体・1 個の物体を過分割しにくい。

意味的な前景（semantic_map > 0）は変えない。PE の特徴は使わない（同じ種類の物体は同じ特徴になり、区別する手がかりが無い）。
検証は docs/instance_segmentation.md。
"""
from typing import List, Tuple

import cv2
import numpy as np

from core.common.edgetam_session import Features
from core.common.mask_utils import (
    fragment_adjacency,
    fragment_stats,
    peak_point,
    select_background_points,
    split_by_distance,
)
from core.common.params import (
    INSTANCE_EDT_SCALE,
    INSTANCE_MERGE_COVER,
    INSTANCE_MERGE_NECK,
    INSTANCE_MIN_AREA_RATIO,
    INSTANCE_OVERSEG_H_FRAC,
    INSTANCE_PROPOSAL_INSIDE,
    INSTANCE_PROPOSAL_NEW_FRACTION,
    INSTANCE_PROPOSAL_NMS_IOU,
)
from core.common.region_estimator import RegionEstimator

METHODS = ("proposals", "merge")


class InstanceSplitter:
    def __init__(self, estimator: RegionEstimator, config: dict):
        """estimator: 精緻化と共通の RegionEstimator（EdgeTAM の口・パッチ格子・点の数・デコードの反復回数・座標変換を借りる）。"""
        self.estimator = estimator
        self.session = estimator.session
        self.method = config["instance_method"]
        if self.method not in METHODS:
            raise ValueError(f"instance_method は {METHODS} のどれか: {self.method}")

    # =========================================================================
    # メインの流れ
    # =========================================================================

    def split(self, semantic_map: np.ndarray, feats: Features) -> Tuple[np.ndarray, List[int]]:
        """semantic_map（0 = 背景、1~ = クラス）を、インスタンスの ID 画像に分ける。

        feats: EdgeTAM エンコーダの出力。
        Returns: (instance_map, instance_classes)。instance_map は (H, W) int32（0 = 背景、1~ = インスタンス）。
                 instance_classes[k - 1] = インスタンス k のクラス（semantic_map の値）。
        """
        height, width = semantic_map.shape
        background = self._background_points(semantic_map)
        pieces: List[Tuple[np.ndarray, int]] = []
        for class_id in np.unique(semantic_map):
            if not class_id:
                continue
            region = semantic_map == class_id
            if self.method == "proposals":
                parts = self._proposals(region, feats, background, width, height)
            else:
                parts = self._merge(region, feats, background, width, height)
            pieces += [(part, int(class_id)) for part in parts]
        return self._assemble(semantic_map.shape, pieces)

    @staticmethod
    def _assemble(shape: Tuple[int, int], pieces: List[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, List[int]]:
        """領域のリストを、ID 画像にする。面積が小さすぎる領域は捨てる。"""
        min_area = int(INSTANCE_MIN_AREA_RATIO * shape[0] * shape[1])
        instance_map = np.zeros(shape, np.int32)
        classes: List[int] = []
        for part, class_id in pieces:
            free = part & (instance_map == 0)
            if free.sum() >= max(min_area, 1):
                classes.append(class_id)
                instance_map[free] = len(classes)
        return instance_map, classes

    # =========================================================================
    # EdgeTAM のデコーダー
    # =========================================================================

    def _background_points(self, semantic_map: np.ndarray) -> List[List[float]]:
        """背景の負例点（画像座標）。意味的な前景から離れた、画像の端につながる背景から選ぶ。"""
        height, width = semantic_map.shape
        grid_h, grid_w = self.estimator.grid
        foreground = cv2.resize((semantic_map > 0).astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA) > 0.02
        points = select_background_points(foreground, self.estimator.background_point_count, None)
        return self.estimator.transform_coords(points.tolist(), width, height)

    def _decode(self, feats: Features, positives, background, width: int, height: int) -> Tuple[np.ndarray, float]:
        """正例点と背景の負例点で EdgeTAM のデコーダーを反復し、(低解像度のロジット, IoU 予測) を返す。最初の反復の初期マスクは、ロジット 0 の空のマスク。"""
        coords = [list(map(float, p)) for p in positives] + [list(map(float, p)) for p in background]
        labels = [1] * len(positives) + [0] * len(background)
        high_res_w, high_res_h = self.estimator.high_res_size
        mask_input = np.zeros((1, high_res_h, high_res_w), np.float32)
        logit, score = None, 0.0
        for _ in range(self.estimator.refine_iterations):
            scores, logits = self.session.decode_prompts(coords, labels, feats, height, width, mask_input)
            best = int(np.argmax(scores))
            mask_input = logits[best:best + 1]
            logit, score = logits[best], float(scores[best])
        return logit, score

    def _low_res(self, region: np.ndarray) -> np.ndarray:
        """領域を、デコーダーのロジットと同じ大きさに縮めたマスク。"""
        return cv2.resize(region.astype(np.float32), self.estimator.high_res_size, interpolation=cv2.INTER_AREA) >= 0.5

    # =========================================================================
    # 提案方式
    # =========================================================================

    def _proposals(self, region, feats, background, width, height) -> List[np.ndarray]:
        low_region = self._low_res(region)
        if not low_region.any():
            return []
        # 提案の点 = 過分割した断片の最内点
        labels = split_by_distance(region, INSTANCE_OVERSEG_H_FRAC, INSTANCE_EDT_SCALE)
        candidates = []
        for value in np.unique(labels):
            if not value:
                continue
            logit, score = self._decode(feats, [peak_point(labels == value)], background, width, height)
            mask = logit > 0
            area = int(mask.sum())
            inside = float((mask & low_region).sum() / area) if area else 0.0
            if area and inside >= INSTANCE_PROPOSAL_INSIDE:
                candidates.append((score * inside, logit, mask & low_region))
        candidates.sort(key=lambda c: -c[0])

        # 確信の高い順に、領域に新しい面積を足し、既に採ったものと重ならないものを採る
        chosen = []
        covered = np.zeros_like(low_region)
        for _, logit, mask in candidates:
            area = int(mask.sum())
            if area == 0 or int((mask & ~covered).sum()) < INSTANCE_PROPOSAL_NEW_FRACTION * area:
                continue
            if any((mask & m).sum() / max((mask | m).sum(), 1) >= INSTANCE_PROPOSAL_NMS_IOU for _, m in chosen):
                continue
            chosen.append((logit, mask))
            covered |= mask
        if not chosen:
            return []

        # 領域の各画素を、ロジットが最大の物体に割り当てる
        stack = np.stack([cv2.resize(l.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) for l, _ in chosen])
        best = stack.argmax(0)
        return [region & (best == k) for k in range(len(chosen))]

    # =========================================================================
    # 結合方式
    # =========================================================================

    def _merge(self, region, feats, background, width, height) -> List[np.ndarray]:
        labels = split_by_distance(region, INSTANCE_OVERSEG_H_FRAC, INSTANCE_EDT_SCALE)
        values = [int(v) for v in np.unique(labels) if v]
        if len(values) <= 1:
            return [labels == v for v in values]
        pairs = fragment_adjacency(labels)
        stats = fragment_stats(labels)

        # 断片ごとに、その最内点だけを指したときのマスクが、隣の断片をどれだけ覆うか
        masks = {v: self._decode(feats, [stats[v][0]], background, width, height)[0] > 0 for v in values}
        small = {v: self._low_res(labels == v) for v in values}

        def covered(a: int, b: int) -> float:
            return float((masks[a] & small[b]).sum() / max(small[b].sum(), 1)) if small[b].any() else 0.0

        parent = {v: v for v in values}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), length in pairs.items():
            same_object = min(covered(a, b), covered(b, a)) >= INSTANCE_MERGE_COVER
            no_neck = length / max(min(stats[a][1], stats[b][1]), 1.0) >= INSTANCE_MERGE_NECK
            if same_object and no_neck:
                parent[find(b)] = find(a)
        groups = {}
        for v in values:
            groups.setdefault(find(v), []).append(v)
        return [np.isin(labels, members) for members in groups.values()]
