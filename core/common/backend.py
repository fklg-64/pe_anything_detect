"""共通ロジックが backend（OpenVINO / PyTorch）に求める入出力の契約。

入出力はすべて np.ndarray。PyTorch backend は境界で torch.Tensor <-> np.ndarray を変換する。
"""

from typing import List, Protocol, Tuple

import numpy as np


class CLIPBackend(Protocol):
    target_size: int
    feature_size: int
    patch_size: int
    mean: np.ndarray  # (3, 1, 1)
    std: np.ndarray   # (3, 1, 1)

    def extract_image_features(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(1,3,S,S) float32 -> (raw_patch_features, fused_layer_features)。画像全体の特徴量は使わないので返さない"""
        ...

    def extract_text_features(self, prompts: List[str]) -> np.ndarray:
        """-> (N, D) float32（L2正規化済み）"""
        ...

    def extract_cluster_tokens(self, patch_features: np.ndarray, fused_cluster_labels: np.ndarray) -> np.ndarray:
        """パッチ特徴 (1,N+1,C) とクラスタ番号 (N,) int64 -> クラスタごとのトークン"""
        ...


class EdgeTamBackend(Protocol):
    """前後処理（transform_coords / preprocess / postprocess）は EdgeTamHelper が提供する。"""

    target_size: int
    mean: np.ndarray  # (3, 1, 1)
    std: np.ndarray   # (3, 1, 1)

    def encode(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(1,3,S,S) float32 -> (high_res_feats0, high_res_feats1, image_embed)"""
        ...

    def decode(
        self,
        high_res_feats0: np.ndarray,
        high_res_feats1: np.ndarray,
        image_embed: np.ndarray,
        point_coords: np.ndarray,  # (1,N,2) float32
        point_labels: np.ndarray,  # (1,N) int32
        mask_input: np.ndarray,    # (1,1,h,w) float32
        has_mask: np.ndarray,      # (1,) float32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """-> (low_res_masks, iou_predictions)"""
        ...
