from typing import Optional, Tuple

import numpy as np

from core.common.params import EDGETAM_LOGIT_CLIP, EDGETAM_MEAN, EDGETAM_STD


class EdgeTamHelper:
    """EdgeTAMの入出力の前後処理（NumPy）。backend（OpenVINO / PyTorch）に依存しない。"""

    def __init__(self, config: dict):
        self.target_size = config["image_size"]

        # 画像の正規化に使用する変数
        self.mean = np.array(EDGETAM_MEAN).reshape(3, 1, 1)
        self.std = np.array(EDGETAM_STD).reshape(3, 1, 1)

    def transform_coords(
        self,
        coords: np.ndarray,
        normalize: bool = False,
        orig_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """入力座標 (NumPy配列) の正規化解除を行います。正規化した座標を、モデルに入力した画像 (image_size 四方) の画素座標にします。"""
        coords = coords.astype(np.float32).copy()

        if normalize:
            assert orig_hw is not None, "orig_hw must be provided when normalize=True"
            h, w = orig_hw
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords[..., 0] *= self.target_size  # unnormalize coords
        coords[..., 1] *= self.target_size
        return coords

    def preprocess(
        self,
        orig_hw: Tuple[int, int],
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_logits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """デコーダーの入力（点とマスクのプロンプト）を作ります (NumPy版)。

        orig_hw: 点の座標を表す画像の大きさ (高さ, 幅)。point_coords (N, 2) は、その画像の画素座標 (x, y)。
        mask_logits: 初期マスクのロジット (1, h, w)。

        Returns: (mask_input (1, 1, h, w), point_coords (1, N, 2)（モデル入力の画素座標）, point_labels (1, N), has_mask (1,))
        """
        unnorm_coords = self.transform_coords(point_coords, normalize=True, orig_hw=orig_hw)[np.newaxis, ...]
        labels = point_labels[np.newaxis, ...]
        mask_input = mask_logits[np.newaxis, :, :, :]
        has_mask = np.ones((1,), dtype=np.float32)
        return mask_input, unnorm_coords, labels, has_mask

    def postprocess(
        self,
        low_res_masks: np.ndarray,
        iou_predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """モデル出力の後処理を行います (NumPy版)。(予測 IoU (マスク数,)、低解像度ロジット (マスク数, h, w)) を返します。

        ロジットは絶対値 EDGETAM_LOGIT_CLIP 以下に収めます。マスクの二値化・拡大は呼び出し側（detail/refiner.py）が必要な分だけ行います。
        """
        low_res_masks = np.clip(low_res_masks, -EDGETAM_LOGIT_CLIP, EDGETAM_LOGIT_CLIP)
        iou_predictions = np.squeeze(iou_predictions, axis=0).astype(np.float32)
        low_res_masks = np.squeeze(low_res_masks, axis=0)
        return iou_predictions, low_res_masks
