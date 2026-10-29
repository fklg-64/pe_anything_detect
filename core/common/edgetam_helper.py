from typing import Optional, Tuple

import cv2
import numpy as np

from core.common.params import EDGETAM_LOGIT_CLIP, EDGETAM_MEAN, EDGETAM_STD, MASK_LOGIT_THRESHOLD


class EdgeTamHelper:
    """EdgeTAMの入出力の前後処理（NumPy）。backend（OpenVINO / PyTorch）に依存しない。"""

    def __init__(self, config: dict):
        self.target_size = config["image_size"]

        # マスク閾値
        self.mask_threshold = MASK_LOGIT_THRESHOLD

        # 画像の正規化に使用する変数
        self.mean = np.array(EDGETAM_MEAN).reshape(3, 1, 1)
        self.std = np.array(EDGETAM_STD).reshape(3, 1, 1)

    def transform_coords(
        self,
        coords: np.ndarray,
        normalize: bool = False,
        orig_hw: Optional[Tuple[int, int]] = None,
        input_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """入力座標 (NumPy配列) の正規化解除を行います。

        input_hw: モデルに入力した画像の内容の大きさ (高さ, 幅)。None なら image_size 四方（アスペクト比を維持するモードでは長方形になる）。
        """
        coords = coords.astype(np.float32).copy()

        if normalize:
            assert orig_hw is not None, "orig_hw must be provided when normalize=True"
            h, w = orig_hw
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        input_h, input_w = input_hw if input_hw is not None else (self.target_size, self.target_size)
        coords[..., 0] *= input_w  # unnormalize coords
        coords[..., 1] *= input_h
        return coords

    def transform_boxes(
        self,
        boxes: np.ndarray,
        normalize: bool = False,
        orig_hw: Optional[Tuple[int, int]] = None,
        input_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """バウンディングボックス (NumPy配列) の変換を行います。"""
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw, input_hw)
        return boxes

    def postprocess_masks(
        self,
        masks: np.ndarray,
        orig_hw: Tuple[int, int],
    ) -> np.ndarray:
        """出力マスクを元画像の解像度 (orig_hw) にリサイズします (NumPy/OpenCV版)。"""
        masks = masks.astype(np.float32)
        target_h, target_w = orig_hw

        batch_size, channels = masks.shape[0], masks.shape[1]
        resized_masks = np.zeros(
            (batch_size, channels, target_h, target_w), dtype=np.float32
        )
        for b in range(batch_size):
            for c in range(channels):
                resized_masks[b, c] = cv2.resize(
                    masks[b, c],
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR,
                )
        return resized_masks

    def preprocess(
        self,
        orig_hw: Tuple[int, int],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_logits: Optional[np.ndarray] = None,
        normalize_coords: bool = True,
        input_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """入力プロンプト (ポイント、ボックス、マスク) の前処理と結合を行います (NumPy版)。"""
        unnorm_coords: Optional[np.ndarray] = None
        labels: Optional[np.ndarray] = None
        unnorm_box: Optional[np.ndarray] = None
        mask_input: Optional[np.ndarray] = None

        # --- 1. ポイントプロンプトの処理 ---
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied if point_coords is supplied.")

            unnorm_coords = self.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=orig_hw, input_hw=input_hw
            )
            labels = point_labels
            if unnorm_coords.ndim == 2:
                unnorm_coords = unnorm_coords[np.newaxis, ...]
                labels = labels[np.newaxis, ...]

        # --- 2. ボックスプロンプトの処理 ---
        if box is not None:
            unnorm_box = self.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw, input_hw=input_hw
            )  # B x 2 x 2

        # --- 3. マスクプロンプトの処理 ---
        if mask_logits is not None:
            if mask_logits.ndim == 3:
                mask_input = mask_logits[np.newaxis, :, :, :]
            elif mask_logits.ndim == 1:
                # フラグ等で空マスクが必要な場合のゼロ埋め配列生成
                batch_size = mask_logits.shape[0]
                input_h, input_w = input_hw if input_hw is not None else (self.target_size, self.target_size)
                mask_input = np.zeros(
                    (batch_size, 1, input_h // 4, input_w // 4),
                    dtype=np.float32,
                )
            has_mask = np.ones((1,), dtype=np.float32)
        else:
            has_mask = np.zeros((1,), dtype=np.float32)

        # --- 4. ポイントとボックスの統合 ---
        concat_points: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if unnorm_coords is not None and labels is not None:
            concat_points = (unnorm_coords, labels)

        if unnorm_box is not None:
            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = np.tile(
                np.array([[2, 3]], dtype=np.int32), (unnorm_box.shape[0], 1)
            )

            if concat_points is not None:
                concat_coords = np.concatenate([box_coords, concat_points[0]], axis=1)
                concat_labels = np.concatenate([box_labels, concat_points[1]], axis=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # プロンプトが何もない場合のエッジケース保護（戻り値の要素数を4個に揃えて保護）
        if concat_points is None:
            return mask_input, None, None, has_mask

        return mask_input, concat_points[0], concat_points[1], has_mask

    def postprocess(
        self,
        low_res_masks: np.ndarray,
        iou_predictions: np.ndarray,
        orig_hw: Tuple[int, int],
        need_masks: bool = True,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """モデル出力の後処理 (オリジナル解像度への拡大および閾値処理) を行います (NumPy版)。

        need_masks=False のときは、リサイズ・二値化を省いて masks に None を返します（logits だけ使う呼び出し用）。
        """
        masks: Optional[np.ndarray] = None
        if need_masks:
            # マスクを元画像の解像度にリサイズ
            masks = self.postprocess_masks(low_res_masks, orig_hw)
            masks = masks > self.mask_threshold
            # 不要なバッチ次元を削除して float 型へキャスト
            masks = np.squeeze(masks, axis=0).astype(np.float32)
        low_res_masks = np.clip(low_res_masks, -EDGETAM_LOGIT_CLIP, EDGETAM_LOGIT_CLIP)
        iou_predictions = np.squeeze(iou_predictions, axis=0).astype(np.float32)
        low_res_masks = np.squeeze(low_res_masks, axis=0)

        return masks, iou_predictions, low_res_masks
