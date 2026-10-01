import os
import sys

# /opt/intel/openvino 関連のパスを sys.path から削除
sys.path = [p for p in sys.path if not p.startswith("/opt/intel/openvino")]

# (任意) 環境変数からも削除しておく
if "PYTHONPATH" in os.environ:
    paths = os.environ["PYTHONPATH"].split(os.pathsep)
    new_paths = [p for p in paths if not p.startswith("/opt/intel/openvino")]
    os.environ["PYTHONPATH"] = os.pathsep.join(new_paths)

import cv2
import numpy as np
from typing import Union, Tuple, List, Optional
from pathlib import Path

import openvino as ov


class OVEdgeTamHelper:
    def __init__(self, target_size: int):
        self.mask_threshold = 0.0
        self.target_size = target_size

    def transform_coords(
        self,
        coords: np.ndarray,
        normalize: bool = False,
        orig_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """入力座標 (NumPy配列) の正規化解除を行います。"""
        coords = coords.astype(np.float32).copy()

        if normalize:
            assert orig_hw is not None, "orig_hw must be provided when normalize=True"
            h, w = orig_hw
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.target_size  # unnormalize coords
        return coords

    def transform_boxes(
        self,
        boxes: np.ndarray,
        normalize: bool = False,
        orig_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """バウンディングボックス (NumPy配列) の変換を行います。"""
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(
        self, masks: np.ndarray, orig_hw: Tuple[int, int]
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
                point_coords, normalize=normalize_coords, orig_hw=orig_hw
            )
            labels = point_labels
            if unnorm_coords.ndim == 2:
                unnorm_coords = unnorm_coords[np.newaxis, ...]
                labels = labels[np.newaxis, ...]

        # --- 2. ボックスプロンプトの処理 ---
        if box is not None:
            unnorm_box = self.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw
            )  # B x 2 x 2

        # --- 3. マスクプロンプトの処理 ---
        if mask_logits is not None:
            if mask_logits.ndim == 3:
                mask_input = mask_logits[np.newaxis, :, :, :]
            elif mask_logits.ndim == 1:
                # フラグ等で空マスクが必要な場合のゼロ埋め配列生成
                batch_size = mask_logits.shape[0]
                feature_size = self.target_size // 4
                mask_input = np.zeros(
                    (batch_size, 1, feature_size, feature_size),
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
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """モデル出力の後処理 (オリジナル解像度への拡大および閾値処理) を行います (NumPy版)。"""
        # マスクを元画像の解像度にリサイズ
        masks = self.postprocess_masks(low_res_masks, orig_hw)
        low_res_masks = np.clip(low_res_masks, -32.0, 32.0)

        if not return_logits:
            masks = masks > self.mask_threshold

        # 不要なバッチ次元を削除して float 型へキャスト
        masks = np.squeeze(masks, axis=0).astype(np.float32)
        iou_predictions = np.squeeze(iou_predictions, axis=0).astype(np.float32)
        low_res_masks = np.squeeze(low_res_masks, axis=0)

        return masks, iou_predictions, low_res_masks


class OVEdgeTamRunner(OVEdgeTamHelper):
    """OpenVINOヘルパークラス"""

    def __init__(
        self,
        model_dir: Union[str, Path] = "./weights/openvino_models",
        target_size: int = 512, 
        use_int8: bool = False,
        device: str = "CPU",
        num_threads: int = 0
    ):
        super.__init__(target_size)
        # 画像サイズ関係
        self.target_size: int = target_size

        # 画像の正規化に使用する変数
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        # OpenCVの処理に使用するスレッド数
        cv2.setNumThreads(2)

        # OpenVINOのモデルを読み込み
        self.load_ov_model(model_dir, use_int8, device, num_threads)

        # 初期実行時の遅延を回避
        self._forward_once()

    def load_ov_model(
        self, 
        model_dir: Union[str, Path],
        use_int8: bool = False,
        device: str = "CPU",
        num_threads: int = 0
    ) -> None:
        core = ov.Core()

        # 実行時の設定
        config = {
            "INFERENCE_NUM_THREADS": str(num_threads),
            "PERFORMANCE_HINT": "LATENCY", 
            "ENABLE_CPU_PINNING": "YES", 
            "NUM_STREAMS": "1",
        }

        # モデルパス
        model_dir = Path(model_dir)
        suffix = "_int8.xml" if use_int8 else ".xml"
        encoder_path = model_dir / f"edgetam_encoder{suffix}"
        decoder_path = model_dir / f"edgetam_decoder{suffix}"

        encoder_model = core.read_model(encoder_path)
        decoder_model = core.read_model(decoder_path)

        # コンパイルモデル
        self.compiled_encoder_model = core.compile_model(encoder_model, device, config)
        self.compiled_decoder_model = core.compile_model(decoder_model, device, config)

    def _forward_once(self):
        # エンコーダモデル
        input_tensor = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        outs = self.compiled_encoder_model([input_tensor])
        high_res_feats0 = outs[self.compiled_encoder_model.output(0)]
        high_res_feats1 = outs[self.compiled_encoder_model.output(1)]
        image_embed = outs[self.compiled_encoder_model.output(2)]
        # デコーダモデル
        num_dummy_points = 1
        point_coords = np.zeros((1, num_dummy_points, 2), dtype=np.float32)
        point_labels = np.ones((1, num_dummy_points), dtype=np.int32)
        _, _, h0, w0 = high_res_feats0.shape
        mask_input = np.zeros((1, 1, h0, w0), dtype=np.float32)
        has_mask = np.zeros((1), dtype=np.float32)
        input_data = [
            high_res_feats0, 
            high_res_feats1, 
            image_embed, 
            point_coords, 
            point_labels, 
            mask_input, 
            has_mask
        ]
        _, _ = self.compiled_decoder_model(input_data)

    def encode(self, image_tensor: np.ndarray):
        outs = self.compiled_encoder_model([image_tensor])
        high_res_feats0 = outs[self.compiled_encoder_model.output(0)]
        high_res_feats1 = outs[self.compiled_encoder_model.output(1)]
        image_embed = outs[self.compiled_encoder_model.output(2)]
        return high_res_feats0, high_res_feats1, image_embed

    def decode(
        self, 
        high_res_feats0: np.ndarray,
        high_res_feats1: np.ndarray,
        image_embed: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: np.ndarray,
        has_mask: np.ndarray,
    ):
        input_data = [
            high_res_feats0, 
            high_res_feats1, 
            image_embed, 
            point_coords, 
            point_labels, 
            mask_input, 
            has_mask
        ]
        outs = self.compiled_decoder_model(input_data)
        low_res_masks = outs[self.compiled_decoder_model.output(0)]
        iou_predictions = outs[self.compiled_decoder_model.output(1)]
        return low_res_masks, iou_predictions