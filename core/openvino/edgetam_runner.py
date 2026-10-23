from core.openvino.runtime_utils import make_compile_config, model_file_suffix  # openvino より先に import する（sys.path の整理）

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import openvino as ov

from core.common.edgetam_helper import EdgeTamHelper
from core.common.params import OPENCV_NUM_THREADS


class OVEdgeTamRunner(EdgeTamHelper):
    """EdgeTAMのOpenVINO backend。推論呼び出しだけを持つ（前後処理は EdgeTamHelper）。"""

    def __init__(self, config: dict):
        super().__init__(config)

        # OpenVINO固有の設定
        self.model_dir = config["model_dir"]
        self.use_int8 = config["et_use_int8"]
        self.device = config["device"].upper()
        self.num_threads = config["ov_num_thread"]

        # OpenCVの処理に使用するスレッド数
        cv2.setNumThreads(OPENCV_NUM_THREADS)

        # OpenVINOのモデルを読み込み
        self.load_ov_model()

        # 初期実行時の遅延を回避
        self._forward_once()

    def load_ov_model(self) -> None:
        core = ov.Core()
        compile_config = make_compile_config(self.num_threads)

        # モデルパス
        model_dir = Path(self.model_dir)
        suffix = model_file_suffix(self.use_int8)
        encoder_path = model_dir / f"edgetam_encoder{suffix}"
        decoder_path = model_dir / f"edgetam_decoder{suffix}"

        encoder_model = core.read_model(encoder_path)
        decoder_model = core.read_model(decoder_path)

        # コンパイルモデル
        self.compiled_encoder_model = core.compile_model(encoder_model, self.device, compile_config)
        self.compiled_decoder_model = core.compile_model(decoder_model, self.device, compile_config)

    def _forward_once(self) -> None:
        """ダミー入力でエンコーダとデコーダを1回ずつ実行する（初回実行の遅延を起動時に済ませる）。"""
        dummy_image = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        high_res_feats0, high_res_feats1, image_embed = self.encode(dummy_image)

        num_dummy_points = 1
        _, _, h0, w0 = high_res_feats0.shape
        self.decode(
            high_res_feats0,
            high_res_feats1,
            image_embed,
            point_coords=np.zeros((1, num_dummy_points, 2), dtype=np.float32),
            point_labels=np.ones((1, num_dummy_points), dtype=np.int32),
            mask_input=np.zeros((1, 1, h0, w0), dtype=np.float32),
            has_mask=np.zeros((1,), dtype=np.float32),
        )

    def encode(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(1,3,S,S) -> (high_res_feats0, high_res_feats1, image_embed)"""
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """プロンプト（点・マスク）とエンコーダの出力 -> (low_res_masks, iou_predictions)"""
        input_data = [
            high_res_feats0,
            high_res_feats1,
            image_embed,
            point_coords,
            point_labels,
            mask_input,
            has_mask,
        ]
        outs = self.compiled_decoder_model(input_data)
        low_res_masks = outs[self.compiled_decoder_model.output(0)]
        iou_predictions = outs[self.compiled_decoder_model.output(1)]
        return low_res_masks, iou_predictions
