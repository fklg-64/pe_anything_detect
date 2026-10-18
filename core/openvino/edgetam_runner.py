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
import yaml
import numpy as np
from typing import Union, Tuple, List, Optional
from pathlib import Path

import openvino as ov

from core.common.edgetam_helper import EdgeTamHelper


class OVEdgeTamRunner(EdgeTamHelper):
    """EdgeTAMのOpenVINO backend。推論呼び出しだけを持つ（前後処理は EdgeTamHelper）。"""

    def __init__(
        self,
        config: yaml
    ):
        super().__init__(config)

        # OpenVINO固有の設定
        self.model_dir = config["model_dir"]
        self.use_int8 = config["et_use_int8"]
        self.device = config["device"].upper()
        self.num_threads = config["ov_num_thread"]

        # OpenCVの処理に使用するスレッド数
        cv2.setNumThreads(2)

        # OpenVINOのモデルを読み込み
        self.load_ov_model()

        # 初期実行時の遅延を回避
        self._forward_once()

    def load_ov_model(self) -> None:
        core = ov.Core()

        # 実行時の設定
        config = {
            "INFERENCE_NUM_THREADS": str(self.num_threads),
            "PERFORMANCE_HINT": "LATENCY", 
            "ENABLE_CPU_PINNING": "YES", 
            # "ENABLE_HYPER_THREADING": "NO",
            "NUM_STREAMS": "1",
        }

        # モデルパス
        model_dir = Path(self.model_dir)
        suffix = "_int8.xml" if self.use_int8 else ".xml"
        encoder_path = model_dir / f"edgetam_encoder{suffix}"
        decoder_path = model_dir / f"edgetam_decoder{suffix}"

        encoder_model = core.read_model(encoder_path)
        decoder_model = core.read_model(decoder_path)

        # コンパイルモデル
        self.compiled_encoder_model = core.compile_model(encoder_model, self.device, config)
        self.compiled_decoder_model = core.compile_model(decoder_model, self.device, config)

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