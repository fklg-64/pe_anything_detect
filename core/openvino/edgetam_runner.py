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
from typing import Union, Tuple, List
from pathlib import Path

import openvino as ov

class OVEdgeTamRunner:
    """OpenVINOヘルパークラス"""

    def __init__(
        self,
        model_dir: Union[str, Path] = "./weights/openvino_models",
        target_size: int = 512, 
        use_int8: bool = False,
        device: str = "CPU",
        num_threads: int = 0
    ):
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