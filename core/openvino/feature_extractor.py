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
from typing import Union, Tuple, List
from pathlib import Path

import openvino as ov
import openvino_tokenizers

class OVCLIPFeatureExtractor:
    """OpenVINO 推論ヘルパークラス"""

    def __init__(
        self,
        config: yaml
    ):
        # 設定値の初期化
        self.init_value(config)

        # OpenCVの処理に使用するスレッド数
        cv2.setNumThreads(2)

        # OpenVINOのモデルを読み込み
        self.load_ov_model()

        # 初期実行時の遅延を回避
        self._forward_once()

    def init_value(self, config: yaml):
        # config関係
        self.target_size = config["image_size"]
        self.model_dir = config["model_dir"]
        self.use_int8 = config["use_int8"]
        self.device = config["device"].upper()
        self.num_threads = config["ov_num_thread"]

        # 画像の正規化に使用する変数
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    def load_ov_model(self) -> None:
        core = ov.Core()
        core.add_extension(openvino_tokenizers._ext_path)

        # 実行時の設定
        config = {
            "INFERENCE_NUM_THREADS": str(self.num_threads),
            "PERFORMANCE_HINT": "LATENCY", 
            "ENABLE_CPU_PINNING": "YES", 
            "NUM_STREAMS": "1",
        }

        # モデルパス
        model_dir = Path(self.model_dir)
        suffix = "_int8.xml" if self.use_int8 else ".xml"
        visual_path = model_dir / f"pe_visual_model{suffix}"
        text_path = model_dir / f"pe_text_model{suffix}"
        patch_logits_path = model_dir / f"pe_patch_logits_model{suffix}"
        tokenizer_path = model_dir / f"pe_tokenizer.xml"

        visual_model = core.read_model(visual_path)
        text_model = core.read_model(text_path)
        patch_logits_model = core.read_model(patch_logits_path)
        tokenizer_model = core.read_model(tokenizer_path)

        # メタデータ
        self.patch_size: int = visual_model.rt_info["patch_size"].astype(int)

        # 特徴マップのサイズ
        self.feature_size = self.target_size // self.patch_size

        # コンパイルモデル
        self.compiled_visual_model = core.compile_model(visual_model, self.device, config)
        self.compiled_text_model = core.compile_model(text_model, self.device, config)
        self.compiled_patch_logits_model = core.compile_model(patch_logits_model, self.device, config)
        self.compiled_tokenizer = core.compile_model(tokenizer_model, self.device, config)

    def _forward_once(self):
        input_tensor = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        _ = self.compiled_visual_model([input_tensor])

    def extract_image_features(self, image_tensor: np.ndarray, normalize=True):
        outs = self.compiled_visual_model([image_tensor])
        raw_image_features = outs[self.compiled_visual_model.output(0)]
        if normalize:
            raw_image_features = raw_image_features / np.linalg.norm(raw_image_features, axis=-1, keepdims=True)
        raw_patch_features = outs[self.compiled_visual_model.output(1)]
        cls2patch_attention = outs[self.compiled_visual_model.output(2)]
        fused_patch_features = outs[self.compiled_visual_model.output(3)]
        return raw_image_features, raw_patch_features, cls2patch_attention, fused_patch_features

    def extract_text_features(self, prompts: List[str], normalize=True):
        tokenizer_outputs = self.compiled_tokenizer(prompts)
        token_ids = tokenizer_outputs[0]
        text_output = self.compiled_text_model([token_ids])[self.compiled_text_model.output(0)]
        if normalize:
            return text_output / np.linalg.norm(text_output, axis=-1, keepdims=True)
        else:
            return text_output

    def extract_cluster_tokens(self, patch_features: np.ndarray, fused_cluster_labels: np.ndarray):
        outs = self.compiled_patch_logits_model([patch_features, fused_cluster_labels])
        return outs[self.compiled_patch_logits_model.output(0)]