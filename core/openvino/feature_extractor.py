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
from PIL import Image
from typing import Union, Tuple, List
from pathlib import Path

import openvino as ov
import openvino_tokenizers

class OVCLIPFeatureExtractor:
    """OpenVINO 推論ヘルパークラス"""

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
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)

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
        core.add_extension(openvino_tokenizers._ext_path)

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
        self.compiled_visual_model = core.compile_model(visual_model, device, config)
        self.compiled_text_model = core.compile_model(text_model, device, config)
        self.compiled_patch_logits_model = core.compile_model(patch_logits_model, device, config)
        self.compiled_tokenizer = core.compile_model(tokenizer_model, device, config)

    def _forward_once(self):
        input_tensor = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        _ = self.compiled_visual_model([input_tensor])

    def load_image(self, data: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, bool]:
        if isinstance(data, (str, Path)):
            img_cv = cv2.imread(str(data), cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError(f"画像の読み込みに失敗しました: {data}")
            is_bgr = True
        elif isinstance(data, Image.Image):
            img_cv = np.array(data.convert("RGB"))
            is_bgr = False
        elif isinstance(data, np.ndarray):
            img_cv = data
            is_bgr = data.ndim == 3 and data.shape[2] == 3
        else:
            raise TypeError(f"未対応の画像形式です: {type(data)}")
        return img_cv, is_bgr

    def preprocess_image(self, img_cv: np.ndarray, is_bgr: bool) -> np.ndarray:
        if img_cv.ndim == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            is_bgr = True

        resized_cv = cv2.resize(
            img_cv, 
            (self.target_size, self.target_size), 
            interpolation=cv2.INTER_LINEAR
        )

        if is_bgr:
            resized_cv = cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB)

        img_float = resized_cv.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))

        img_chw = (img_chw - self.mean) / self.std

        return np.expand_dims(img_chw, axis=0)

    def extract_image_features(self, image_tensor: np.ndarray, normalize=True):
        outs = self.compiled_visual_model([image_tensor])
        raw_image_features = outs[self.compiled_visual_model.output(0)]
        if normalize:
            raw_image_features = raw_image_features / np.linalg.norm(raw_image_features, axis=-1, keepdims=True)
        raw_patch_features = outs[self.compiled_visual_model.output(1)]
        cls2patch_attention = outs[self.compiled_visual_model.output(2)]
        fused_layer_features = outs[self.compiled_visual_model.output(3)]
        return raw_image_features, raw_patch_features, cls2patch_attention, fused_layer_features

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