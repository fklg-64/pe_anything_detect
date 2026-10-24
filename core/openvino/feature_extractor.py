from core.openvino.runtime_utils import limit_blas_threads, make_compile_config, model_file_suffix  # openvino より先に import する（sys.path の整理）

from pathlib import Path
from typing import List

import cv2
import numpy as np

import openvino as ov
import openvino_tokenizers

from core.common.params import OPENCV_NUM_THREADS


class OVCLIPFeatureExtractor:
    """CLIP（PE）のOpenVINO backend。画像特徴量・テキスト特徴量・クラスタごとのトークンを求める。"""

    def __init__(self, config: dict):
        self.target_size = config["image_size"]
        self.model_dir = config["model_dir"]
        self.use_int8 = config["pe_use_int8"]
        self.device = config["device"].upper()
        self.num_threads = config["ov_num_thread"]

        # 画像の正規化に使用する変数
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

        # OpenCVの処理に使用するスレッド数
        cv2.setNumThreads(OPENCV_NUM_THREADS)

        # NumPyのBLASのスレッド数（OpenVINOのスレッドと競合して推論が遅くなるのを避ける。runtime_utils 参照）
        limit_blas_threads()

        # OpenVINOのモデルを読み込み
        self.load_ov_model()

        # 初期実行時の遅延を回避
        self._forward_once()

    def load_ov_model(self) -> None:
        core = ov.Core()
        core.add_extension(openvino_tokenizers._ext_path)
        compile_config = make_compile_config(self.num_threads)

        # モデルパス
        model_dir = Path(self.model_dir)
        suffix = model_file_suffix(self.use_int8)
        visual_path = model_dir / f"pe_visual_model{suffix}"
        text_path = model_dir / f"pe_text_model{suffix}"
        patch_logits_path = model_dir / f"pe_patch_logits_model{suffix}"
        tokenizer_path = model_dir / "pe_tokenizer.xml"

        visual_model = core.read_model(visual_path)
        # 未使用の出力は計算・転送が無駄なので、コンパイル前にグラフから外す
        for result in visual_model.get_results():
            if result.output(0).get_names() & {"cls2patch_attention"}:
                visual_model.remove_result(result)
        text_model = core.read_model(text_path)
        patch_logits_model = core.read_model(patch_logits_path)
        tokenizer_model = core.read_model(tokenizer_path)

        # メタデータ
        self.patch_size: int = visual_model.rt_info["patch_size"].astype(int)

        # 特徴マップのサイズ
        self.feature_size = self.target_size // self.patch_size

        # コンパイルモデル
        self.compiled_visual_model = core.compile_model(visual_model, self.device, compile_config)
        self.compiled_text_model = core.compile_model(text_model, self.device, compile_config)
        self.compiled_patch_logits_model = core.compile_model(patch_logits_model, self.device, compile_config)
        self.compiled_tokenizer = core.compile_model(tokenizer_model, self.device, compile_config)

    def _forward_once(self) -> None:
        input_tensor = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        self.compiled_visual_model([input_tensor])

    def extract_image_features(self, image_tensor: np.ndarray, normalize: bool = True):
        """(1,3,S,S) -> (raw_image_features, raw_patch_features, fused_layer_features)"""
        outs = self.compiled_visual_model([image_tensor])
        raw_image_features = outs[self.compiled_visual_model.output("raw_image_features")]
        if normalize:
            raw_image_features = raw_image_features / np.linalg.norm(raw_image_features, axis=-1, keepdims=True)
        raw_patch_features = outs[self.compiled_visual_model.output("raw_patch_features")]
        fused_patch_features = outs[self.compiled_visual_model.output("fused_layer_features")]
        return raw_image_features, raw_patch_features, fused_patch_features

    def extract_text_features(self, prompts: List[str], normalize: bool = True) -> np.ndarray:
        """プロンプトの文字列 -> (N, D) のテキスト特徴量"""
        tokenizer_outputs = self.compiled_tokenizer(prompts)
        token_ids = tokenizer_outputs[0]
        text_output = self.compiled_text_model([token_ids])[self.compiled_text_model.output(0)]
        if normalize:
            return text_output / np.linalg.norm(text_output, axis=-1, keepdims=True)
        return text_output

    def extract_cluster_tokens(self, patch_features: np.ndarray, fused_cluster_labels: np.ndarray) -> np.ndarray:
        """パッチ特徴 (1,N+1,C) とクラスタ番号 (N,) -> クラスタごとの代表トークン"""
        outs = self.compiled_patch_logits_model([patch_features, fused_cluster_labels])
        return outs[self.compiled_patch_logits_model.output(0)]
