import os
import sys

def _cleanup_openvino_paths():
    """/opt/intel/openvino 関連のパスを sys.path および PYTHONPATH から除去"""
    sys.path = [p for p in sys.path if not p.startswith("/opt/intel/openvino")]
    if "PYTHONPATH" in os.environ:
        paths = os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [p for p in paths if not p.startswith("/opt/intel/openvino")]
        )
        
_cleanup_openvino_paths()

from pathlib import Path
from typing import Callable, List, Tuple, Union

import nncf
import openvino as ov
from openvino_tokenizers import convert_tokenizer
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from timm.models.eva import Eva

from config import *
from core.utils import load_config
from core.torch.edgetam_runner import DecodeModel, EdgeTAMPredictor, RepVitModel
from core.torch.feature_extractor import TorchCLIPFeatureExtractor

class BaseOpenVINOExporter:
    """OpenVINO モデルの変換・保存・量子化を行う共通基底クラス"""

    def __init__(self, save_dir: Union[str, Path] = "./weights/openvino"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_and_compress(self, ov_model: ov.Model, model_name: str):
        """FP32モデルの保存およびINT8量子化/圧縮モデルの保存"""
        # FP32 保存
        fp32_path = self.save_dir / f"{model_name}.xml"
        ov.save_model(ov_model, fp32_path)
        print(f"  [OK] Saved FP32: {fp32_path.name}")

        # INT8 量子化/圧縮
        compressed_model = nncf.compress_weights(
            ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM
        )

        # INT8 保存
        int8_path = self.save_dir / f"{model_name}_int8.xml"
        ov.save_model(compressed_model, int8_path)
        print(f"  [OK] Saved INT8: {int8_path.name}")

        # 量子化モデルのINT8使用率を表示
        self._visualize_quantization_stats(compressed_model)

    def _visualize_quantization_stats(self, ov_model: ov.Model):
        """量子化モデル内の全パラメータに対するINT8使用率を算出・可視化"""
        total_params = 0
        int8_params = 0

        for op in ov_model.get_ops():
            if op.get_type_name() == "Constant":
                shape = op.get_shape()
                num_elements = 1
                for dim in shape:
                    num_elements *= dim

                total_params += num_elements
                if op.get_element_type() in (ov.Type.i8, ov.Type.u8):
                    int8_params += num_elements

        int8_ratio = (int8_params / total_params * 100) if total_params > 0 else 0.0
        bar_length = 30
        int8_bar_len = max(1 if int8_ratio > 0 else 0, int(bar_length * (int8_ratio / 100)))

        print("\n  ================ 量子化モデルのINT8使用率 (Stats) ================")
        print(f"  Total Parameters : {total_params:,}")
        print(f"  INT8 Parameters  : {int8_params:,}")
        print(f"  INT8 Usage Ratio : {int8_ratio:.2f}%")
        print("  -----------------------------------------------------------")
        print(f"  INT8 Ratio : [{'█' * int8_bar_len}{' ' * (bar_length - int8_bar_len)}] {int8_ratio:.2f}%")
        print("  ===========================================================\n")

    def _run_export_pipeline(self, tasks: List[Tuple[str, Callable]]):
        """登録されたエクスポートタスクを順次実行"""
        print("=" * 60)
        print(f" Starting OpenVINO Model Export -> {self.save_dir.resolve()}")
        print("=" * 60)

        total = len(tasks)
        for i, (description, task_func) in enumerate(tasks, 1):
            print(f"\n[{i}/{total}] {description}...")
            task_func()

        print("\n" + "=" * 60)
        print(" All models exported successfully!")
        print("=" * 60)


class CLIPOpenVINOExporter(BaseOpenVINOExporter):
    """TorchCLIPFeatureExtractor の各コンポーネントを OpenVINO 形式に変換・保存するクラス"""

    def __init__(self, extractor: TorchCLIPFeatureExtractor, save_dir: str = "./weights/openvino"):
        super().__init__(save_dir)
        self.extractor = extractor

    def export_visual_model(self):
        """画像エンコーダ (Visual Model) の OpenVINO 変換"""
        image_size = self.extractor.force_image_size
        dummy_input = torch.randn(1, 3, image_size, image_size)

        ov_visual = ov.convert_model(
            self.extractor.model.visual,
            example_input=dummy_input
        )
        ov_visual.reshape({0: ov.PartialShape([1, 3, -1, -1])})

        custom_output_names = [
            "raw_image_features",
            "raw_patch_features",
            "cls2patch_attention",
            "fused_layer_features",
        ]
        for i, name in enumerate(custom_output_names):
            if i < len(ov_visual.outputs):
                ov_visual.output(i).tensor.set_names({name})

        ov_visual.rt_info["patch_size"] = str(
            self.extractor.model.visual.trunk.patch_embed.patch_size[0]
        )
        self._save_and_compress(ov_visual, "pe_visual_model")

    def export_text_model(self):
        """テキストエンコーダ (Text Model) の OpenVINO 変換"""
        context_length = self.extractor.model.context_length
        dummy_text_input = torch.zeros((1, context_length), dtype=torch.long)

        ov_text = ov.convert_model(
            self.extractor.model.text,
            example_input=dummy_text_input,
        )
        ov_text.reshape({0: ov.PartialShape([-1, context_length])})
        self._save_and_compress(ov_text, "pe_text_model")

    def export_custom_model(self):
        """パッチ特徴量抽出モデル (Patch Logits Model) の OpenVINO 変換"""
        class CustomWrapper(nn.Module):
            def __init__(self, model: Eva):
                super().__init__()
                self.model: Eva = model

            def forward(self, patch_features: torch.Tensor, cluster_labels: torch.Tensor):
                return self.model.custom_forward(
                    patch_features, cluster_labels, normalize=True
                )

        w, h = self.extractor.force_image_size, self.extractor.force_image_size
        pw, ph = w // self.extractor.patch_size, h // self.extractor.patch_size
        patch_num = pw * ph
        embed_dim = self.extractor.model.visual.trunk.embed_dim

        dummy_patch_features = torch.randn(1, patch_num + 1, embed_dim)
        dummy_cluster_labels = torch.randn(patch_num)
        custom_wrapper = CustomWrapper(self.extractor.model.visual.trunk).eval()

        ov_custom = ov.convert_model(
            custom_wrapper,
            example_input=(dummy_patch_features, dummy_cluster_labels)
        )
        self._save_and_compress(ov_custom, "pe_patch_logits_model")

    def export_tokenizer_model(self):
        """トークナイザーの OpenVINO 変換"""
        hf_tokenizer = AutoTokenizer.from_pretrained("timm/PE-Core-T-16-384")
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_tokenizer.pad_token_id = 0
        max_length = self.extractor.tokenizer.context_length

        ov_tokenizer = convert_tokenizer(
            hf_tokenizer, max_length=max_length, use_max_padding=True, with_detokenizer=False
        )
        ov.save_model(ov_tokenizer, self.save_dir / "pe_tokenizer.xml")

    def export_all(self):
        """全 CLIP 関連モデルの変換を実行"""
        tasks = [
            ("Exporting Visual Model", self.export_visual_model),
            ("Exporting Text Model", self.export_text_model),
            ("Exporting Patch Logits Model", self.export_custom_model),
            ("Exporting Tokenizer", self.export_tokenizer_model),
        ]
        self._run_export_pipeline(tasks)


class EdgeTAMOpenVINOExporter(BaseOpenVINOExporter):
    """EdgeTAM モデル（Encoder / Decoder）を OpenVINO 形式に変換・保存するクラス"""

    def __init__(
        self,
        edgetam_encoder: RepVitModel,
        edgetam_decoder: DecodeModel,
        save_dir: str = "./weights/openvino",
        device: str = "cpu",
    ):
        super().__init__(save_dir)
        self.edgetam_encoder = edgetam_encoder
        self.edgetam_decoder = edgetam_decoder
        self.device = device

    def export_encoder_model(self):
        """画像エンコーダ (Visual Model) の OpenVINO 変換"""
        image_size = self.edgetam_encoder.edgetam_model.image_size
        dummy_input = torch.randn(1, 3, image_size, image_size, device=self.device)

        ov_encoder = ov.convert_model(self.edgetam_encoder, example_input=dummy_input)
        ov_encoder.reshape({0: ov.PartialShape([-1, 3, -1, -1])})

        self._save_and_compress(ov_encoder, "edgetam_encoder")

    def export_decoder_model(self):
        """画像デコーダ (Prompt & Mask Decoder) の OpenVINO 変換"""
        image_size = self.edgetam_encoder.edgetam_model.image_size

        # 1. エンコーダーを実行して特徴量を取得
        dummy_image = torch.randn(1, 3, image_size, image_size, device=self.device)
        with torch.no_grad():
            high_res_feats0, high_res_feats1, image_embed = self.edgetam_encoder(dummy_image)

        # 2. プロンプト用ダミー入力
        num_dummy_points = 1
        point_coords = torch.zeros((1, num_dummy_points, 2), dtype=torch.float32, device=self.device)
        point_labels = torch.ones((1, num_dummy_points), dtype=torch.int32, device=self.device)

        _, _, h0, w0 = high_res_feats0.shape
        mask_input = torch.zeros((1, 1, h0, w0), dtype=torch.float32, device=self.device)
        has_mask = torch.tensor([0.0], dtype=torch.float32, device=self.device)

        example_inputs = (
            high_res_feats0,
            high_res_feats1,
            image_embed,
            point_coords,
            point_labels,
            mask_input,
            has_mask,
        )

        # 3. OpenVINO 変換
        ov_decoder = ov.convert_model(self.edgetam_decoder, example_input=example_inputs)

        # 4. 動的 Shape 設定
        c0, c1, c2 = high_res_feats0.shape[1], high_res_feats1.shape[1], image_embed.shape[1]
        ov_decoder.reshape({
            0: ov.PartialShape([-1, c0, -1, -1]),   # high_res_feats0: [B, C0, H/4, W/4]
            1: ov.PartialShape([-1, c1, -1, -1]),   # high_res_feats1: [B, C1, H/8, W/8]
            2: ov.PartialShape([-1, c2, -1, -1]),   # image_embed:     [B, C2, H/16, W/16]
            3: ov.PartialShape([-1, -1, 2]),        # point_coords:    [B, N_points, 2]
            4: ov.PartialShape([-1, -1]),           # point_labels:    [B, N_points]
            5: ov.PartialShape([-1, 1, -1, -1]),    # mask_input:      [B, 1, H/4, W/4]
            6: ov.PartialShape([-1]),               # has_mask:        [B]
        })

        self._save_and_compress(ov_decoder, "edgetam_decoder")

    def export_all(self):
        """全 EdgeTAM モデルの変換を実行"""
        tasks = [
            ("Exporting Encoder Model", self.export_encoder_model),
            ("Exporting Decoder Model", self.export_decoder_model),
        ]
        self._run_export_pipeline(tasks)


if __name__ == "__main__":
    # 設定データ
    config = load_config("./configs/torch_config.yaml")
    # OpenVINOモデルに変換するモードに強制変更
    config["to_openvino"] = True

    # Extractor / EdgeTAM モデルの初期化
    extractor = TorchCLIPFeatureExtractor(config)
    edgetam_encoder = RepVitModel(config)
    edgetam_decoder = DecodeModel(config)

    # CLIP モデルのエクスポート
    exporter_pe = CLIPOpenVINOExporter(extractor, save_dir=config["ov_model_dir"])
    exporter_pe.export_all()

    # EdgeTAM モデルのエクスポート
    exporter_edgetam = EdgeTAMOpenVINOExporter(edgetam_encoder, edgetam_decoder, save_dir=config["ov_model_dir"])
    exporter_edgetam.export_all()