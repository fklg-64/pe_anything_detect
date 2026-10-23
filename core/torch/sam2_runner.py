"""公式 SAM2.1 の PyTorch backend。

EdgeTAM（edgetam_runner.py）と同じ入出力（np.ndarray）で、SegmentationRunner / SemanticRefiner にそのままつながる。
EdgeTAM と SAM2.1 はデコーダー（prompt encoder + mask decoder）の入出力と特徴量の形状が同じなので、
forward / preprocess / postprocess は EdgeTAM のクラスをそのまま使い、モデルの取得（公式 SAM2.1、パッケージ名 sam2_official）だけを差し替える。
EdgeTAM 側（core/torch/sam2、edgetam_runner.py）は変更していない。

設定（configs/torch_config_sam2.yaml）: sam2_model_path / sam2_model_config / image_size / device
"""
import sys
from pathlib import Path

# core/torch を sys.path に加える。公式 SAM2.1 のコピー core/torch/sam2_official を、トップレベルのパッケージ `sam2_official` として import するため
core_dir = Path(__file__).resolve().parent
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

from typing import Dict, Tuple

import torch
import torch.nn as nn

from core.common.edgetam_helper import EdgeTamHelper
from core.torch.edgetam_runner import DecodeModel, RepVitModel, TorchEdgeTamBackend
from sam2_official.build_sam import build_sam2
from sam2_official.modeling.sam2_base import SAM2Base
from sam2_official.utils.transforms import SAM2Transforms


class Sam2Model:
    """公式 SAM2.1 のモデルの読み込みと保持。(設定, 重み, 入力サイズ, デバイス) ごとにキャッシュする（EdgeTamModel は 1 つだけ）。"""

    _cache: Dict[tuple, Tuple[SAM2Base, SAM2Transforms]] = {}

    @classmethod
    def get_model(cls, config_path: str, checkpoint_path: str, image_size: int, device: str = "cpu") -> Tuple[SAM2Base, SAM2Transforms]:
        key = (config_path, checkpoint_path, image_size, device)
        if key not in cls._cache:
            model: SAM2Base = build_sam2(config_path, checkpoint_path, device=device, image_size=image_size)
            transforms = SAM2Transforms(
                resolution=model.image_size,
                mask_threshold=0.0,
                max_hole_area=0.0,
                max_sprinkle_area=0.0,
            )
            cls._cache[key] = (model, transforms)
        return cls._cache[key]


def _load_sam2(module: nn.Module, config: dict) -> None:
    """EdgeTAM のクラス（RepVitModel / DecodeModel）が読む属性を、公式 SAM2.1 のモデルで埋める。

    `edgetam_model` という属性名は EdgeTAM のクラスと共通（中身は SAM2.1 の SAM2Base）。
    """
    module.checkpoint_path = config["sam2_model_path"]
    module.config_path = config["sam2_model_config"]
    module.device = config["device"]
    module.dtype = getattr(torch, config["dtype"])
    module.edgetam_model, module._transforms = Sam2Model.get_model(
        module.config_path, module.checkpoint_path, config["image_size"], module.device
    )


class Sam2EncoderModel(RepVitModel):
    """SAM2.1 の画像エンコーダー（Hiera + FPN）。forward は RepVitModel と同じ
    （neck(trunk(x)) → conv_s0 / conv_s1 → no_mem_embed の加算 → [high_res0, high_res1, image_embed]）。"""

    def __init__(self, config: dict):
        nn.Module.__init__(self)  # RepVitModel.__init__ は EdgeTAM の設定キーを読むので呼ばない
        _load_sam2(self, config)
        self.trunk = self.edgetam_model.image_encoder.trunk
        self.neck = self.edgetam_model.image_encoder.neck
        self.scalp: int = self.edgetam_model.image_encoder.scalp


class Sam2DecoderModel(DecodeModel):
    """SAM2.1 のデコーダー（prompt encoder + mask decoder）。forward / preprocess / postprocess は DecodeModel と同じ。"""

    def __init__(self, config: dict):
        nn.Module.__init__(self)  # DecodeModel.__init__ は EdgeTAM の設定キーを読むので呼ばない
        _load_sam2(self, config)
        self.mask_threshold: float = self._transforms.mask_threshold
        self.no_mask_embed = self.edgetam_model.sam_prompt_encoder.no_mask_embed


class TorchSam2Backend(TorchEdgeTamBackend):
    """SAM2.1 の PyTorch backend。TorchEdgeTamBackend と同じ口（encode / decode が np.ndarray で入出力。前後処理は EdgeTamHelper）。"""

    def __init__(self, config: dict):
        EdgeTamHelper.__init__(self, config)  # TorchEdgeTamBackend.__init__ は EdgeTAM のモデルを作るので呼ばない
        self.device = config["device"]
        self.edgetam_encoder = Sam2EncoderModel(config)
        self.edgetam_decoder = Sam2DecoderModel(config)
