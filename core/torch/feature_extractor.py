import os
import sys

# /opt/intel/openvino 関連のパスを sys.path から削除
sys.path = [p for p in sys.path if not p.startswith("/opt/intel/openvino")]

# (任意) 環境変数からも削除しておく
if "PYTHONPATH" in os.environ:
    paths = os.environ["PYTHONPATH"].split(os.pathsep)
    new_paths = [p for p in paths if not p.startswith("/opt/intel/openvino")]
    os.environ["PYTHONPATH"] = os.pathsep.join(new_paths)

import types
import numpy as np

import torch
import open_clip
from open_clip.model import CustomTextCLIP

from core.torch.monkey_patch import *

class TorchCLIPFeatureExtractor:
    """CLIPモデルの初期化、モンキーパッチ適用、および特徴量抽出を担当するクラス"""

    def __init__(
        self,
        model_id: str = "hf-hub:timm/PE-Core-T-16-384",
        force_image_size: int = 512,
        to_openvino: bool = False
    ):
        self.model_id = model_id
        self.force_image_size = force_image_size
        self.to_openvino = to_openvino

        self.model, self.preprocess = self._setup_model()

        self.patch_size = self.model.visual.trunk.patch_embed.patch_size[0]
        self.feature_size = force_image_size // self.patch_size

        self.tokenizer = open_clip.get_tokenizer(model_id)

    def _setup_model(self):
        # モデルの読み込み
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_id, force_image_size=self.force_image_size
        )
        model: CustomTextCLIP

        # モンキーパッチの適用
        model.encode_image = types.MethodType(encode_image, model)
        model.visual.forward = types.MethodType(timmmodel_forward, model.visual)
        model.visual.trunk.forward = types.MethodType(eva_forward, model.visual.trunk)
        model.visual.trunk.forward_head = types.MethodType(eva_forward_head, model.visual.trunk)
        model.visual.trunk.forward_features = types.MethodType(eva_forward_features, model.visual.trunk)
        model.visual.trunk.custom_forward = types.MethodType(custom_attentionpoollatent_forward, model.visual.trunk)
        model.visual.trunk.attn_pool.forward = types.MethodType(attentionpoollatent_forward, model.visual.trunk.attn_pool)

        if self.to_openvino:
            setattr(model.visual, "to_openvino", True)
            setattr(
                model.visual.trunk, 
                "store_layers", 
                [3, 6, 9, 12]
            )
        else:
            setattr(model.visual, "to_openvino", False)
            setattr(
                model.visual.trunk, 
                "store_layers", 
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            )

        return model, preprocess

    def extract_image_features(self, image_tensor: torch.Tensor):
        with torch.no_grad():
            return self.model.encode_image(image_tensor, normalize=True)

    def extract_text_features(self, text_tokens: torch.Tensor):
        with torch.no_grad():
            return self.model.encode_text(text_tokens, normalize=True)

    def extract_cluster_tokens(
        self, 
        patch_features: torch.Tensor, 
        cluster_labels: np.ndarray
    ):
        with torch.no_grad():
            return self.model.visual.trunk.custom_forward(
                patch_features, cluster_labels, normalize=True
            )