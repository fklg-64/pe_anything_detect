import os
import sys

from typing import List, Union

import yaml
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
        config: yaml
    ):
        self.init_value(config)

        self.model, self.preprocess = self._setup_model()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

        # パッチサイズ、特徴マップサイズ
        self.patch_size = self.model.visual.trunk.patch_embed.patch_size[0]
        self.feature_size = self.force_image_size // self.patch_size

    def init_value(self, config: yaml):
        self.model_id = config["pe_model_id"]
        self.force_image_size = config["image_size"]
        self.to_openvino = config["to_openvino"]

        # パッチサイズ、特徴マップサイズ
        self.patch_size: Union[int, None] = None
        self.feature_size: Union[int, None] = None

    def _setup_model(self):
        # モデルの読み込み
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_id, force_image_size=self.force_image_size
        )
        model: CustomTextCLIP

        # モンキーパッチの適用
        model.encode_image = types.MethodType(encode_image, model)
        model.visual.forward = types.MethodType(timmmodel_forward, model.visual)
        # OpenVINO化の有無で設定変更（pytorchで処理する場合は、詳細な特徴量も取得）
        if not self.to_openvino:
            for i in range(12):
                model.visual.trunk.blocks[i].attn.forward = types.MethodType(attention_rope_forward, model.visual.trunk.blocks[i].attn)
                model.visual.trunk.blocks[i].forward = types.MethodType(eva_block_forward, model.visual.trunk.blocks[i])
        model.visual.trunk.forward = types.MethodType(eva_forward, model.visual.trunk)
        model.visual.trunk.forward_head = types.MethodType(eva_forward_head, model.visual.trunk)
        model.visual.trunk.forward_features = types.MethodType(eva_forward_features, model.visual.trunk)
        model.visual.trunk.custom_forward = types.MethodType(custom_attentionpoollatent_forward, model.visual.trunk)
        model.visual.trunk.attn_pool.forward = types.MethodType(attentionpoollatent_forward, model.visual.trunk.attn_pool)

        # OpenVINO化の有無で設定変更
        if self.to_openvino:
            setattr(model.visual, "to_openvino", True)
            setattr(model.visual.trunk, "to_openvino", True)
            setattr(
                model.visual.trunk, 
                "store_layers", 
                [8, 9, 10]
            )
        else:
            setattr(model.visual, "to_openvino", False)
            setattr(model.visual.trunk, "to_openvino", False)
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


class TorchCLIPBackend(TorchCLIPFeatureExtractor):
    """CLIPのPyTorch backend。共通ロジックに合わせ、入出力は np.ndarray。

    OpenVINOへ変換したモデルと同じグラフ（to_openvino=True）で動かし、出力の形式・正規化を OVCLIPFeatureExtractor に揃える。
    """

    def __init__(self, config: yaml):
        config = {**config, "to_openvino": True}
        super().__init__(config)
        self.device = config["device"]
        self.model.to(self.device).eval()

        # 画像の正規化に使用する変数（OVCLIPFeatureExtractorと同じ）
        self.target_size = self.force_image_size
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def extract_image_features(self, image_tensor: np.ndarray, normalize: bool = True):
        x = torch.from_numpy(np.ascontiguousarray(image_tensor)).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            raw_image, raw_patch, _, fused = self.model.encode_image(x, normalize=normalize)
        return self._to_numpy(raw_image), self._to_numpy(raw_patch), self._to_numpy(fused)

    def extract_text_features(self, prompts: List[str], normalize: bool = True) -> np.ndarray:
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            return self._to_numpy(self.model.encode_text(tokens, normalize=normalize))

    def extract_cluster_tokens(self, patch_features: np.ndarray, fused_cluster_labels: np.ndarray) -> np.ndarray:
        patch = torch.from_numpy(np.ascontiguousarray(patch_features)).to(device=self.device, dtype=torch.float32)
        labels = torch.from_numpy(np.ascontiguousarray(fused_cluster_labels)).to(device=self.device, dtype=torch.int64)
        with torch.no_grad():
            return self._to_numpy(self.model.visual.trunk.custom_forward(patch, labels, normalize=True))
