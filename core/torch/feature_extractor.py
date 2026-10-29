import types
from typing import List

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.model import CustomTextCLIP

from core.common.params import PE_NUM_BLOCKS

# monkey_patch は import した時点で timm 側へのパッチ（動的な画像サイズ対応）が適用される。モデル生成より前に import すること
from core.torch.monkey_patch import (
    attention_rope_forward,
    attentionpoollatent_forward,
    custom_attentionpoollatent_forward,
    encode_image,
    eva_block_forward,
    eva_forward,
    eva_forward_features,
    eva_forward_head,
    timmmodel_forward,
)


class TorchCLIPFeatureExtractor:
    """CLIPモデルの初期化、モンキーパッチ適用、および特徴量抽出を担当するクラス"""

    def __init__(self, config: dict):
        self.model_id = config["pe_model_id"]
        self.force_image_size = config["image_size"]
        self.to_openvino = config["to_openvino"]
        # クラスタリングに使う層（第 N ブロックの出力）。OpenVINO モデルの fused_layer_features に連結して出力される
        self.cluster_layers = list(config.get("cluster_layers", [8, 9, 10]))

        self.model, self.preprocess = self._setup_model()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

        # パッチサイズ、特徴マップサイズ
        self.patch_size: int = self.model.visual.trunk.patch_embed.patch_size[0]
        self.feature_size: int = self.force_image_size // self.patch_size

    def _setup_model(self):
        # モデルの読み込み
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_id, force_image_size=self.force_image_size
        )
        model: CustomTextCLIP
        trunk = model.visual.trunk

        # モンキーパッチの適用
        model.encode_image = types.MethodType(encode_image, model)
        model.visual.forward = types.MethodType(timmmodel_forward, model.visual)
        # OpenVINO化の有無で設定変更（pytorchで処理する場合は、詳細な特徴量も取得）
        if not self.to_openvino:
            for block in trunk.blocks[:PE_NUM_BLOCKS]:
                block.attn.forward = types.MethodType(attention_rope_forward, block.attn)
                block.forward = types.MethodType(eva_block_forward, block)
        trunk.forward = types.MethodType(eva_forward, trunk)
        trunk.forward_head = types.MethodType(eva_forward_head, trunk)
        trunk.forward_features = types.MethodType(eva_forward_features, trunk)
        trunk.custom_forward = types.MethodType(custom_attentionpoollatent_forward, trunk)
        trunk.attn_pool.forward = types.MethodType(attentionpoollatent_forward, trunk.attn_pool)

        # OpenVINO化の有無で設定変更
        model.visual.to_openvino = self.to_openvino
        trunk.to_openvino = self.to_openvino
        # 保持する層。OpenVINO化では cluster_layers だけ、PyTorchの詳細な特徴量では全層
        trunk.store_layers = self.cluster_layers if self.to_openvino else list(range(1, PE_NUM_BLOCKS + 1))

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

    def __init__(self, config: dict):
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


class TorchCLIPProbe(TorchCLIPFeatureExtractor):
    """実験用: 各層の hidden と各ブロックの出力を取り出し、後半ブロックを attention マスク付きで再実行したり、
    任意のトークン列・ラベルで attention pool したりできる。入出力は np.ndarray（CLIPBackend の契約も満たす）。

    全 12 層を生の値で扱うため、OpenVINO 変換用のグラフ（to_openvino=True）ではなく、パッチ済みの非変換経路で動かす。
    """

    def __init__(self, config: dict):
        super().__init__({**config, "to_openvino": False})
        self.device = config["device"]
        self.model.to(self.device).eval()
        self.trunk = self.model.visual.trunk
        self.num_layers = len(self.trunk.blocks)

        # 画像の正規化に使用する変数（OVCLIPFeatureExtractorと同じ）
        self.target_size = self.force_image_size
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def embed(self, image_tensor: np.ndarray):
        """パッチ埋め込み + 位置埋め込み + norm_pre。(hidden0 (1,1+N,C), rope) を返す。"""
        x = self.trunk.patch_embed(self._tensor(image_tensor))
        x, rope = self.trunk._pos_embed(x)
        return self.trunk.norm_pre(x), rope

    @torch.no_grad()
    def run_blocks(self, x: torch.Tensor, rope, start: int = 0, end=None, attn_mask=None, collect: bool = False):
        """x を第 start ブロック（0 起点）から end の手前まで通す。

        attn_mask: (1,1,1+N,1+N) の bool（True=見える）または加算する float。
        返り値: 各ブロックの出力のリスト。collect=True のときは (出力, attention 出力, MLP 出力, attention logits) のリストの組。
        """
        outs, attn_outs, mlp_outs, logits = [], [], [], []
        for blk in self.trunk.blocks[start:end]:
            x, x_attn, x_mlp, attn_logits = blk(x, rope=rope, attn_mask=attn_mask)
            outs.append(x)
            if collect:
                attn_outs.append(x_attn)
                mlp_outs.append(x_mlp)
                logits.append(attn_logits)
        return (outs, attn_outs, mlp_outs, logits) if collect else outs

    @torch.no_grad()
    def forward_all(self, image_tensor: np.ndarray):
        """全層の hidden（hs[0]=埋め込み直後, hs[l]=第 l ブロックの出力）と最終 norm 後のトークンを返す。"""
        x, rope = self.embed(image_tensor)
        hs = [x] + self.run_blocks(x, rope)
        return {"hs": hs, "rope": rope, "final": self.trunk.norm(hs[-1])}

    @torch.no_grad()
    def pool(self, tokens: torch.Tensor, labels: np.ndarray) -> np.ndarray:
        """トークン列 (1,1+N,C)（最終 norm 後を想定）をラベルごとに attention pool する。(最大ラベル+1, 512) を返す。"""
        labels = torch.from_numpy(np.asarray(labels)).to(device=self.device, dtype=torch.int64)
        return self.trunk.custom_forward(tokens, labels, normalize=True).cpu().numpy()

    # ---- CLIPBackend の契約（OVCLIPFeatureExtractor と同じ出力の形式） ----

    def fused_features(self, hs, layers=None) -> torch.Tensor:
        """指定層（既定は cluster_layers）を層ごとに正規化して連結し、再び正規化する（OpenVINO 変換済みモデルと同じ）。"""
        layers = self.cluster_layers if layers is None else layers
        return F.normalize(torch.cat([F.normalize(hs[l], dim=-1) for l in layers], dim=2), dim=-1)

    @torch.no_grad()
    def extract_image_features(self, image_tensor: np.ndarray, normalize: bool = True):
        out = self.forward_all(image_tensor)
        cls_token = self.trunk.forward_head(out["final"])[0]
        if normalize:
            cls_token = F.normalize(cls_token, dim=-1)
        return cls_token.cpu().numpy(), out["final"].cpu().numpy(), self.fused_features(out["hs"]).cpu().numpy()

    @torch.no_grad()
    def extract_text_features(self, prompts: List[str], normalize: bool = True) -> np.ndarray:
        tokens = self.tokenizer(prompts).to(self.device)
        return self.model.encode_text(tokens, normalize=normalize).cpu().numpy()

    def extract_cluster_tokens(self, patch_features: np.ndarray, fused_cluster_labels: np.ndarray) -> np.ndarray:
        return self.pool(self._tensor(patch_features), fused_cluster_labels)
