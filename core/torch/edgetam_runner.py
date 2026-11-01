import sys
from pathlib import Path

# core/torch を sys.path に加える。EdgeTAM のベンダリングコード core/torch/sam2 を、トップレベルのパッケージ `sam2` として import するため
core_dir = Path(__file__).resolve().parent
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

from core.common.edgetam_helper import EdgeTamHelper
from core.common.params import EDGETAM_WEIGHTS_URL
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.utils.transforms import SAM2Transforms

from timm.models.repvit import RepVit


def ensure_edgetam_weights(checkpoint_path) -> Path:
    """重みが checkpoint_path に無ければ、そのパスへダウンロードする（途中で失敗しても壊れたファイルを残さない）"""
    path = Path(checkpoint_path)
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"EdgeTAM の重みが見つからないためダウンロードします: {EDGETAM_WEIGHTS_URL} -> {path}")
    try:
        urllib.request.urlretrieve(EDGETAM_WEIGHTS_URL, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path

class EdgeTamModel:
    """SAM2モデルの読み込みと保持を担うクラス"""
    _sam_model = None
    _transforms = None

    @classmethod
    def get_model(cls, config_path, checkpoint_path, device="cpu") -> Tuple[SAM2Base, SAM2Transforms]:
        if cls._sam_model is None:
            ensure_edgetam_weights(checkpoint_path)
            cls._sam_model: SAM2Base = build_sam2(config_path, checkpoint_path, device=device)
        if cls._transforms is None:
            # 閾値
            mask_threshold = 0.0
            # tramsform
            cls._transforms: SAM2Transforms = SAM2Transforms(
                resolution=cls._sam_model.image_size,
                mask_threshold=mask_threshold,
                max_hole_area=0.0,
                max_sprinkle_area=0.0,
            )
        return cls._sam_model, cls._transforms


class BaseModel(nn.Module):
    """EdgeTAMのモデルの重み・設定の場所と、デバイス・精度を設定から受け取る基底クラス。"""

    def __init__(self, config: dict):
        super().__init__()
        self.checkpoint_path = config["edgetam_model_path"]
        self.config_path = config["edgetam_model_config"]
        self.device = config["device"]
        self.dtype = getattr(torch, config["dtype"])


class RepVitModel(BaseModel):
    """EdgeTAMの画像エンコーダ（RepViT + FPN）。OpenVINO変換の元モデルも兼ねる。"""

    def __init__(self, config: dict):
        super().__init__(config)
        # クラス経由で共有インスタンスを取得（初回のみロードされる）
        self.edgetam_model, self._transforms = EdgeTamModel.get_model(
            self.config_path, self.checkpoint_path, self.device
        )
        # エンコーダー部分のみを抽出
        self.trunk: RepVit = self.edgetam_model.image_encoder.trunk
        self.neck: FpnNeck = self.edgetam_model.image_encoder.neck
        self.scalp: int = self.edgetam_model.image_encoder.scalp

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self._transforms(x)[None, ...]

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Backbone & FPN 推論
        features, pos = self.neck(self.trunk(x))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        # SAM decoder 用に level 0, 1 の特徴量を投影
        features[0] = self.edgetam_model.sam_mask_decoder.conv_s0(features[0])
        features[1] = self.edgetam_model.sam_mask_decoder.conv_s1(features[1])

        # 対象レベルの特徴量マップを取得 [level0, level1, image_embed]
        feature_maps = features[-self.edgetam_model.num_feature_levels :]

        # B, C, H, W に合わせてブロードキャスト加算
        if self.edgetam_model.directly_add_no_mem_embed:
            no_mem_embed = self.edgetam_model.no_mem_embed.reshape(1, -1, 1, 1)
            feature_maps[-1] = feature_maps[-1] + no_mem_embed

        # high_res_feats[0], high_res_feats[1], image_embed
        return feature_maps[0], feature_maps[1], feature_maps[2]


class DecodeModel(BaseModel):
    """EdgeTAM モデルのデコード処理用 PyTorch モジュール。
    
    OpenVINO 変換に対応するためのインターフェースおよびグラフ構造を保持しています。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # モデルと画像変換器の共有インスタンスを取得
        self.edgetam_model, self._transforms = EdgeTamModel.get_model(
            self.config_path, self.checkpoint_path, self.device
        )
        self.no_mask_embed = self.edgetam_model.sam_prompt_encoder.no_mask_embed

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """マスク入力のダウンスケーリングエンコーディング"""
        return self.edgetam_model.sam_prompt_encoder.mask_downscaling(masks)

    def _dense_pe(self, image_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """画像埋め込みと同じ大きさの密な位置エンコーディング (1, C, H, W) と、H・W（0 次元テンソル）を返す。

        `sam_prompt_encoder.get_dense_pe()` は、モデルの設定の画像サイズ（例: 512 → 32×32）で固定の大きさになり、OpenVINO 変換でも
        定数として焼き込まれて、他の画像サイズでは加算の形が合わなくなる。ここでは形を image_embed のテンソルから求める
        （`.shape` の整数ではなく、全て 1 のベクトルの和で H・W を得るので、トレース時にも定数にならない）。

        位置エンコーディング sin / cos(2π (x'·g_x + y'·g_y))（x'・y' は格子の中心を [-1, 1] にした座標、g はガウス行列）は、x と y に分けられる。
        加法定理で、列ごと（W 個）・行ごと（H 個）に sin / cos を求めて、掛け合わせる（H×W 個の sin / cos を毎回計算しない。
        デコードのたびに実行されるので、512 で約 0.5ms の差になる）。PositionEmbeddingRandom._pe_encoding と同じ値（誤差 1e-6 程度）。
        """
        ones_y, ones_x = torch.ones_like(image_embed[0, 0, :, 0]), torch.ones_like(image_embed[0, 0, 0, :])
        height, width = ones_y.sum(), ones_x.sum()
        y = 2 * ((ones_y.cumsum(dim=0) - 0.5) / height) - 1  # (H,)
        x = 2 * ((ones_x.cumsum(dim=0) - 0.5) / width) - 1  # (W,)
        gaussian = self.edgetam_model.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix  # (2, F)
        angle_y = 2 * np.pi * y[:, None] * gaussian[1][None, :]  # (H, F)
        angle_x = 2 * np.pi * x[:, None] * gaussian[0][None, :]  # (W, F)
        sin_y, cos_y = torch.sin(angle_y)[:, None, :], torch.cos(angle_y)[:, None, :]  # (H, 1, F)
        sin_x, cos_x = torch.sin(angle_x)[None, :, :], torch.cos(angle_x)[None, :, :]  # (1, W, F)
        pe = torch.cat([sin_y * cos_x + cos_y * sin_x, cos_y * cos_x - sin_y * sin_x], dim=-1)  # (H, W, 2F)
        return pe.permute(2, 0, 1).unsqueeze(0), height, width

    @torch.no_grad()
    def forward(
        self,
        high_res_feats0: torch.Tensor,
        high_res_feats1: torch.Tensor,
        image_embed: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OpenVINO 変換に対応した forward 推論。入力の画像サイズ（image_embed の大きさ）は可変。

        注意: dense_embeddings の計算処理 (has_mask_b を使用したゲート演算) は
        ONNX/OpenVINO 変換時の動的条件分岐エラーを防ぐための必須処理です。

        画像サイズに依存する 2 か所（密な位置エンコーディング、点の座標の正規化）は、モデルの設定の画像サイズを使わず、
        image_embed の大きさから求める。point_coords は入力画像の画素座標（EdgeTamHelper.transform_coords の出力）。
        """
        concat_points = (point_coords, point_labels)
        prompt_encoder = self.edgetam_model.sam_prompt_encoder

        # 画像サイズ依存の部分: 密な位置エンコーディングと、点の座標の正規化に使う入力画像の大きさ（埋め込みの大きさ × ストライド）
        dense_pe, embed_height, embed_width = self._dense_pe(image_embed)
        stride = prompt_encoder.input_image_size[0] // prompt_encoder.image_embedding_size[0]  # 設定の画像サイズでの比（16）。定数でよい
        configured_input_size = prompt_encoder.input_image_size

        # 1. 疎プロンプト (ポイント) のエンコード
        prompt_encoder.input_image_size = (embed_height * stride, embed_width * stride)
        try:
            sparse_embeddings, _ = prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        finally:
            prompt_encoder.input_image_size = configured_input_size

        # 2. 密プロンプト (マスク) のエンコード (OpenVINO 互換のゲート演算)
        has_mask_b = has_mask.reshape(-1, 1, 1, 1)
        masks_gated = mask_input * has_mask_b
        mask_embeds = self._embed_masks(masks_gated)
        no_mask_embeds = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
        
        dense_embeddings = mask_embeds * has_mask_b + no_mask_embeds * (1.0 - has_mask_b)

        # 3. マスクデコーダーによるマスク予測
        batched_mode = concat_points[0].shape[0] > 1
        low_res_masks, iou_predictions, _, _ = self.edgetam_model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=[high_res_feats0, high_res_feats1],
        )

        return low_res_masks, iou_predictions


class TorchEdgeTamBackend(EdgeTamHelper):
    """EdgeTAMのPyTorch backend。共通ロジックに合わせ、入出力は np.ndarray（前後処理は EdgeTamHelper）。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.device = config["device"]
        self.edgetam_encoder = RepVitModel(config)
        self.edgetam_decoder = DecodeModel(config)

    def _to_tensor(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).to(device=self.device, dtype=dtype)

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def encode(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outs = self.edgetam_encoder(self._to_tensor(image_tensor))
        return tuple(self._to_numpy(o) for o in outs)

    def decode(
        self,
        high_res_feats0: np.ndarray,
        high_res_feats1: np.ndarray,
        image_embed: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: np.ndarray,
        has_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        low_res_masks, iou_predictions = self.edgetam_decoder(
            self._to_tensor(high_res_feats0),
            self._to_tensor(high_res_feats1),
            self._to_tensor(image_embed),
            point_coords=self._to_tensor(point_coords),
            point_labels=self._to_tensor(point_labels, torch.int32),
            mask_input=self._to_tensor(mask_input),
            has_mask=self._to_tensor(has_mask),
        )
        return self._to_numpy(low_res_masks), self._to_numpy(iou_predictions)
