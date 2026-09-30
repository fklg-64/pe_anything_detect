import sys
from pathlib import Path

core_dir = Path(__file__).resolve().parent
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))
    
from typing import Union, List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.transforms import SAM2Transforms

from timm.models.repvit import RepVit

class EdgeTAMPredictor:
    def __init__(
        self, 
        checkpoint_path: str = "./weights/edgetam.pt", 
        config_path: str = "configs/edgetam.yaml",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        モデルの初期化と重みのロードを行います。
        """
        self.device = device
        self.dtype = dtype
        
        # モデルとPredictorの構築
        sam_model = build_sam2(config_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(
        self, 
        image_input: Union[str, Path, np.ndarray, Image.Image], 
    ) -> None:
        # 画像パスが渡された場合はNumPy配列（RGB）に変換
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
            image_np = np.array(image)
        elif isinstance(image_input, Image.Image):
            image_np = np.array(image_input.convert("RGB"))
        else:
            image_np = image_input

        # 推論モードとAutocastを適用して推論
        with torch.inference_mode():
            self.predictor.set_image(image_np)

    def predict(
        self, 
        box: Union[List[float], np.ndarray] = None,
        point_coords: Union[List[List[float]], np.ndarray] = None,
        point_labels: Union[List[int], np.ndarray] = None,
        mask_input: Union[List[int], np.ndarray] = None,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        画像パスとプロンプトを受け取り、マスクを予測して返します。
        
        :param box: バウンディングボックスプロンプト [x1, y1, x2, y2]
        :param point_coords: ポイント座標 [[x1, y1], [x2, y2], ...]
        :param point_labels: ポイントラベル [1, 0, ...] (1: 前景, 0: 背景)
        :param mask_input: マスク画像
        :param multimask_output: 複数マスクを出力するかどうか
        :return: (masks, scores, logits)
        """
        # 推論モードとAutocastを適用して推論
        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )

        return masks, scores, logits


class EdgeTamModel:
    """SAM2モデルの読み込みと保持を担うクラス"""
    _sam_model = None
    _transforms = None

    @classmethod
    def get_model(cls, config_path, checkpoint_path, device="cpu") -> Tuple[SAM2Base, SAM2Transforms]:
        if cls._sam_model is None:
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


# class RepVitModel(nn.Module):
#     def __init__(self, config_path: str = "./configs/edgetam.yaml", checkpoint_path: str = "weights/edgetam.pt", device: str = "cpu"):
#         super().__init__()
#         # クラス経由で共有インスタンスを取得（初回のみロードされる）
#         self.edgetam_model, self._transforms = EdgeTamModel.get_model(config_path, checkpoint_path, device)
#         # エンコーダー部分のみを抽出
#         self.trunk: RepVit = self.edgetam_model.image_encoder.trunk
#         self.neck: FpnNeck = self.edgetam_model.image_encoder.neck
#         self.scalp: int = self.edgetam_model.image_encoder.scalp

#         # Spatial dim for backbone feature maps
#         self._bb_feat_sizes = [
#             (self.edgetam_model.image_size // 4, self.edgetam_model.image_size // 4), 
#             (self.edgetam_model.image_size // 8, self.edgetam_model.image_size // 8), 
#             (self.edgetam_model.image_size // 16, self.edgetam_model.image_size // 16), 
#         ]

#     def transform(self, x):
#         return self._transforms(x)[None, ...]

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         # Forward through backbone
#         features, pos = self.neck(self.trunk(x))
#         if self.scalp > 0:
#             # Discard the lowest resolution features
#             features, pos = features[: -self.scalp], pos[: -self.scalp]

#         # precompute projected level 0 and level 1 features in SAM decoder
#         # to avoid running it again on every SAM click
#         features[0] = self.edgetam_model.sam_mask_decoder.conv_s0(
#             features[0]
#         )
#         features[1] = self.edgetam_model.sam_mask_decoder.conv_s1(
#             features[1]
#         )

#         feature_maps = features[-self.edgetam_model.num_feature_levels :]

#         # flatten NxCxHxW to HWxNxC
#         vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

#         if self.edgetam_model.directly_add_no_mem_embed:
#             vision_feats[-1] = vision_feats[-1] + self.edgetam_model.no_mem_embed

#         feats = [
#             feat.permute(1, 2, 0).view(1, -1, *feat_size)
#             for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
#         ][::-1]

#         # high_res_feats[0], high_res_feats[1], image_embed
#         return feats[0], feats[1], feats[2]


class RepVitModel(nn.Module):

    def __init__(
        self,
        config_path: str = "./configs/edgetam.yaml",
        checkpoint_path: str = "weights/edgetam.pt",
        device: str = "cpu",
    ):
        super().__init__()
        # クラス経由で共有インスタンスを取得（初回のみロードされる）
        self.edgetam_model, self._transforms = EdgeTamModel.get_model(
            config_path, checkpoint_path, device
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# class DecodeModel(nn.Module):
#     def __init__(self, config_path: str = "./configs/edgetam.yaml", checkpoint_path: str = "weights/edgetam.pt", device: str = "cpu"):
#         # nn.Module の初期化
#         super().__init__()
#         # クラス経由で共有インスタンスを取得（初回のみロードされる）
#         self.edgetam_model, self._transforms = EdgeTamModel.get_model(config_path, checkpoint_path, device)
#         # 閾値
#         self.mask_threshold = self._transforms.mask_threshold
#         # 重み
#         self.no_mask_embed = self.edgetam_model.sam_prompt_encoder.no_mask_embed

#     def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
#         """Embeds mask inputs."""
#         mask_embedding = self.edgetam_model.sam_prompt_encoder.mask_downscaling(masks)
#         return mask_embedding

#     @torch.no_grad()
#     def forward(
#         self, 
#         high_res_feats0: torch.Tensor, 
#         high_res_feats1: torch.Tensor, 
#         image_embed: torch.Tensor, 
#         point_coords: torch.Tensor, 
#         point_labels: torch.Tensor, 
#         mask_input: torch.Tensor, 
#         has_mask: torch.Tensor,
#     ):
#         concat_points = (point_coords, point_labels)

#         sparse_embeddings, _ = self.edgetam_model.sam_prompt_encoder(
#             points=concat_points,
#             boxes=None,
#             masks=None,
#         )

#         # dense_embeddingsのパッチ処理（OpenVINO対応）
#         has_mask_b = has_mask.reshape(-1, 1, 1, 1)
#         masks_gated = mask_input * has_mask_b
#         mask_embeds = self._embed_masks(masks_gated)
#         no_mask_embeds = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
#         dense_embeddings = mask_embeds * has_mask_b + no_mask_embeds * (1.0 - has_mask_b)

#         # Predict masks
#         batched_mode = (
#             concat_points is not None and concat_points[0].shape[0] > 1
#         )  # multi object prediction
#         low_res_masks, iou_predictions, _, _ = self.edgetam_model.sam_mask_decoder(
#             image_embeddings=image_embed,
#             image_pe=self.edgetam_model.sam_prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=True,
#             repeat_image=batched_mode,
#             high_res_features=[high_res_feats0, high_res_feats1],
#         )

#         return low_res_masks, iou_predictions

#     def preprocess(
#         self, 
#         orig_hw: Tuple[int, int], 
#         point_coords: Optional[torch.Tensor] = None, 
#         point_labels: Optional[torch.Tensor] = None, 
#         box: Optional[torch.Tensor] = None, 
#         mask_logits: Optional[torch.Tensor] = None, 
#         normalize_coords: bool = True
#     ):
#         unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
#         if point_coords is not None:
#             assert (
#                 point_labels is not None
#             ), "point_labels must be supplied if point_coords is supplied."
#             unnorm_coords = self._transforms.transform_coords(
#                 point_coords, normalize=normalize_coords, orig_hw=orig_hw
#             )
#             if len(unnorm_coords.shape) == 2:
#                 unnorm_coords, labels = unnorm_coords[None, ...], point_labels[None, ...]
#         if box is not None:
#             unnorm_box = self._transforms.transform_boxes(
#                 box, normalize=normalize_coords, orig_hw=orig_hw
#             )  # Bx2x2
#         if mask_logits is not None:
#             if len(mask_logits.shape) == 3:
#                 mask_input = mask_logits[None, :, :, :]
#             elif len(mask_logits.shape) == 1:
#                 bs = mask_logits.shape[0]
#                 mask_input = mask_logits.reshape(-1, 1, 1, 1) * \
#                     torch.zeros((bs, 1, self.edgetam_model.image_size // 4, self.edgetam_model.image_size // 4), dtype=torch.float32)

#         if unnorm_coords is not None:
#             concat_points = (unnorm_coords, labels)
#         else:
#             concat_points = None

#         # Embed prompts
#         if unnorm_box is not None:
#             box_coords = unnorm_box.reshape(-1, 2, 2)
#             box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
#             box_labels = box_labels.repeat(unnorm_box.size(0), 1)
#             # we merge "boxes" and "points" into a single "concat_points" input (where
#             # boxes are added at the beginning) to sam_prompt_encoder
#             if concat_points is not None:
#                 concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
#                 concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
#                 concat_points = (concat_coords, concat_labels)
#             else:
#                 concat_points = (box_coords, box_labels)

#         return mask_input, concat_points[0], concat_points[1]

#     def postprocess(self, low_res_masks: torch.Tensor, iou_predictions: torch.Tensor, orig_hw: Tuple[int, int], return_logits: bool = False):
#         # Upscale the masks to the original image resolution
#         masks = self._transforms.postprocess_masks(
#             low_res_masks, orig_hw
#         )
#         low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
#         if not return_logits:
#             masks = masks > self.mask_threshold

#         masks = masks.squeeze(0).float()
#         iou_predictions = iou_predictions.squeeze(0).float()
#         low_res_masks = low_res_masks.squeeze(0)

#         return masks, iou_predictions, low_res_masks


class DecodeModel(nn.Module):
    """EdgeTAM モデルのデコード処理用 PyTorch モジュール。
    
    OpenVINO 変換に対応するためのインターフェースおよびグラフ構造を保持しています。
    """

    def __init__(
        self,
        config_path: str = "./configs/edgetam.yaml",
        checkpoint_path: str = "weights/edgetam.pt",
        device: str = "cpu",
    ):
        super().__init__()
        # モデルと画像変換器の共有インスタンスを取得
        self.edgetam_model, self._transforms = EdgeTamModel.get_model(
            config_path, checkpoint_path, device
        )
        self.mask_threshold: float = self._transforms.mask_threshold
        self.no_mask_embed = self.edgetam_model.sam_prompt_encoder.no_mask_embed

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """マスク入力のダウンスケーリングエンコーディング"""
        return self.edgetam_model.sam_prompt_encoder.mask_downscaling(masks)

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
        """OpenVINO 変換に対応した forward 推論。
        
        注意: dense_embeddings の計算処理 (has_mask_b を使用したゲート演算) は
        ONNX/OpenVINO 変換時の動的条件分岐エラーを防ぐための必須処理です。
        """
        concat_points = (point_coords, point_labels)

        # 1. 疎プロンプト (ポイント) のエンコード
        sparse_embeddings, _ = self.edgetam_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
        )

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
            image_pe=self.edgetam_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=[high_res_feats0, high_res_feats1],
        )

        return low_res_masks, iou_predictions

    def preprocess(
        self,
        orig_hw: Tuple[int, int],
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        mask_logits: Optional[torch.Tensor] = None,
        normalize_coords: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """入力プロンプト (ポイント、ボックス、マスク) の前処理と結合を行います。"""
        unnorm_coords: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None
        unnorm_box: Optional[torch.Tensor] = None
        mask_input: Optional[torch.Tensor] = None

        # --- 1. ポイントプロンプトの処理 ---
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied if point_coords is supplied.")

            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=orig_hw
            )
            labels = point_labels
            if unnorm_coords.ndim == 2:
                unnorm_coords = unnorm_coords[None, ...]
                labels = labels[None, ...]

        # --- 2. ボックスプロンプトの処理 ---
        if box is not None:
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw
            )  # B x 2 x 2

        # --- 3. マスクプロンプトの処理 ---
        if mask_logits is not None:
            if mask_logits.ndim == 3:
                mask_input = mask_logits[None, :, :, :]
            elif mask_logits.ndim == 1:
                # フラグ等で空マスクが必要な場合のゼロ埋めテンソル生成
                batch_size = mask_logits.shape[0]
                target_size = self.edgetam_model.image_size // 4
                mask_input = torch.zeros(
                    (batch_size, 1, target_size, target_size),
                    dtype=torch.float32,
                    device=mask_logits.device,
                )
            has_mask = torch.ones((1, ), dtype=torch.float32)
        else:
            has_mask = torch.zeros((1, ), dtype=torch.float32)

        # --- 4. ポイントとボックスの統合 ---
        concat_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if unnorm_coords is not None and labels is not None:
            concat_points = (unnorm_coords, labels)

        if unnorm_box is not None:
            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = torch.tensor(
                [[2, 3]], dtype=torch.int, device=unnorm_box.device
            ).repeat(unnorm_box.size(0), 1)

            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # プロンプトが何もない場合のエッジケース保護
        if concat_points is None:
            return mask_input, None, None

        return mask_input, concat_points[0], concat_points[1], has_mask

    def postprocess(
        self,
        low_res_masks: torch.Tensor,
        iou_predictions: torch.Tensor,
        orig_hw: Tuple[int, int],
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """モデル出力の後処理 (オリジナル解像度への拡大および閾値処理) を行います。"""
        # マスクを元画像の解像度にリサイズ
        masks = self._transforms.postprocess_masks(low_res_masks, orig_hw)
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        if not return_logits:
            masks = masks > self.mask_threshold

        # 不要なバッチ次元を削除して float 型へキャスト
        masks = masks.squeeze(0).float()
        iou_predictions = iou_predictions.squeeze(0).float()
        low_res_masks = low_res_masks.squeeze(0)

        return masks, iou_predictions, low_res_masks