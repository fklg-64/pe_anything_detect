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


class RepVitModel(nn.Module):
    def __init__(self, config_path: str = "./configs/edgetam.yaml", checkpoint_path: str = "weights/edgetam.pt", device: str = "cpu"):
        super().__init__()
        # クラス経由で共有インスタンスを取得（初回のみロードされる）
        self.edgetam_model, self._transforms = EdgeTamModel.get_model(config_path, checkpoint_path, device)
        # エンコーダー部分のみを抽出
        self.trunk: RepVit = self.edgetam_model.image_encoder.trunk
        self.neck: FpnNeck = self.edgetam_model.image_encoder.neck
        self.scalp: int = self.edgetam_model.image_encoder.scalp

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (self.edgetam_model.image_size // 4, self.edgetam_model.image_size // 4), 
            (self.edgetam_model.image_size // 8, self.edgetam_model.image_size // 8), 
            (self.edgetam_model.image_size // 16, self.edgetam_model.image_size // 16), 
        ]

    def transform(self, x):
        return self._transforms(x)[None, ...]

    def forward(self, x: torch.Tensor):
        # Forward through backbone
        features, pos = self.neck(self.trunk(x))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        features[0] = self.edgetam_model.sam_mask_decoder.conv_s0(
            features[0]
        )
        features[1] = self.edgetam_model.sam_mask_decoder.conv_s1(
            features[1]
        )

        feature_maps = features[-self.edgetam_model.num_feature_levels :]

        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        if self.edgetam_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.edgetam_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        # high_res_feats[0], high_res_feats[1], image_embed
        return feats[0], feats[1], feats[2]


class DecodeModel(nn.Module):
    def __init__(self, config_path: str = "./configs/edgetam.yaml", checkpoint_path: str = "weights/edgetam.pt", device: str = "cpu"):
        # nn.Module の初期化
        super().__init__()
        # クラス経由で共有インスタンスを取得（初回のみロードされる）
        self.edgetam_model, self._transforms = EdgeTamModel.get_model(config_path, checkpoint_path, device)
        # 閾値
        self.mask_threshold = self._transforms.mask_threshold

    def _prep_prompts(
        self, 
        orig_hw: List[Tuple[int, int]], 
        point_coords: Optional[torch.Tensor], 
        point_labels: Optional[torch.Tensor], 
        box: Optional[torch.Tensor], 
        mask_logits: Optional[torch.Tensor], 
        normalize_coords: bool
    ):
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=orig_hw
            )
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], point_labels[None, ...]
        if box is not None:
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw
            )  # Bx2x2
        if mask_logits is not None:
            if len(mask_logits.shape) == 3:
                mask_input = mask_logits[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self, 
        high_res_feats0: torch.Tensor, 
        high_res_feats1: torch.Tensor, 
        image_embed: torch.Tensor, 
        orig_hw: List[Tuple[int, int]], 
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.edgetam_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        low_res_masks, iou_predictions, _, _ = self.edgetam_model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.edgetam_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=[high_res_feats0, high_res_feats1],
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, orig_hw
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def forward(
        self, 
        high_res_feats0: torch.Tensor, 
        high_res_feats1: torch.Tensor, 
        image_embed: torch.Tensor, 
        orig_hw: torch.Tensor, 
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 入力画像サイズ
        _orig_hw = [orig_hw[0], orig_hw[1]]

        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            _orig_hw, point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict(
            high_res_feats0, 
            high_res_feats1, 
            image_embed, 
            _orig_hw, 
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks.squeeze(0).float()
        iou_predictions = iou_predictions.squeeze(0).float()
        low_res_masks = low_res_masks.squeeze(0)

        return masks, iou_predictions, low_res_masks