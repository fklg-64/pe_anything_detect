import sys
from pathlib import Path

core_dir = Path(__file__).resolve().parent
if str(core_dir) not in sys.path:
    sys.path.insert(0, str(core_dir))
    
from typing import Union, List, Dict, Any, Tuple
import torch
from PIL import Image
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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