import cv2
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Union

COLOR_PALETTE: List[int] = [
    0, 0, 0,
    255, 0, 0,
    0, 255, 0,
    0, 0, 255,
    255, 255, 0,
    0, 255, 255,
    255, 0, 255,
    192, 192, 192,
    128, 0, 0,
    128, 128, 0,
    0, 128, 0,
    128, 0, 128,
    0, 0, 128,
    255, 165, 0,
    255, 105, 180,
    255, 69, 0,
    255, 20, 147,
    34, 139, 34,
    0, 255, 127,
    255, 182, 193,
    255, 215, 0,
    139, 69, 19,
    244, 164, 96,
    72, 61, 139,
    0, 191, 255,
    70, 130, 180,
    147, 112, 219,
    153, 50, 204,
    255, 228, 225,
    255, 0, 255,
    255, 228, 181,
    255, 99, 71,
    0, 250, 154,
    221, 160, 221,
    238, 130, 238,
    186, 85, 211,
    255, 240, 245,
    219, 112, 147,
    255, 69, 0,
    240, 128, 128,
    255, 105, 180,
]


class Timer:
    """ミリ秒単位で処理時間を計測し表示するコンテキストマネージャー"""
    def __init__(self, name: str, width: int = 12):
        self.name = name
        self.width = width
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        print(f"{self.name:<{self.width}} : {self.elapsed_ms:6.2f} ms")


def load_image(data: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """
    data:
        str or Path : パス
        Image       : Image型のRGB画像
        ndarray     : ndarray型のRGB画像 (BGRで入力しないこと)
    """
    if isinstance(data, (str, Path)):
        img_cv = np.array(Image.open(data).convert("RGB"))
    elif isinstance(data, Image.Image):
        img_cv = np.array(data.convert("RGB"))
    elif isinstance(data, np.ndarray):
        img_cv = data
    else:
        raise TypeError(f"未対応の画像形式です: {type(data)}")
    return img_cv


def preprocess_image(img_cv: np.ndarray, target_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    NNに入力するための前処理

    img_cv: ndarray型のRGB画像
    """
    resized_cv = cv2.resize(
        img_cv, 
        (target_size, target_size), 
        interpolation=cv2.INTER_LINEAR
    )

    img_float = resized_cv.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))

    img_chw = (img_chw - mean) / std

    return np.expand_dims(img_chw, axis=0)