import cv2
import time
import yaml
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


def load_image(data: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """入力データをRGB順序のnumpy.ndarray(uint8)に統一して読み込みます。

    Args:
        data: 画像ファイルパス(str/Path)、PIL Image、またはRGB配列(ndarray)

    Returns:
        np.ndarray: HWC形式・RGB順序の画像配列 (uint8)
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


def preprocess_image(
    img_cv: np.ndarray, 
    target_size: int, 
    mean: np.ndarray, 
    std: np.ndarray
) -> np.ndarray:
    """ニューラルネットワーク入力用の前処理 (リサイズ, [0,1]正規化, 標準化, NCHW化)

    Args:
        img_cv (np.ndarray): HWC形式・RGB順序の画像配列 (uint8)
        target_size (int): リサイズ後の正方形サイズ (幅・高さ)
        mean (np.ndarray): チャンネルごとの平均値
        std (np.ndarray): チャンネルごとの標準偏差

    Returns:
        np.ndarray: NCHW形式 (1, C, H, W) の float32 配列
    """
    # 1. リサイズ (H, W, C)
    resized_cv = cv2.resize(
        img_cv, 
        (target_size, target_size), 
        interpolation=cv2.INTER_LINEAR
    )

    # 2. float32化および [0, 1] 範囲への正規化
    img_float = resized_cv.astype(np.float32) / 255.0

    # 3. 軸の入れ替え (H, W, C) -> (C, H, W)
    img_chw = np.transpose(img_float, (2, 0, 1))

    # 4. mean / std のブロードキャスト対応 (1D配列(C,)の場合、(C, 1, 1)に変形)
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    if mean_arr.ndim == 1:
        mean_arr = mean_arr[:, None, None]
    if std_arr.ndim == 1:
        std_arr = std_arr[:, None, None]

    # 5. 標準化
    img_norm = (img_chw - mean_arr) / std_arr

    # 6. バッチ次元の追加 (C, H, W) -> (1, C, H, W)
    return np.expand_dims(img_norm, axis=0)

# yaml設定データの読み込み
def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config