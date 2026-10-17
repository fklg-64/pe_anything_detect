import cv2
import time
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Union

class Timer:
    """ミリ秒単位で処理時間を計測し表示するコンテキストマネージャー"""
    def __init__(self, name: str, width: int = 20):
        self.name = name
        self.width = width
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        print(f"{self.name:<{self.width}} : {self.elapsed_ms:6.2f} ms")

# 画像読み込み
def load_image(data: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """画像をRGB形式のnumpy配列に変換する。"""
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, (str, Path)):
        with Image.open(data) as image:
            return np.array(image.convert("RGB"))

    if isinstance(data, Image.Image):
        return np.array(data.convert("RGB"))

    raise TypeError(f"未対応の画像形式です: {type(data)}")

# 画像リサイズ（CLIP / EdgeTAM で共通。正規化前のuint8のまま返す）
def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """画像を target_size 四方にリサイズする。"""
    return cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )

# 画像の正規化
def normalize_image(
    resized: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """リサイズ済み画像を正規化し、NCHW形式に変換する。"""
    # Normalize to [0, 1]
    image = resized.astype(np.float32) / 255.0

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]

    # Normalize and add batch dimension
    return ((image - mean) / std)[None, ...]

# 画像前処理
def preprocess_image(
    image: np.ndarray,
    target_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """画像をリサイズ・正規化し、NCHW形式に変換する。"""
    return normalize_image(resize_image(image, target_size), mean, std)

# yaml設定データの読み込み
def load_config(config_path: str = "./configs/ov_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config