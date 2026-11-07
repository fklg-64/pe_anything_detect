"""全体で共有する小さなヘルパー（処理時間の計測、画像の読み込み・前処理、特徴量の数値演算、設定ファイルの読み込み）。"""

import contextlib
import cv2
import threading
import time
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union

from core.common.params import IMAGE_EXTENSIONS


class Timer:
    """ミリ秒単位で処理時間を計測し表示するコンテキストマネージャー"""

    _deferred: Optional[list] = None  # Timer.deferred() の中で、表示する行を貯めておく先
    _lock = threading.Lock()          # _deferred を守る（領域の推定は別スレッドからも log する）

    def __init__(self, name: str, width: int = 20):
        self.name = name
        self.width = width
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        Timer.log(f"{self.name:<{self.width}} : {self.elapsed_ms:6.2f} ms")

    @classmethod
    def log(cls, line: str) -> None:
        """1 行表示する。deferred() の中なら、抜けるときにまとめて表示する。"""
        with cls._lock:
            if cls._deferred is None:
                print(line)
            else:
                cls._deferred.append(line)

    @classmethod
    @contextlib.contextmanager
    def deferred(cls):
        """この中の Timer と log の表示を貯めて、抜けるときにまとめて表示する（端末への出力を、計測する処理の内側で待たないため。入れ子なら外側にまとめる）。"""
        with cls._lock:
            nested = cls._deferred is not None
            if not nested:
                cls._deferred = []
        if nested:
            yield
            return
        try:
            yield
        finally:
            with cls._lock:
                lines, cls._deferred = cls._deferred, None
            for line in lines:
                print(line)


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
    """リサイズ済み画像を正規化し、NCHW形式に変換する。

    チャンネルごとに、連続した (1, C, H, W) の配列へ直接書く（HWC の並びのまま転置した配列を返すと、モデルへの入力のたびにコピーが入る）。
    計算は (x / 255 - mean) / std を float32 で順に行うので、値は従来と同じ。
    """
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)

    height, width, channels = resized.shape
    output = np.empty((1, channels, height, width), dtype=np.float32)
    for c in range(channels):
        plane = output[0, c]
        np.divide(resized[:, :, c], 255.0, out=plane, dtype=np.float32)  # [0, 1] へ
        plane -= mean[c]
        plane /= std[c]
    return output

# 特徴量の数値演算
def normalize_rows(features: np.ndarray) -> np.ndarray:
    """特徴 (N, C) を行ごとに L2 正規化する。"""
    return features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# yaml設定データの読み込み
def load_config(config_path: str = "./configs/ov_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# データセット定義yamlから画像パスの一覧を作る（ディレクトリは再帰的に探索）
def load_dataset_paths(dataset_path: str = "./configs/dataset.yaml") -> list[Path]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)["image_paths"]

    image_paths: list[Path] = []
    for entry in entries:
        entry = Path(entry)
        if entry.is_dir():
            image_paths.extend(sorted(p for p in entry.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
        else:
            image_paths.append(entry)
    return image_paths
