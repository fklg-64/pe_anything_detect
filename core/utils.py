import time
from typing import List, Optional, Tuple

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