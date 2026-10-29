"""モデルへの入力画像のサイズの決め方（リサイズモード）と、入力画像・パッチ格子・元画像の対応。

モード（config の resize_mode）:
- "square": アスペクト比を無視して image_size 四方にリサイズする（従来どおり）。
- "fit":    アスペクト比を保ち、image_size x image_size に収まる最大の、パッチサイズと 32（EdgeTAM の入力の条件）の最小公倍数の倍数にリサイズする
            （縦横比のずれが小さい寸法を選ぶので、長辺が image_size より小さくなることがある）。
- "pad":    "fit" と同じ大きさにリサイズし、短辺側の下／右を端の画素の複製（Replication）で image_size 四方まで埋める。
            特徴量を抽出した後は、パディング部分の特徴を捨てる（crop_*）。

どのモードでも、特徴抽出後（crop 後）の「有効なパッチ格子」(valid_grid) は元画像の全体を線形に覆う。
元画像の座標 (x, y) は、格子上では (x / width * valid_w, y / height * valid_h) になり、モードの違いは吸収される。
"""

from dataclasses import dataclass
from math import lcm
from typing import Tuple

import cv2
import numpy as np

from core.common.params import EDGETAM_INPUT_ALIGNMENT, FIT_ASPECT_TOLERANCE, RESIZE_MODES


@dataclass(frozen=True)
class ImageLayout:
    """1枚の画像に対する、モデル入力と元画像の対応。"""

    mode: str
    patch_size: int
    input_h: int      # モデルに入力する画像（パディング込み）の大きさ
    input_w: int
    content_h: int    # そのうち画像の内容がある領域（左上から）。パッチサイズの倍数
    content_w: int

    @property
    def grid(self) -> Tuple[int, int]:
        """モデルの出力のパッチ格子 (高さ, 幅)（パディング込み）。"""
        return self.input_h // self.patch_size, self.input_w // self.patch_size

    @property
    def valid_grid(self) -> Tuple[int, int]:
        """パディングを除いた、画像の内容があるパッチ格子 (高さ, 幅)。"""
        return self.content_h // self.patch_size, self.content_w // self.patch_size

    @property
    def content_hw(self) -> Tuple[int, int]:
        return self.content_h, self.content_w

    @property
    def has_padding(self) -> bool:
        return (self.content_h, self.content_w) != (self.input_h, self.input_w)

    # ---- 画像 ----

    def apply(self, image: np.ndarray) -> np.ndarray:
        """元画像 -> モデルに入力する画像（uint8、正規化前。パディング込み）。"""
        content = cv2.resize(image, (self.content_w, self.content_h), interpolation=cv2.INTER_LINEAR)
        if not self.has_padding:
            return content
        return cv2.copyMakeBorder(
            content, 0, self.input_h - self.content_h, 0, self.input_w - self.content_w, cv2.BORDER_REPLICATE
        )

    def content(self, resized: np.ndarray) -> np.ndarray:
        """apply の結果からパディングを除いた画像。"""
        return resized[:self.content_h, :self.content_w]

    # ---- 特徴量（パディング部分を捨てる） ----

    def crop_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """(1, 1+gh*gw, C)（先頭は CLS）-> (1, 1+vh*vw, C)。パディングが無ければそのまま返す。"""
        if not self.has_padding:
            return tokens
        gh, gw = self.grid
        vh, vw = self.valid_grid
        patches = tokens[:, 1:, :].reshape(tokens.shape[0], gh, gw, -1)[:, :vh, :vw]
        return np.concatenate([tokens[:, :1, :], patches.reshape(tokens.shape[0], vh * vw, -1)], axis=1)

    def crop_map(self, feature_map: np.ndarray) -> np.ndarray:
        """(..., H, W) の特徴マップ（格子に対して整数倍の解像度）を有効な領域に切り出す。パディングが無ければそのまま返す。"""
        if not self.has_padding:
            return feature_map
        gh, gw = self.grid
        vh, vw = self.valid_grid
        height, width = feature_map.shape[-2:]
        return np.ascontiguousarray(feature_map[..., :height * vh // gh, :width * vw // gw])

    def crop_edgetam(self, feats: Tuple) -> Tuple:
        """EdgeTAM エンコーダの出力 (high_res_feats0, high_res_feats1, image_embed) の各マップを有効な領域に切り出す。"""
        if not self.has_padding:
            return feats
        return tuple(self.crop_map(f) for f in feats)


def make_layout(orig_hw: Tuple[int, int], image_size: int, patch_size: int, mode: str = "square") -> ImageLayout:
    """元画像の大きさ (高さ, 幅) からモデル入力のレイアウトを決める。"""
    if mode not in RESIZE_MODES:
        raise ValueError(f"resize_mode は {RESIZE_MODES} のいずれかにしてください: {mode!r}")
    if mode == "square":
        side = image_size // patch_size * patch_size
        return ImageLayout(mode, patch_size, side, side, side, side)

    # 内容の辺は、パッチサイズと EdgeTAM の入力の条件の両方を満たす単位の倍数にする
    unit = lcm(patch_size, EDGETAM_INPUT_ALIGNMENT)
    side = image_size // unit * unit
    height, width = orig_hw
    scale = side / max(height, width)

    def snap(length: int) -> int:
        """倍率を掛けた長さに最も近い unit の倍数（1 unit 以上、side 以下）。"""
        return int(min(max(round(length * scale / unit), 1) * unit, side))

    if mode == "fit":
        content_h, content_w = _fit_size(height, width, unit, side)
        return ImageLayout(mode, patch_size, content_h, content_w, content_h, content_w)
    content_h, content_w = snap(height), snap(width)
    return ImageLayout(mode, patch_size, side, side, content_h, content_w)


def _fit_size(height: int, width: int, unit: int, side: int) -> Tuple[int, int]:
    """side 四方に収まる unit の倍数の (高さ, 幅) のうち、元画像の縦横比に近く、なるべく大きいものを選ぶ。

    長辺を side から unit ずつ縮め、短辺は縦横比を保つ長さに最も近い unit の倍数にして、縦横比の相対誤差を測る。
    誤差が FIT_ASPECT_TOLERANCE 以下の最初（最大）の寸法を返す。無ければ誤差が最小のもの（同じなら大きいほう）。
    """
    ratio = min(height, width) / max(height, width)  # 短辺 / 長辺
    best = None
    for long_side in range(side, 0, -unit):
        short_side = min(max(round(long_side * ratio / unit), 1) * unit, long_side)
        error = abs(short_side / long_side / ratio - 1.0)
        if error <= FIT_ASPECT_TOLERANCE:
            best = (long_side, short_side)
            break
        if best is None or error < best[2]:
            best = (long_side, short_side, error)
    long_side, short_side = best[:2]
    return (long_side, short_side) if height >= width else (short_side, long_side)
