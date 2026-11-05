"""EdgeTAM backend（OpenVINO / PyTorch）を呼ぶ口。エンコーダー・デコーダーの実行と、その前後処理をまとめる。

デコーダーは同時に 1 つだけ実行する（lock）。領域の推定を別スレッドで重ねても（parallel.py）、モデルの実行は直列になる。
クラスごとのデコード（decode_prompts）も、格子分離のセルごとの一括デコード（decode_batch）も、この口を通る。
"""

import threading
from typing import List, Tuple, Union

import numpy as np

from core.utils import Timer, normalize_image
from core.common.backend import EdgeTamBackend
from core.common.edgetam_helper import EdgeTamHelper

Features = Tuple[np.ndarray, np.ndarray, np.ndarray]  # EdgeTAM エンコーダの出力 (high_res_feats0, high_res_feats1, image_embed)


class EdgeTamSession:
    def __init__(self, backend: Union[EdgeTamBackend, EdgeTamHelper]):
        self._backend = backend
        self._lock = threading.Lock()
        self.target_size: int = backend.target_size

    def encode(self, resized: np.ndarray) -> Features:
        """画像（リサイズ済み・正規化前、uint8）の特徴量を求める。"""
        return self._backend.encode(normalize_image(resized, self._backend.mean, self._backend.std))

    def decode_prompts(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Features,
        height: int,
        width: int,
        mask_input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """点と初期マスクのプロンプトで 1 回デコードする。(予測 IoU (候補数,), 低解像度ロジット (候補数, h, w)) を返す。

        point_coords: 元画像（height x width）の画素座標。mask_input: 初期マスクのロジット (1, h, w)。
        """
        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self._backend.preprocess(
            (height, width),
            np.array(point_coords, dtype=np.float32),
            np.array(point_labels, dtype=np.int32),
            mask_input,
        )

        high_res_feats0, high_res_feats1, image_embed = feats
        with self._lock, Timer("    ┌ デコード"):
            low_res_masks, iou_predictions = self._backend.decode(
                high_res_feats0, high_res_feats1, image_embed,
                point_coords=point_coords_norm,
                point_labels=point_labels_norm,
                mask_input=mask_input_tensor,
                has_mask=has_mask,
            )

        with Timer("    ┌ リサイズ"):
            return self._backend.postprocess(low_res_masks, iou_predictions)

    def decode_batch(
        self,
        high_res_feats0: np.ndarray,
        high_res_feats1: np.ndarray,
        image_embed: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: np.ndarray,
        has_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前処理済みの入力（バッチ）をそのままデコードする。(low_res_masks, iou_predictions)。格子分離が使う。"""
        with self._lock:
            return self._backend.decode(
                high_res_feats0, high_res_feats1, image_embed, point_coords, point_labels, mask_input, has_mask
            )
