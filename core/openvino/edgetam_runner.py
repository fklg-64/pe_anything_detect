from core.openvino.runtime_utils import limit_cpu_threads, model_file_suffix  # openvino より先に import する（sys.path の整理）

from pathlib import Path
from typing import Tuple

import numpy as np

import openvino as ov

from core.common.edgetam_helper import EdgeTamHelper
from core.openvino.compiled_model import OVModel, StaticPlan, static_input_side
from core.openvino.graph_edits import (
    NCHW_FROM_NHWC,
    NHWC_FROM_NCHW,
    decoder_feats0_from_nhwc,
    encoder_feats0_as_nhwc,
    remove_decoder_output_reorder,
)

_UNEXPECTED_GRAPH = "のグラフが想定と違うため、グラフを書き換えられません（変換し直したモデルなど。config の ov_static_shape: false なら書き換えずに動きます）"


def _optimize_static_decoder(model: "ov.Model") -> None:
    """静的形状のデコーダーのグラフから Reorder を除く（graph_edits.py）。出力側の matmul と、入力 high_res_feats0 を NHWC で受けること。"""
    if not (remove_decoder_output_reorder(model) and decoder_feats0_from_nhwc(model)):
        raise RuntimeError("デコーダー" + _UNEXPECTED_GRAPH)


def _optimize_static_encoder(model: "ov.Model") -> None:
    """静的形状のエンコーダーが high_res_feats0 を NHWC で出すようにする（デコーダーの入力と合わせる。graph_edits.py）。"""
    if not encoder_feats0_as_nhwc(model):
        raise RuntimeError("エンコーダー" + _UNEXPECTED_GRAPH)


class OVEdgeTamRunner(EdgeTamHelper):
    """EdgeTAMのOpenVINO backend。推論呼び出しだけを持つ（前後処理は EdgeTamHelper）。"""

    def __init__(self, config: dict):
        super().__init__(config)

        # OpenVINO固有の設定
        self.model_dir = config["model_dir"]
        self.use_int8 = config["et_use_int8"]
        self.device = config["device"].upper()
        self.num_threads = config["ov_num_thread"]
        # エンコーダーの入力の一辺（ov_static_shape が真のときだけ。None なら動的形状。compiled_model.py）
        self.static_side = static_input_side(config)

        # OpenCV・NumPy(BLAS)のスレッド数（OpenVINOのスレッドと競合して推論が遅くなるのを避ける。runtime_utils 参照）
        limit_cpu_threads()

        # OpenVINOのモデルを読み込み
        self.load_ov_model()

        # 初期実行時の遅延を回避
        self._forward_once()

    def load_ov_model(self) -> None:
        core = ov.Core()

        # モデルパス
        model_dir = Path(self.model_dir)
        suffix = model_file_suffix(self.use_int8)
        encoder_path = model_dir / f"edgetam_encoder{suffix}"
        decoder_path = model_dir / f"edgetam_decoder{suffix}"

        side = self.static_side
        # 静的形状のモデルは、グラフを書き換えて速くする（出力は同じ）。外から見える特徴量は NCHW のまま、high_res_feats0 を内部では NHWC で持つ。
        # 動的形状のモデルは書き換えない（compiled_model.py）
        encoder_plan = decoder_plan = None
        if side is not None:
            encoder_plan = StaticPlan({"x": [1, 3, side, side]}, _optimize_static_encoder, output_perms={0: NCHW_FROM_NHWC})
        self.encoder_model = OVModel(core, encoder_path, self.device, self.num_threads, static=encoder_plan)
        if side is not None:
            decoder_plan = StaticPlan(self._decoder_static_shapes(), _optimize_static_decoder, input_perms={0: NHWC_FROM_NCHW})
        self.decoder_model = OVModel(core, decoder_path, self.device, self.num_threads, static=decoder_plan)

    def _decoder_static_shapes(self):
        """静的形状のデコーダーに固定する形。特徴量と初期マスクは、静的形状のエンコーダーの出力の形（点の数は固定しない）。

        形がこれと違う入力（格子分離のセルごとのデコード）は、動的形状のデコーダーで推論する。
        """
        feats0, feats1, embed = self.encoder_model.output_shapes
        return {
            "high_res_feats0": list(feats0),
            "high_res_feats1": list(feats1),
            "image_embed": list(embed),
            "point_coords": [1, -1, 2],
            "point_labels": [1, -1],
            "mask_input": [1, 1, feats0[2], feats0[3]],
            "has_mask": [1],
        }

    def _forward_once(self) -> None:
        """ダミー入力でエンコーダとデコーダを1回ずつ実行する（初回実行の遅延を起動時に済ませる）。"""
        dummy_image = np.zeros((1, 3, self.target_size, self.target_size), dtype=np.float32)
        high_res_feats0, high_res_feats1, image_embed = self.encode(dummy_image)

        num_dummy_points = 1
        _, _, h0, w0 = high_res_feats0.shape
        self.decode(
            high_res_feats0,
            high_res_feats1,
            image_embed,
            point_coords=np.zeros((1, num_dummy_points, 2), dtype=np.float32),
            point_labels=np.ones((1, num_dummy_points), dtype=np.int32),
            mask_input=np.zeros((1, 1, h0, w0), dtype=np.float32),
            has_mask=np.zeros((1,), dtype=np.float32),
        )

    def encode(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(1,3,S,S) -> (high_res_feats0, high_res_feats1, image_embed)"""
        high_res_feats0, high_res_feats1, image_embed = self.encoder_model([image_tensor])
        return high_res_feats0, high_res_feats1, image_embed

    def decode(
        self,
        high_res_feats0: np.ndarray,
        high_res_feats1: np.ndarray,
        image_embed: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: np.ndarray,
        has_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """プロンプト（点・マスク）とエンコーダの出力 -> (low_res_masks, iou_predictions)"""
        low_res_masks, iou_predictions = self.decoder_model([
            high_res_feats0,
            high_res_feats1,
            image_embed,
            point_coords,
            point_labels,
            mask_input,
            has_mask,
        ])
        return low_res_masks, iou_predictions
