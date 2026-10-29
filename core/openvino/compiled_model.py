"""OpenVINO のモデル 1 つの読み込みと実行。入力の大きさを固定した（静的形状の）モデルでの実行にも対応する。

変換済みのモデルは、バッチ・画像の大きさ・点の数が動的形状。動的形状のモデルは、入力の大きさごとのカーネル選択やメモリ確保を実行時に行うので、
入力の大きさが決まっているなら、読み込み時に reshape して静的形状でコンパイルした方が速い（変換し直す必要は無い）。ただし、出力は動的形状のモデルと
bit 単位では一致しない（数値が微小に変わる）。config の `ov_static_shape` で切り替える（既定は動的形状）。
"""

from core.openvino.runtime_utils import make_compile_config  # openvino より先に import する（sys.path の整理）

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import openvino as ov

from core.common.params import EDGETAM_INPUT_ALIGNMENT


def static_input_side(config: dict) -> Optional[int]:
    """config の `ov_static_shape` が真なら、モデルに入力する画像の一辺（画素）を返す。偽なら None（動的形状）。

    入力の大きさが画像ごとに変わる resize_mode: fit では使えない。square / pad は、一辺が image_size で一定。
    """
    if not config.get("ov_static_shape", False):
        return None
    resize_mode = config.get("resize_mode", "square")
    if resize_mode == "fit":
        raise ValueError("ov_static_shape は resize_mode: fit では使えません（入力の大きさが画像ごとに変わるため）")
    side = config["image_size"]
    if side % EDGETAM_INPUT_ALIGNMENT:
        raise ValueError(f"ov_static_shape では image_size を {EDGETAM_INPUT_ALIGNMENT} の倍数にしてください: {side}")
    return side


class OVModel:
    """OpenVINO のモデル 1 つ。`model(inputs)` で推論し、出力（numpy 配列）をモデルの出力の順にリストで返す。

    static_shapes（入力名 -> 固定する形。-1 の次元は固定しない）を渡すと、その形に reshape したモデルでコンパイルする。
    入力の形がそれに合うときは静的形状のモデルで、合わないとき（例: 格子分離がセルの数をバッチにして呼ぶデコーダー）は、
    動的形状のモデルで推論する。動的形状のモデルは、必要になったときに初めてコンパイルする。
    static_shapes が None なら、動的形状のモデルだけを使う。

    prepare: 読み込んだモデルに、コンパイルの前に施す変更（未使用の出力を外すなど）。
    """

    def __init__(
        self,
        core: ov.Core,
        path: Path,
        device: str,
        num_threads: int,
        prepare: Optional[Callable[[ov.Model], None]] = None,
        static_shapes: Optional[Dict[str, List[int]]] = None,
    ):
        self._core, self._path, self._device = core, path, device
        self._compile_config, self._prepare = make_compile_config(num_threads), prepare
        self._dynamic: Optional[ov.CompiledModel] = None
        self._static: Optional[ov.CompiledModel] = None
        self._static_dims: List[Optional[List[int]]] = []

        if static_shapes is None:
            self._dynamic, self._model = self._compile(None)
            primary = self._dynamic
        else:
            self._static, self._model = self._compile(static_shapes)
            primary = self._static
            # 入力は複数の名前を持つことがある（例: image_embed と内部名）ので、名前の集合で照合する
            port_names = [port.get_names() for port in primary.inputs]
            unknown = set(static_shapes) - set().union(*port_names)
            if unknown:
                raise ValueError(f"{path.name} に無い入力です: {sorted(unknown)}（入力: {port_names}）")
            self._static_dims = [
                next((dims for name, dims in static_shapes.items() if name in names), None) for names in port_names
            ]
        self._output_names = [port.get_names() for port in primary.outputs]
        self._primary = primary

    def _compile(self, static_shapes: Optional[Dict[str, List[int]]]) -> Tuple[ov.CompiledModel, ov.Model]:
        model = self._core.read_model(self._path)
        if self._prepare is not None:
            self._prepare(model)
        if static_shapes is not None:
            model.reshape(static_shapes)
        return self._core.compile_model(model, self._device, self._compile_config), model

    @property
    def rt_info(self):
        """モデルのメタデータ（変換時に書き込んだ patch_size など）。"""
        return self._model.rt_info

    @property
    def output_shapes(self) -> List[Tuple[int, ...]]:
        """出力の形。静的形状のモデルのときだけ使える（動的形状の次元があると失敗する）。"""
        return [tuple(port.get_partial_shape().to_shape()) for port in self._primary.outputs]

    def output_index(self, name: str) -> int:
        """名前の付いた出力の、出力の順での位置。"""
        for index, names in enumerate(self._output_names):
            if name in names:
                return index
        raise KeyError(f"{self._path.name} に出力 {name!r} はありません: {self._output_names}")

    def _fits_static(self, inputs: Sequence[np.ndarray]) -> bool:
        """すべての入力が、固定した形（-1 の次元は問わない）に合うか。"""
        for array, dims in zip(inputs, self._static_dims):
            if dims is None:
                continue
            if array.ndim != len(dims) or any(d != -1 and d != n for d, n in zip(dims, array.shape)):
                return False
        return True

    def __call__(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
        if self._static is not None and self._fits_static(inputs):
            compiled = self._static
        else:
            if self._dynamic is None:
                self._dynamic, _ = self._compile(None)
            compiled = self._dynamic
        outs = compiled(list(inputs))
        return [outs[index] for index in range(len(outs))]
