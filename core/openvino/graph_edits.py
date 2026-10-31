"""変換済み OpenVINO モデルに、コンパイルの前に施すグラフの書き換え（モデルのファイルは変えない）。

どの関数も、静的形状に reshape した後のモデルを in-place で書き換える。出力は書き換え前と同じ値（浮動小数の丸めまで一致する）。
想定したグラフと違っていれば何も変えず False を返す（変換し直したモデルでも、遅くなるだけで動く）。
"""

import core.openvino.runtime_utils  # noqa: F401  openvino より先に import する（sys.path の整理。runtime_utils.py 参照）

import numpy as np
import openvino as ov
import openvino.opset13 as ops


def _constant_values(output: ov.Output) -> list | None:
    node = output.get_node()
    return node.get_data().tolist() if node.get_type_name() == "Constant" else None


def remove_decoder_output_reorder(model: ov.Model) -> bool:
    """デコーダーの最後の matmul を、チャネルが末尾の（NHWC の）まま行うようにつなぎ替える。

    元のグラフは、最後の Gelu の出力（畳み込みの出力なので CPU プラグインの内部では NHWC）を `[1, C, H*W]` に reshape して
    `hyper_in [1, 4, C]` と掛けるので、NHWC -> NCHW の Reorder（128×128×32 で 1 回約 3〜4ms）が毎回入る。
    これを `Transpose -> Reshape [1, H*W, C] -> MatMul(·, hyper_in[:, 1:], transpose_b)` に置き換えると、Transpose と Reshape が畳み込みの出力の
    レイアウトのまま扱われて Reorder が無くなる。末尾の `Slice`（マスクの候補 1~3 だけを返す）は、掛ける前の hyper_in に移すので、計算も 1/4 減る。
    バッチは 1 のモデルだけが対象（静的形状のデコーダー）。動的形状のモデル（格子分離のセルごとのデコード）には適用しない。

    元: Gelu [1,C,H,W] -> Reshape [1,C,H*W] -> MatMul(hyper_in [1,4,C], ·) -> Reshape [1,4,H,W] -> Slice(axis 1, 1:) -> Result
    """
    result = model.get_results()[0]
    slice_node = result.input_value(0).get_node()
    if slice_node.get_type_name() != "Slice" or slice_node.get_input_size() != 5:
        return False
    start, stop, step, axes = (_constant_values(slice_node.input_value(i)) for i in range(1, 5))
    if start != [1] or step != [1] or axes != [1] or stop is None or stop[0] < 4:
        return False
    reshape_out = slice_node.input_value(0).get_node()
    if reshape_out.get_type_name() != "Reshape":
        return False
    matmul = reshape_out.input_value(0).get_node()
    if matmul.get_type_name() != "MatMul" or matmul.get_attributes() != {"transpose_a": False, "transpose_b": False}:
        return False
    hyper_in = matmul.input_value(0)
    flatten = matmul.input_value(1).get_node()
    if flatten.get_type_name() != "Reshape":
        return False
    upscaled = flatten.input_value(0)  # [1, C, H, W]
    shape = upscaled.get_partial_shape()
    if shape.is_dynamic or shape.rank.get_length() != 4 or shape.to_shape()[0] != 1:
        return False
    _, channels, height, width = shape.to_shape()
    hyper_shape = hyper_in.get_partial_shape()
    if hyper_shape.is_dynamic or list(hyper_shape.to_shape()) != [1, 4, channels]:
        return False

    nhwc = ops.transpose(upscaled, ops.constant(np.array([0, 2, 3, 1], dtype=np.int64)))
    flat = ops.reshape(nhwc, ops.constant(np.array([1, height * width, channels], dtype=np.int64)), special_zero=False)
    candidates = ops.slice(
        hyper_in, ops.constant(np.array([1], dtype=np.int64)), ops.constant(np.array([4], dtype=np.int64)),
        ops.constant(np.array([1], dtype=np.int64)), ops.constant(np.array([1], dtype=np.int64)),
    )
    product = ops.matmul(flat, candidates, transpose_a=False, transpose_b=True)  # [1, H*W, 3]
    masks = ops.reshape(
        ops.transpose(product, ops.constant(np.array([0, 2, 1], dtype=np.int64))),
        ops.constant(np.array([1, 3, height, width], dtype=np.int64)), special_zero=False,
    )
    result.input(0).replace_source_output(masks.output(0))
    model.validate_nodes_and_infer_types()
    return True


# 4 次元の特徴量の軸の並びの変換（NCHW <-> NHWC）。OVModel の static_input_perms / static_output_perms に渡す
NHWC_FROM_NCHW = (0, 2, 3, 1)
NCHW_FROM_NHWC = (0, 3, 1, 2)


def encoder_feats0_as_nhwc(model: ov.Model) -> bool:
    """エンコーダーの出力 0（high_res_feats0 [1, C, H, W]）を NHWC で出す（出力の直前に Transpose を足す）。

    畳み込みの出力は CPU プラグインの内部では NHWC なので、NCHW で出すと Reorder が入る（約 2〜3ms）。NHWC のまま出せば無い。
    出力の形が [1, H, W, C] に変わるので、呼び出し側は OVModel の static_output_perms で NCHW のビューに戻す。
    """
    result = model.get_results()[0]
    source = result.input_value(0)
    shape = source.get_partial_shape()
    if shape.is_dynamic or shape.rank.get_length() != 4 or shape.to_shape()[0] != 1:
        return False
    result.input(0).replace_source_output(ops.transpose(source, ops.constant(np.array(NHWC_FROM_NCHW, dtype=np.int64))).output(0))
    model.validate_nodes_and_infer_types()
    return True


def decoder_feats0_from_nhwc(model: ov.Model) -> bool:
    """デコーダーの入力 high_res_feats0 を NHWC `[1, H, W, C]` で受け取る（内部で NCHW に Transpose する）。

    この特徴量は 2 つ目の逆畳み込みの出力に足されるので、NCHW で受けると毎回 NHWC への Reorder が入る（約 4ms）。
    エンコーダーの出力を NHWC で出す（encoder_feats0_as_nhwc）と、Reorder も Transpose のコピーも無い。入力の名前は変えない。
    """
    for index, parameter in enumerate(model.get_parameters()):
        if "high_res_feats0" in parameter.output(0).get_names():
            shape = parameter.get_partial_shape()
            if shape.is_dynamic or shape.rank.get_length() != 4 or shape.to_shape()[0] != 1:
                return False
            _, channels, height, width = shape.to_shape()
            new = ops.parameter([1, height, width, channels], parameter.get_element_type(), name=parameter.get_friendly_name())
            new.output(0).get_tensor().set_names(set(parameter.output(0).get_names()))
            nchw = ops.transpose(new, ops.constant(np.array(NCHW_FROM_NHWC, dtype=np.int64)))
            parameter.output(0).replace(nchw.output(0))
            model.replace_parameter(index, new)
            model.validate_nodes_and_infer_types()
            return True
    return False
