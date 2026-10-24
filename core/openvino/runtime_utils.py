"""OpenVINO backend（PE / EdgeTAM）に共通の準備。

このモジュールは import するだけで、システム版 OpenVINO（/opt/intel/openvino）を sys.path / PYTHONPATH から外し、NumPy の BLAS のスレッド数を制限する。
pip 版 openvino との衝突を避けるため、各 backend は `import openvino` より前にこのモジュールを import すること。
"""

import os
import sys
from typing import Dict

SYSTEM_OPENVINO_DIR = "/opt/intel/openvino"

# NumPy の BLAS（OpenBLAS）のスレッド数。BLAS は既定で論理コア数（20）のスレッドを起こし、行列積（`@`）の後もしばらくスピン待ちするので、
# OpenVINO のスレッド（ピン止め）とコアを取り合い、直後の推論が 2〜4 倍遅くなる（行列積 2ms の後のデコードが 7ms → 29ms）。
# 本パイプラインの行列積は小さい（1024×256）ので、1 スレッドで 4ms 程度で済み、結果も同じ
BLAS_NUM_THREADS = 1


def _remove_system_openvino_from_path() -> None:
    sys.path = [p for p in sys.path if not p.startswith(SYSTEM_OPENVINO_DIR)]

    # 子プロセスなどに引き継がれないよう、環境変数からも削除しておく
    if "PYTHONPATH" in os.environ:
        paths = os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in paths if not p.startswith(SYSTEM_OPENVINO_DIR))


def limit_blas_threads(num_threads: int = BLAS_NUM_THREADS) -> None:
    """NumPy の BLAS のスレッド数を制限する。

    環境変数は NumPy を import する前でないと効かないので、import 済みのライブラリには threadpoolctl で設定する
    （scikit-learn の依存で入っている。無ければ環境変数だけ）。
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=num_threads, user_api="blas")


_remove_system_openvino_from_path()
limit_blas_threads()  # NumPy を import する前なら環境変数が効く。各 backend の初期化でも呼び直す（後から読み込まれたライブラリ向け）


def make_compile_config(num_threads: int) -> Dict[str, str]:
    """core.compile_model に渡す実行時の設定。低遅延（1枚ずつ処理）向け。"""
    return {
        "INFERENCE_NUM_THREADS": str(num_threads),
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": "YES",
        "NUM_STREAMS": "1",
    }


def model_file_suffix(use_int8: bool) -> str:
    """モデルファイル名の末尾。INT8 量子化モデルは `_int8.xml`、そうでなければ `.xml`。"""
    return "_int8.xml" if use_int8 else ".xml"
