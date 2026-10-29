"""OpenVINO backend（PE / EdgeTAM）に共通の準備。

このモジュールは import するだけで、システム版 OpenVINO（/opt/intel/openvino）を sys.path / PYTHONPATH から外し、NumPy の BLAS のスレッド数を制限する。
pip 版 openvino との衝突を避けるため、各 backend は `import openvino` より前にこのモジュールを import すること。
"""

import os
import sys
from typing import Dict

from core.common.params import BLAS_NUM_THREADS, OPENCV_NUM_THREADS, SYSTEM_OPENVINO_DIR


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


def limit_cpu_threads() -> None:
    """OpenCV と NumPy の BLAS のスレッド数を制限する（OpenVINO のスレッドとコアを取り合って、推論が遅くなるのを避ける）。"""
    import cv2  # cv2 は NumPy を読み込むので、モジュールの import 時に BLAS の環境変数を設定し終えるまで読まない

    cv2.setNumThreads(OPENCV_NUM_THREADS)
    limit_blas_threads()


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
