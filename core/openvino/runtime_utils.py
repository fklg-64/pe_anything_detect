"""OpenVINO backend（PE / EdgeTAM）に共通の準備。

このモジュールは import するだけで、システム版 OpenVINO（/opt/intel/openvino）を sys.path / PYTHONPATH から外す。
pip 版 openvino との衝突を避けるため、各 backend は `import openvino` より前にこのモジュールを import すること。
"""

import os
import sys
from typing import Dict

SYSTEM_OPENVINO_DIR = "/opt/intel/openvino"


def _remove_system_openvino_from_path() -> None:
    sys.path = [p for p in sys.path if not p.startswith(SYSTEM_OPENVINO_DIR)]

    # 子プロセスなどに引き継がれないよう、環境変数からも削除しておく
    if "PYTHONPATH" in os.environ:
        paths = os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in paths if not p.startswith(SYSTEM_OPENVINO_DIR))


_remove_system_openvino_from_path()


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
