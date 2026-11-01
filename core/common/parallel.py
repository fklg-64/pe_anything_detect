"""モデルの推論と NumPy / OpenCV の整形を重ねるための、小さなスレッド部品。

OpenVINO・PyTorch の推論は GIL を離すので、推論の最中に別スレッドで NumPy の整形を進められる。
このパイプラインでスレッドに載せる仕事は 3 つ（どれも、結果を取り出す順を呼び出し側が決めるので、出力はスレッドの有無で変わらない）。

  - EdgeTAM の特徴の kNN グラフ作り      PE の推論と重ねる           （SegmentationRunner）
  - 領域（クラス・取りこぼしの成分）の推定  次の領域のデコードと重ねる  （SemanticRefiner）
  - クラスごとの小さな成分の検出          クラス間で重ねる            （mask_utils.remove_small_regions）

worker_threads が 0 のときは、スレッドを使わず、結果を取り出すとき（result()）にその場で実行する。
"""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Generic, List, Optional, TypeVar, Union

T = TypeVar("T")


class _InlineJob(Generic[T]):
    """スレッドを使わないときの仕事。result() を呼んだときに実行する。Future と同じ使い方ができる。"""

    def __init__(self, function: Callable[[], T]):
        self._function = function

    def result(self) -> T:
        return self._function()

    def cancel(self) -> bool:
        return True


Job = Union[Future, _InlineJob]


class Workers:
    """仕事を投入するスレッドプール。num_threads が 0 なら、スレッドを使わない（仕事は result() でその場で実行する）。"""

    def __init__(self, num_threads: int):
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="segment") if num_threads > 0 else None
        )

    def submit(self, function: Callable[[], T]) -> Job:
        return self._executor.submit(function) if self._executor is not None else _InlineJob(function)


class JobGroup:
    """仕事をまとめて投入し、`with` を抜けるとき（例外も含む）に、まだ始まっていない仕事を取り消す。"""

    def __init__(self, workers: Workers):
        self._workers = workers
        self._jobs: List[Job] = []

    def submit(self, function: Callable[[], T]) -> Job:
        job = self._workers.submit(function)
        self._jobs.append(job)
        return job

    def __enter__(self) -> "JobGroup":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for job in self._jobs:
            job.cancel()
