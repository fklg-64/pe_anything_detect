"""処理のタイムライン図（docs/overview.md 用）を作る: 1 枚の画像の処理で、各処理がいつ・どのスレッドで走ったかを並べる。

    python tools/docs/make_timeline_image.py [--image PATH] [--out docs/images/timeline.jpg] [--model-dir DIR]

Timer の開始・終了と、スレッドに載せる 3 つの仕事（kNN グラフ・領域の推定・小領域の検出）を記録して描く。
同じ画像を 2 回流してウォームアップし、続く 5 回のうち合計時間が中央値の回の記録を使う。OpenVINO 版（configs/ov_config.yaml）で実行する。
"""
import argparse
import contextlib
import io
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.common import segmentation_runner
from core.common.detail import mask_cleanup
from core.common.detail.region_estimator import RegionEstimator
from core.common.segmentation_runner import SegmentationRunner
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import Timer, load_config, load_dataset_paths, load_image
from tools.docs.make_doc_images import save_figure  # noqa: F401  (日本語フォントの設定もここで入る)

EVENTS: list = []  # (名前, スレッド名, 開始, 終了)
LOCK = threading.Lock()


def record(name: str, start: float, end: float) -> None:
    with LOCK:
        EVENTS.append((name, threading.current_thread().name, start, end))


def instrument() -> None:
    """Timer と、スレッドに載せる仕事を、開始・終了つきで記録するように差し替える。"""
    original_enter, original_exit = Timer.__enter__, Timer.__exit__

    def enter(self):
        self._t0 = time.perf_counter()
        return original_enter(self)

    def exit_(self, *args):
        result = original_exit(self, *args)
        record(self.name.strip().lstrip("┌ "), self._t0, time.perf_counter())
        return result

    Timer.__enter__, Timer.__exit__ = enter, exit_

    def timed(function, label):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record(label, start, time.perf_counter())
        return wrapper

    segmentation_runner.edge_graph_of = timed(segmentation_runner.edge_graph_of, "kNN グラフ")  # SegmentationRunner._start_edge_graph が呼ぶ名前
    RegionEstimator.estimate = timed(RegionEstimator.estimate, "領域の推定")
    mask_cleanup._small_pieces = timed(mask_cleanup._small_pieces, "小領域の検出")


# 帯の色: 種類ごと
COLORS = {"デコード": "#d62728", "PE": "#1f77b4", "EdgeTAM": "#9467bd", "整形": "#2ca02c", "その他": "#7f7f7f"}


def kind(name: str) -> str:
    if name == "デコード":
        return "デコード"
    if name in ("特徴量抽出処理時間",):
        return "PE"
    if "EdgeTAM特徴量抽出" in name or "edgetam特徴量抽出" in name:
        return "EdgeTAM"
    if name in ("kNN グラフ", "領域の推定", "小領域の検出"):
        return "整形"
    return "その他"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="既定は configs/dataset.yaml の先頭の画像")
    parser.add_argument("--out", default="docs/images/timeline.jpg")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    config = load_config()
    if args.model_dir:
        config["model_dir"] = args.model_dir
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    image = load_image(args.image or load_dataset_paths()[0])
    with contextlib.redirect_stdout(io.StringIO()):
        runner.segment(image)
        runner.segment(image)
    instrument()
    runs = []  # (合計 ms, 記録)
    for _ in range(5):
        EVENTS.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            runner.segment(image)
            runs.append(((time.perf_counter() - start) * 1000, list(EVENTS)))
    runs.sort(key=lambda run: run[0])
    total, events = runs[len(runs) // 2]  # 合計時間が中央値の回を使う（外れ値の回を避ける）
    EVENTS[:] = events

    # 描くもの: 主要な段階（メインスレッド）と、スレッドに載せた仕事・デコード
    stages = ["画像前処理時間", "EdgeTAM特徴量抽出処理時間", "特徴量抽出処理時間", "粗い前景マスク取得処理時間", "詳細セマンティックセグメンテーション処理時間"]
    labels = {
        "画像前処理時間": "前処理", "EdgeTAM特徴量抽出処理時間": "EdgeTAM\nエンコーダ", "特徴量抽出処理時間": "PE",
        "粗い前景マスク取得処理時間": "粗い前景", "詳細セマンティックセグメンテーション処理時間": "精緻化（クラス・取りこぼし・後処理）",
    }
    lanes = ["メイン: 段階", "ワーカー 1", "ワーカー 2", "デコーダー（同時に 1 つ）"]
    t0 = min(e[2] for e in EVENTS)
    workers = sorted({e[1] for e in EVENTS if e[1].startswith("segment")})
    rows = []  # (レーン番号, 名前, 開始 ms, 終了 ms, 種類)
    for name, thread, a, b in EVENTS:
        clean = name.strip()
        if clean.endswith("回目マスク推定") or clean.startswith("リサイズ") or clean.startswith("格子"):
            continue
        if clean == "デコード":
            rows.append((3, "デコード", (a - t0) * 1000, (b - t0) * 1000, "デコード"))
        elif thread.startswith("segment") and clean in ("kNN グラフ", "領域の推定", "小領域の検出"):
            lane = 1 + min(workers.index(thread), 1)
            rows.append((lane, clean, (a - t0) * 1000, (b - t0) * 1000, "整形"))
        elif thread == "MainThread" and clean in stages:
            rows.append((0, labels[clean], (a - t0) * 1000, (b - t0) * 1000, kind(clean)))

    print(f"合計 {total:.0f} ms。段階（メインスレッド）の開始と所要時間:")
    for lane, name, a, b, _ in sorted(rows, key=lambda row: row[2]):
        if lane == 0:
            print(f"  {name.replace(chr(10), ' '):36s} 開始 {a:7.1f} ms  所要 {b - a:7.1f} ms")
    for lane, label in ((1, "ワーカー 1"), (2, "ワーカー 2"), (3, "デコーダー")):
        busy = sum(b - a for l, _, a, b, _ in rows if l == lane)
        count = sum(1 for l, *_ in rows if l == lane)
        print(f"  {label}: {count} 件、合計 {busy:.1f} ms")

    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    for lane, name, a, b, k in rows:
        ax.barh(lane, b - a, left=a, height=0.62, color=COLORS[k], edgecolor="white", linewidth=0.6)
        if b - a > total * 0.06:
            ax.text((a + b) / 2, lane, name, ha="center", va="center", fontsize=8, color="white")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes)
    ax.invert_yaxis()
    ax.set_xlabel("経過時間 (ms)")
    ax.set_xlim(0, total)
    ax.set_title(f"1 枚の処理のタイムライン（合計 {total:.0f} ms）。ワーカーは PE や次のデコードと重なって走る", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, COLORS.keys(), loc="upper right", fontsize=8, ncol=5, bbox_to_anchor=(1.0, 1.28), frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, bbox_inches="tight")
    print("保存:", out, f"（合計 {total:.0f} ms、記録 {len(EVENTS)} 件）")


if __name__ == "__main__":
    main()
