"""密集した長方形の物体の分離の説明資料（docs/rect_instances.md）用の画像を作る: package の実画像をパイプライン（dense_instances = true）に通し、
結果のインスタンスと、大きさを決めた縁の間隔のヒストグラムを 1 枚にして docs/images/ に保存する。

    python tools/docs/make_rect_doc_image.py [--images "package (2).png" "package (1)_30.png"] [--out-dir docs/images] [--name rect_instances]

行ごとに、左 = インスタンス（色 + 境界線。ResultSaver の保存と同じ）、中 = S 用（縦 33 画素続く縁）のヒストグラムと選んだ S、
右 = L 用（縦 9 画素）のヒストグラムと、ヒストグラムの L の候補・敷き詰めの質で直した後の L。OpenVINO 版（configs/ov_config.yaml）で実行する。
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.openvino.runtime_utils import make_compile_config  # noqa: F401  (import openvino より前に /opt/intel/openvino を外す)
from core.common.params import RECT_HISTOGRAM_SMOOTH_SIGMA, RECT_IMAGE_SIZE, RECT_LONG_EDGE_LINE, RECT_LONG_GAP_POWER, RECT_LONG_MIN_RATIO, RECT_SHORT_EDGE_LINE
from core.common.rect_size import estimate_gaps, gap_histograms
from core.common.segmentation_runner import SegmentationRunner
from core.common.visualization import ResultSaver
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_image, resize_image
from tools.docs.make_doc_images import save_figure

PACKAGE_DIR = Path("/home/llm/datasets/02_object_detection/package")
DEFAULT_IMAGES = ["package (2).png", "package (1)_30.png"]
BLUE, ORANGE, GREEN = "#2a6df4", "#e8730c", "#1a9c6b"
MAX_GAP_SHOWN = 300


def smoothed(histogram: np.ndarray) -> np.ndarray:
    """本体（estimate_gaps）と同じ平滑化。最大値 1 に正規化する。"""
    h = cv2.GaussianBlur(histogram.astype(np.float32).reshape(1, -1), (0, 0), RECT_HISTOGRAM_SMOOTH_SIGMA).ravel()
    return h / max(float(h.max()), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=DEFAULT_IMAGES, help=f"{PACKAGE_DIR} の中のファイル名")
    ap.add_argument("--out-dir", default="docs/images")
    ap.add_argument("--name", default="rect_instances")
    args = ap.parse_args()

    config = load_config("./configs/ov_config.yaml")
    config["dense_instances"] = True
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    saver = ResultSaver(config)

    fig, axes = plt.subplots(len(args.images), 3, figsize=(13.5, 3.9 * len(args.images)), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]}, squeeze=False)
    for row, name in zip(axes, args.images):
        image = load_image(PACKAGE_DIR / name)
        result = runner.segment(image)
        if result.instances is None:
            raise SystemExit(f"{name}: 大きさが求まらなかった")
        with tempfile.TemporaryDirectory() as tmp:
            saver._save_instances(Path(tmp), image, result.instances.instance_map)
            overlay = cv2.cvtColor(cv2.imread(str(Path(tmp) / "rect_instance_overlay.png")), cv2.COLOR_BGR2RGB)

        small = resize_image(image, RECT_IMAGE_SIZE)
        foreground = cv2.resize(result.semantic_map, (RECT_IMAGE_SIZE, RECT_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) > 0
        short_hist, long_hist = (smoothed(h) for h in gap_histograms(small, foreground, (RECT_SHORT_EDGE_LINE, RECT_LONG_EDGE_LINE)))
        candidate, _ = estimate_gaps(small, foreground)
        final_length, width = result.instances.size

        row[0].imshow(overlay)
        row[0].set_title(f"{name[:-4]}: {len(result.instances.scores)} 個", fontsize=11)
        row[0].axis("off")
        x = np.arange(len(short_hist))
        row[1].plot(x, short_hist, color=BLUE, lw=1.2)
        row[1].axvline(width, color=ORANGE, lw=1.4, label=f"S = {width:.0f}")
        row[1].set_title(f"S 用: 縦 {RECT_SHORT_EDGE_LINE} 画素続く縁の間隔", fontsize=10)
        weighted = long_hist * x ** RECT_LONG_GAP_POWER       # L を選ぶ基準（S の RECT_LONG_MIN_RATIO 倍より長い間隔の中で最大のもの）
        weighted = weighted / max(float(weighted[int(RECT_LONG_MIN_RATIO * width):].max()), 1e-6)
        row[2].plot(x, weighted, color=BLUE, lw=1.2)
        row[2].axvspan(0, RECT_LONG_MIN_RATIO * width, color="#999999", alpha=0.15, lw=0)
        row[2].axvline(candidate, color=GREEN, lw=1.4, ls="--", label=f"ヒストグラムの候補 {candidate:.0f}")
        if final_length != candidate:
            row[2].axvline(final_length, color=ORANGE, lw=1.4, label=f"敷き詰めの質で直した L = {final_length:.0f}")
        else:
            row[2].axvline(final_length, color=ORANGE, lw=1.4, ls=(0, (1, 2)), label=f"L = {final_length:.0f}（直さない）")
        row[2].set_title(f"L 用: 縦 {RECT_LONG_EDGE_LINE} 画素続く縁の間隔 × 間隔²（灰色は S の {RECT_LONG_MIN_RATIO} 倍以下）", fontsize=10)
        for ax, limit in ((row[1], 150), (row[2], MAX_GAP_SHOWN)):
            ax.set_xlim(0, limit)
            ax.set_ylim(-0.03, 1.08 if ax is row[1] else 1.25)
            ax.set_xlabel("隣り合う平行な縁の間隔（画素。作業解像度 512）", fontsize=9)
            ax.legend(loc="upper right", fontsize=9, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(args.out_dir) / f"{args.name}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path, bbox_inches="tight")
    plt.close(fig)
    print("保存:", path)


if __name__ == "__main__":
    main()
