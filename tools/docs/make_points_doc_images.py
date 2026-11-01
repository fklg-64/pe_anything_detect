"""前景のプロンプト点の選び方の説明資料（docs/detailed_segmentation.md §2）用の画像を作る: 従来（パッチの中心）と現在（EdgeTAM の高解像度特徴量の画素）で、
点・最終結果・正解の輪郭を並べる。MVTec の画像（正解 = fg_mask）を使う。

    python tools/docs/make_points_doc_images.py [--images toothbrush/035 toothbrush/005] [--out docs/images/points_toothbrush.jpg]

従来の選び方は core/common/mask_utils.prompt_points（RegionEstimator.foreground_points を差し替えて再現する）。OpenVINO 版（configs/ov_config.yaml）で実行する。
MVTec の場所は --mvtec（既定: E:/WPy64-31241/datasets/mvtec_anomaly_detection）。正解は MVTec 配布物に無い fg_mask/<カテゴリ>/<番号>.png。
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.openvino.runtime_utils import make_compile_config  # noqa: F401  (import openvino より前に /opt/intel/openvino を外す)
from core.common.mask_utils import prompt_points
from core.common.segmentation_runner import SegmentationRunner
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_image
from tools.docs.make_doc_images import DISPLAY, outline, overlay, points_on, save_panels

MVTEC = Path("E:/WPy64-31241/datasets/mvtec_anomaly_detection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=["toothbrush/035", "toothbrush/005"], help="MVTec のカテゴリ/番号")
    parser.add_argument("--out", default="docs/images/points_toothbrush.jpg")
    parser.add_argument("--mvtec", default=str(MVTEC), help="MVTec のフォルダ（正解 fg_mask/ を含む）")
    args = parser.parse_args()
    mvtec = Path(args.mvtec)

    config = load_config()
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    estimator = runner.refiner.estimator
    current_points = estimator.foreground_points

    def legacy_points(inputs, coarse_mask, comps, core):
        """従来の選び方: core 上のパッチの中心から、各成分の最内点を含めて最遠点で選ぶ。"""
        return prompt_points(core, [c.peak for c in comps], estimator.foreground_points_per_class)

    panels = []
    for name in args.images:
        category, index = name.split("/")
        image = load_image(mvtec / category / "train" / "good" / f"{index}.png")
        gt = np.array(Image.open(mvtec / "fg_mask" / category / f"{index}.png"))
        gt = (gt[..., 0] if gt.ndim == 3 else gt) > 0
        base = cv2.resize(image, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)
        scale = DISPLAY / max(image.shape[:2])
        gt_display = cv2.resize(gt.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
        for points, label in ((legacy_points, "従来（パッチの中心）"), (current_points, "現在（高解像度の画素）")):
            estimator.foreground_points = points
            result = runner.segment(image)
            predicted = result.semantic_map > 0
            truth = cv2.resize(gt.astype(np.uint8), predicted.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            iou = (predicted & truth).sum() / max((predicted | truth).sum(), 1)
            panel = overlay(base, cv2.resize(predicted.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), (255, 0, 0), 0.45)
            panel = outline(panel, gt_display, (255, 255, 0), 2)
            panel = points_on(panel, result.point_coords, result.point_labels, scale, radius=6)
            panels.append((f"{category} {index}  {label}\nIoU {iou:.3f}", panel))
            print(f"{name}  {label}  点 {len(result.point_coords)}  IoU {iou:.3f}")
    del estimator.foreground_points  # インスタンスの差し替えを外す
    save_panels(Path(args.out), panels, ncols=4)


if __name__ == "__main__":
    main()
