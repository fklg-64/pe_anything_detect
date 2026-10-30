"""前景のプロンプト点の選び方の説明資料（docs/detailed_segmentation.md §2）用の画像を作る: 従来（パッチの中心）と現在（EdgeTAM の高解像度特徴量の画素）で、
点・最終結果・正解の輪郭を並べる。MVTec の画像（正解 = fg_mask）を使う。

    python tools/docs/make_points_doc_images.py [--images toothbrush/035 toothbrush/005] [--out docs/images/points_toothbrush.jpg]

従来の選び方は SemanticRefiner.use_object_points = False（core/common/mask_utils.prompt_points）。OpenVINO 版（configs/ov_config.yaml）で実行する。
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
from core.common.segmentation_runner import SegmentationRunner
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_image
from tools.docs.make_doc_images import DISPLAY, outline, overlay, points_on, save_panels

MVTEC = Path("/home/llm/datasets/mvtec_anomaly_detection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=["toothbrush/035", "toothbrush/005"], help="MVTec のカテゴリ/番号")
    parser.add_argument("--out", default="docs/images/points_toothbrush.jpg")
    args = parser.parse_args()

    config = load_config()
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    panels = []
    for name in args.images:
        category, index = name.split("/")
        image = load_image(MVTEC / category / "train" / "good" / f"{index}.png")
        gt = np.array(Image.open(MVTEC / "fg_mask" / category / f"{index}.png"))
        gt = (gt[..., 0] if gt.ndim == 3 else gt) > 0
        base = cv2.resize(image, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)
        scale = DISPLAY / max(image.shape[:2])
        gt_display = cv2.resize(gt.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
        for use_object_points, label in ((False, "従来（パッチの中心）"), (True, "現在（高解像度の画素）")):
            runner.refiner.use_object_points = use_object_points
            result = runner.segment(image)
            predicted = result.semantic_map > 0
            truth = cv2.resize(gt.astype(np.uint8), predicted.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            iou = (predicted & truth).sum() / max((predicted | truth).sum(), 1)
            panel = overlay(base, cv2.resize(predicted.astype(np.uint8), (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), (255, 0, 0), 0.45)
            panel = outline(panel, gt_display, (255, 255, 0), 2)
            panel = points_on(panel, result.point_coords, result.point_labels, scale, radius=6)
            panels.append((f"{category} {index}  {label}\nIoU {iou:.3f}", panel))
            print(f"{name}  {label}  点 {len(result.point_coords)}  IoU {iou:.3f}")
    save_panels(Path(args.out), panels, ncols=4)


if __name__ == "__main__":
    main()
