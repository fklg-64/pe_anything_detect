"""前景/背景プロンプトの評価スクリプト（OpenVINO版パイプライン）。

MVTec ADのgood画像と fg_mask（物体マスク）を使い、プロンプトを差し替えたときの
前景IoU・クラスの割れやすさを測る。プロンプトのディレクトリを引数で切り替えて比較する。

    python tools/eval_prompts.py                       # configs/ov_config.yaml の prompts_dir を評価
    python tools/eval_prompts.py --prompts-dir ./prompts_candidate

指標:
    IoU   : 最終セマンティックマップの前景(>0)と fg_mask のIoU（物体カテゴリ平均）
    minIoU: 画像ごとのIoUの最小値（検出失敗の有無）
    frag  : 1 - 最大クラスの面積比。単一物体の画像では0に近いほど「同じ物体が別クラスに割れていない」
dev は index=0,7,14.. 、holdout は index=3,14,25.. の画像（プロンプトの当てはめ過ぎの確認用）。
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.segmentation_runner import SegmentationRunner
from core.utils import load_config, load_image, preprocess_image

# fg_mask が全面前景になるテクスチャ系カテゴリは対象外
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]


def segment(runner: SegmentationRunner, image_path: str):
    """SegmentationRunner.run と同じ処理（保存なし）。(semantic_map, prompt_indices) を返す。"""
    image = load_image(image_path)
    height, width = image.shape[:2]
    with contextlib.redirect_stdout(io.StringIO()):  # Timerの出力を抑える
        tensor = preprocess_image(image, runner.extractor.target_size, runner.extractor.mean, runner.extractor.std)
        _, patch, fused = runner._feature_extraction(tensor)
        fg_map, fg_mask, clusters, cluster_index = runner._generate_foreground_map(patch, fused)
        labels_list, labels = runner.semantic_segmentation(clusters, cluster_index)
        fg_mask, labels = runner.refine_coarse_segmentation(labels, fg_mask)
        semantic_map, prompt_indices, _, _ = runner.refine_semantic_segmentation(
            image, fg_map, fg_mask, labels, labels_list, width, height
        )
    return semantic_map, prompt_indices


def evaluate(runner, mvtec: Path, indices):
    results = {}
    for category in OBJECT_CATEGORIES:
        for i in indices:
            name = f"{i:03d}.png"
            semantic_map, prompt_indices = segment(runner, str(mvtec / category / "train" / "good" / name))
            gt = np.array(Image.open(mvtec / "fg_mask" / category / name))
            gt = (gt[..., 0] if gt.ndim == 3 else gt) > 0
            gt = cv2.resize(gt.astype(np.uint8), semantic_map.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            pred = semantic_map > 0
            areas = np.bincount(semantic_map.ravel(), minlength=len(prompt_indices) + 1)[1:]
            results.setdefault(category, []).append({
                "iou": (pred & gt).sum() / max((pred | gt).sum(), 1),
                "frag": 1.0 - areas.max() / areas.sum() if areas.sum() else 1.0,
            })
    return results


def summarize(results):
    ious = [r["iou"] for rs in results.values() for r in rs]
    frags = [r["frag"] for rs in results.values() for r in rs]
    return {"IoU": np.mean(ious), "minIoU": np.min(ious), "frag": np.mean(frags)}, {
        c: float(np.mean([r["iou"] for r in rs])) for c, rs in results.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/ov_config.yaml")
    parser.add_argument("--prompts-dir", default=None, help="foreground.yaml / background.yaml のあるディレクトリ")
    parser.add_argument("--mvtec", default="/home/llm/datasets/mvtec_anomaly_detection")
    parser.add_argument("--num", type=int, default=5, help="カテゴリごとの画像数")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompts_dir:
        config["prompts_dir"] = args.prompts_dir
    with contextlib.redirect_stdout(io.StringIO()):
        runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    print(f"fg={len(runner.foreground_prompts)}個, bg={len(runner.background_prompts)}個")

    for split, indices in (("dev", [i * 7 for i in range(args.num)]), ("holdout", [3 + i * 11 for i in range(args.num)])):
        total, per_category = summarize(evaluate(runner, Path(args.mvtec), indices))
        print(f"{split:8s} " + " ".join(f"{k}={v:.3f}" for k, v in total.items()))
        print("         " + " ".join(f"{c}={v:.2f}" for c, v in per_category.items()))
