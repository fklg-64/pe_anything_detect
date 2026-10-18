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

--gt-dir を指定すると、MVTec の代わりに任意の画像と正解マスクで評価する（小さな物体の見逃しを測る用途）。

    python tools/eval_prompts.py --gt-dir ./datasets/develop_gt

    <gt-dir>/images/<名前>.png|jpg  元画像
    <gt-dir>/masks/<名前>.png       正解マスク（0=背景）。物体ごとに別の値で塗ると物体単位で数える。
                                    1色（0/255など）だけなら、連結成分を1物体として数える。
指標（--gt-dir）:
    IoU      : 前景(>0)の IoU
    recall   : 物体の検出率。正解の物体の面積の半分以上が予測の前景に入っていれば検出
    recall_S/M/L: 物体の大きさ別の検出率（画像面積に対して S<0.5%, M<5%, L>=5%）
    fp_area  : 正解の前景の外を前景と予測した面積（画像面積に対する比）
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config, load_image

# fg_mask が全面前景になるテクスチャ系カテゴリは対象外
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]


def segment(runner: SegmentationRunner, image_path: str):
    """SegmentationRunner.segment の結果（保存なし）。(semantic_map, prompt_indices) を返す。"""
    image = load_image(image_path)
    with contextlib.redirect_stdout(io.StringIO()):  # Timerの出力を抑える
        result = runner.segment(image)
    return result.semantic_map, result.prompt_indices


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


# 物体の大きさの区分（画像面積に対する比）
SIZE_BINS = (("S", 0.005), ("M", 0.05), ("L", float("inf")))
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def gt_objects(mask: np.ndarray):
    """正解マスクを物体ごとの2値マスクのリストにする。"""
    values = [v for v in np.unique(mask) if v != 0]
    if len(values) == 1:
        n, comp = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        return [comp == i for i in range(1, n)]
    return [mask == v for v in values]


def evaluate_gt_dir(runner, gt_dir: Path):
    """画像ごとの指標と、物体ごとの検出結果を返す。"""
    images, objects = [], []
    for mask_path in sorted((gt_dir / "masks").glob("*.png")):
        image_path = next((gt_dir / "images" / f"{mask_path.stem}{s}" for s in IMAGE_SUFFIXES
                           if (gt_dir / "images" / f"{mask_path.stem}{s}").exists()), None)
        if image_path is None:
            print(f"画像が見つかりません: {mask_path.stem}")
            continue
        semantic_map, _ = segment(runner, str(image_path))
        mask = np.array(Image.open(mask_path))
        mask = mask[..., 0] if mask.ndim == 3 else mask
        mask = cv2.resize(mask, semantic_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
        pred, gt = semantic_map > 0, mask > 0
        for obj in gt_objects(mask):
            ratio = obj.sum() / obj.size
            size = next(name for name, upper in SIZE_BINS if ratio < upper)
            objects.append({"image": mask_path.stem, "size": size, "detected": (pred & obj).sum() >= 0.5 * obj.sum()})
        detected = [o["detected"] for o in objects if o["image"] == mask_path.stem]
        images.append({
            "image": mask_path.stem,
            "iou": (pred & gt).sum() / max((pred | gt).sum(), 1),
            "recall": float(np.mean(detected)) if detected else 1.0,
            "fp_area": (pred & ~gt).sum() / pred.size,
        })
    return images, objects


def report_gt_dir(images, objects):
    print(f"{'画像':<36}{'IoU':>7}{'recall':>8}{'fp_area':>9}")
    for r in images:
        print(f"{r['image'][:35]:<36}{r['iou']:>7.3f}{r['recall']:>8.2f}{r['fp_area']:>9.3f}")
    line = (f"{'平均':<36}{np.mean([r['iou'] for r in images]):>7.3f}"
            f"{np.mean([o['detected'] for o in objects]) if objects else 1.0:>8.2f}"
            f"{np.mean([r['fp_area'] for r in images]):>9.3f}")
    print(line)
    for name, _ in SIZE_BINS:
        hits = [o["detected"] for o in objects if o["size"] == name]
        print(f"recall_{name}: {np.mean(hits):.2f} ({sum(hits)}/{len(hits)}物体)" if hits else f"recall_{name}: -")


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
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="configの値を上書き（例: k_clusters=12）")
    parser.add_argument("--gt-dir", default=None, help="任意の画像と正解マスクで評価する（images/ と masks/ を持つディレクトリ）")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompts_dir:
        config["prompts_dir"] = args.prompts_dir
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = type(config[key])(yaml.safe_load(value))
    with contextlib.redirect_stdout(io.StringIO()):
        runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    print(f"fg={len(runner.foreground_prompts)}個, bg={len(runner.background_prompts)}個")

    if args.gt_dir:
        report_gt_dir(*evaluate_gt_dir(runner, Path(args.gt_dir)))
        sys.exit(0)

    for split, indices in (("dev", [i * 7 for i in range(args.num)]), ("holdout", [3 + i * 11 for i in range(args.num)])):
        total, per_category = summarize(evaluate(runner, Path(args.mvtec), indices))
        print(f"{split:8s} " + " ".join(f"{k}={v:.3f}" for k, v in total.items()))
        print("         " + " ".join(f"{c}={v:.2f}" for c, v in per_category.items()))
