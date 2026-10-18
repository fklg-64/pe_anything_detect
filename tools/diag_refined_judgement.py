"""リファイン後のマスクを、粗い段階と同じ方法で前景/背景に再判定したときの効果を診断する。

    python tools/diag_refined_judgement.py                              # MVTec (fg_mask)
    python tools/diag_refined_judgement.py --gt-dir ./datasets/synthetic_v1  # 画像+正解マスク(images/, masks/)

リファイン後のセマンティックマップを、クラスごとの連結成分に分ける。各成分を CLIP のパッチ格子へ落として
extract_cluster_tokens でトークンにし、粗い段階と同じ _classify_cluster で前景/背景を判定する。
判定が「背景」の成分を消したとき、次がどうなるかを見る（実装前の見積もり）。

    正しく消す : 正解の背景が多い成分（正解前景の画素が半分以下）を消した
    誤って消す : 正解の前景が多い成分を消した（前景の見逃しが増える）
    --min-patches 未満の成分は判定せず残す（トークンが不安定なため）。既定は MIN_ISOLATED_PATCHES。

判定を入れる価値があるのは、「正しく消す」の画素数が「誤って消す」を上回り、IoU が上がるとき。
"""
import argparse
import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.common.params import MIN_ISOLATED_PATCHES
from core.common.segmentation_runner import SegmentationRunner
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_image, normalize_image, resize_image
from eval_prompts import OBJECT_CATEGORIES

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
PATCH_BINS = (("1-2パッチ", 3), ("3-9パッチ", 10), ("10パッチ以上", float("inf")))


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    return (mask[..., 0] if mask.ndim == 3 else mask) > 0


def mvtec_samples(mvtec: Path, num: int):
    for category in OBJECT_CATEGORIES:
        for i in [i * 7 for i in range(num)]:
            name = f"{i:03d}.png"
            yield f"{category}/{name}", mvtec / category / "train" / "good" / name, load_mask(mvtec / "fg_mask" / category / name)


def gt_dir_samples(gt_dir: Path):
    for mask_path in sorted((gt_dir / "masks").glob("*.png")):
        image_path = next((gt_dir / "images" / f"{mask_path.stem}{s}" for s in IMAGE_SUFFIXES
                           if (gt_dir / "images" / f"{mask_path.stem}{s}").exists()), None)
        if image_path is not None:
            yield mask_path.stem, image_path, load_mask(mask_path)


def patch_labels(components, size: int) -> np.ndarray:
    """成分ごとに、面積の半分以上が成分に入るパッチへ番号(1~)を振る。それ以外のパッチは 0。"""
    labels = np.zeros(size * size, dtype=np.int64)
    best = np.zeros(size * size, dtype=np.float32)
    for index, comp in enumerate(components, start=1):
        fraction = cv2.resize(comp.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA).reshape(-1)
        take = (fraction > 0.5) & (fraction > best)
        labels[take] = index
        best[take] = fraction[take]
    return labels


def judge_components(runner, image: np.ndarray, components):
    """成分ごとに (パッチ数, 前景と判定したか) を返す。"""
    coarse, extractor = runner.coarse, runner.extractor
    with contextlib.redirect_stdout(io.StringIO()):
        tensor = normalize_image(resize_image(image, extractor.target_size), extractor.mean, extractor.std)
        _, patch, _ = coarse.extract_features(tensor)
        labels = patch_labels(components, coarse.feature_size)
        tokens = extractor.extract_cluster_tokens(patch, labels) if labels.any() else []
    fg_num = len(coarse.foreground_prompts)
    result = []
    for index in range(1, len(components) + 1):
        count = int((labels == index).sum())
        if count == 0 or index >= len(tokens):
            result.append((count, None))
            continue
        _, prompt = coarse._classify_cluster(tokens[index])
        result.append((count, prompt < fg_num))
    return result


def diagnose(runner, samples, min_patches: int):
    rows, images = [], []
    for name, image_path, gt in samples:
        image = load_image(image_path)
        with contextlib.redirect_stdout(io.StringIO()):
            semantic_map = runner.segment(image).semantic_map
        gt = cv2.resize(gt.astype(np.uint8), semantic_map.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0

        components = []
        for class_id in np.unique(semantic_map[semantic_map > 0]):
            n, comp = cv2.connectedComponents((semantic_map == class_id).astype(np.uint8), connectivity=8)
            components += [comp == i for i in range(1, n)]
        verdicts = judge_components(runner, image, components) if components else []

        pred_before = semantic_map > 0
        kept = pred_before.copy()
        for comp, (count, is_fg) in zip(components, verdicts):
            area = int(comp.sum())
            truth_fg = (comp & gt).sum() > 0.5 * area
            dropped = is_fg is False and count >= min_patches
            rows.append({"image": name, "patches": count, "area": area, "truth_fg": truth_fg,
                         "judged_fg": is_fg, "dropped": dropped,
                         "fp_px": int((comp & ~gt).sum()), "tp_px": int((comp & gt).sum())})
            if dropped:
                kept &= ~comp
        iou = lambda pred: (pred & gt).sum() / max((pred | gt).sum(), 1)
        images.append({"image": name, "before": iou(pred_before), "after": iou(kept)})
    return rows, images


def report(rows, images):
    print(f"成分数={len(rows)}  画像数={len(images)}")
    print(f"{'区分':<14}{'成分':>6}{'正解が背景':>10}{'消した':>7}{'正しく消す':>10}{'誤って消す':>10}{'削減FP画素':>12}{'失うTP画素':>12}")
    groups = defaultdict(list)
    for r in rows:
        groups["全体"].append(r)
        groups[next(name for name, upper in PATCH_BINS if r["patches"] < upper)].append(r)
    for name in ["全体"] + [n for n, _ in PATCH_BINS]:
        rs = groups.get(name, [])
        if not rs:
            continue
        good = [r for r in rs if r["dropped"] and not r["truth_fg"]]
        bad = [r for r in rs if r["dropped"] and r["truth_fg"]]
        print(f"{name:<14}{len(rs):>6}{sum(not r['truth_fg'] for r in rs):>10}{sum(r['dropped'] for r in rs):>7}"
              f"{len(good):>10}{len(bad):>10}{sum(r['fp_px'] for r in good):>12}{sum(r['tp_px'] for r in bad):>12}")
    undecided = sum(r["judged_fg"] is None for r in rows)
    print(f"判定できなかった成分（1パッチ未満）: {undecided}")
    before = np.mean([r["before"] for r in images])
    after = np.mean([r["after"] for r in images])
    print(f"前景IoU（画像平均）: 判定なし {before:.4f} -> 判定あり {after:.4f}  ({after - before:+.4f})")
    worse = [r for r in images if r["after"] < r["before"] - 1e-6]
    better = [r for r in images if r["after"] > r["before"] + 1e-6]
    print(f"IoUが上がった画像 {len(better)} / 下がった画像 {len(worse)}")
    for r in sorted(worse, key=lambda r: r["after"] - r["before"])[:5]:
        print(f"  悪化: {r['image']} {r['before']:.3f} -> {r['after']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/ov_config.yaml")
    parser.add_argument("--prompts-dir", default=None)
    parser.add_argument("--mvtec", default="/home/llm/datasets/mvtec_anomaly_detection")
    parser.add_argument("--num", type=int, default=5, help="MVTec のカテゴリごとの画像数")
    parser.add_argument("--gt-dir", default=None, help="images/ と masks/ を持つディレクトリ（MVTec の代わりに使う）")
    parser.add_argument("--min-patches", type=int, default=MIN_ISOLATED_PATCHES, help="これ未満のパッチ数の成分は判定せず残す")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompts_dir:
        config["prompts_dir"] = args.prompts_dir
    with contextlib.redirect_stdout(io.StringIO()):
        runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)

    samples = gt_dir_samples(Path(args.gt_dir)) if args.gt_dir else mvtec_samples(Path(args.mvtec), args.num)
    report(*diagnose(runner, samples, args.min_patches))
