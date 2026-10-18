"""合成データセット（tools/make_synthetic_dataset.py の出力）で、前景マスク・有無・個数を評価する。

    python tools/eval_synthetic.py --data ./datasets/synthetic_v1
    python tools/eval_synthetic.py --data ./datasets/synthetic_v1 --prompts-dir ./prompts/candidates/color_free_bg

指標:
    IoU      : 予測の前景(セマンティックマップ>0)と、正解マスク(インスタンスID>0)のIoU
    presence : 物体の有無の正解率。予測の前景に、最小面積以上の連結成分があれば「あり」
    count_mae: 予測の連結成分数と正解の個数の平均絶対誤差
    count_ok : 個数が一致した割合
    ※ 個数は連結成分で数えるので、物体が接触・重なっている画像（meta の overlap=True）は不利になる
      （インスタンス分割が入るまでは、overlap=False の行で見る）
"""
import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config
from eval_prompts import segment


def contrast_bin(value: float) -> str:
    return "低(<15)" if value < 15 else "中(15-40)" if value < 40 else "高(>=40)"


def count_bin(count: int) -> str:
    return "0個" if count == 0 else "1個" if count == 1 else "2-3個" if count <= 3 else "4個以上"


def evaluate(runner, data: Path, min_area_ratio: float):
    records = []
    for line in open(data / "meta.jsonl", encoding="utf-8"):
        meta = json.loads(line)
        image_path = data / "images" / f"{meta['id']}.png"
        gt = np.array(Image.open(data / "masks" / f"{meta['id']}.png")) > 0
        semantic_map, _ = segment(runner, str(image_path))
        pred = semantic_map > 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
        min_area = min_area_ratio * pred.size
        pred_count = int(sum(stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n)))
        records.append({
            "meta": meta,
            "iou": (pred & gt).sum() / (pred | gt).sum() if (pred | gt).any() else 1.0,
            "presence": (pred_count > 0) == (meta["count"] > 0),
            "count_err": abs(pred_count - meta["count"]),
        })
    return records


def report(records):
    groups = {
        "全体": lambda m: "all",
        "背景の種類": lambda m: m["bg_kind"],
        "輝度差": lambda m: contrast_bin(m["contrast"]),
        "物体の個数": lambda m: count_bin(m["count"]),
        "重なり": lambda m: "あり" if m["overlap"] else "なし",
    }
    print(f"{'条件':<14}{'値':<12}{'枚数':>5}{'IoU':>8}{'presence':>10}{'count_mae':>11}{'count_ok':>10}")
    for title, key in groups.items():
        table = defaultdict(list)
        for r in records:
            table[key(r["meta"])].append(r)
        for value, rs in sorted(table.items()):
            print(f"{title:<14}{value:<12}{len(rs):>5}{np.mean([r['iou'] for r in rs]):>8.3f}"
                  f"{np.mean([r['presence'] for r in rs]):>10.2f}{np.mean([r['count_err'] for r in rs]):>11.2f}"
                  f"{np.mean([r['count_err'] == 0 for r in rs]):>10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="./configs/ov_config.yaml")
    parser.add_argument("--prompts-dir", default=None)
    parser.add_argument("--min-area-ratio", type=float, default=0.001, help="有無・個数で数える連結成分の最小面積（画像面積に対する比）")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompts_dir:
        config["prompts_dir"] = args.prompts_dir
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = type(config[key])(yaml.safe_load(value))
    with contextlib.redirect_stdout(io.StringIO()):
        runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    report(evaluate(runner, Path(args.data), args.min_area_ratio))
