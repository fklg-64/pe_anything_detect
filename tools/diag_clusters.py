"""粗いセグメンテーションの診断: クラスタ単位の前景/背景の判定を、MVTec の正解マスクと比べる。

KMeans の各クラスタについて、前景プロンプト・背景プロンプトとの最大類似度の差（gap）と、
正解マスクに含まれる割合を求め、パッチ単位の誤り（背景→前景 / 前景→背景）を集計する。
判定方法を変えたときの効果を、EdgeTAM の精緻化を通す前に確かめる用途。

    python tools/diag_clusters.py
    python tools/diag_clusters.py --prompts-dir ./prompts/candidates/xxx --json ./clusters.json

--json で、クラスタごとの値（cat, img, k, n, gt, gap, fg_prompt, bg_prompt）を保存できる。
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.common.params import CLIP_LOGIT_SCALE
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config, load_image, normalize_image, resize_image
from eval_prompts import OBJECT_CATEGORIES


def cluster_rows(runner, mvtec: Path, indices):
    coarse = runner.coarse
    fg_num = len(coarse.foreground_prompts)
    prompts = coarse.foreground_prompts + coarse.background_prompts
    size = coarse.feature_size
    rows = []
    for category in OBJECT_CATEGORIES:
        for i in indices:
            name = f"{i:03d}.png"
            image = load_image(mvtec / category / "train" / "good" / name)
            gt = np.array(Image.open(mvtec / "fg_mask" / category / name))
            gt = (gt[..., 0] if gt.ndim == 3 else gt) > 0
            gt = cv2.resize(gt.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA).reshape(-1)
            with contextlib.redirect_stdout(io.StringIO()):
                tensor = normalize_image(resize_image(image, runner.extractor.target_size), runner.extractor.mean, runner.extractor.std)
                _, patch, fused = coarse.extract_features(tensor)
                labels = coarse.fit_predict_kmeans(fused[0, 1:, :], k=coarse.k_clusters, niter=coarse.niter)
                tokens = runner.extractor.extract_cluster_tokens(patch, labels)
            for k, token in enumerate(tokens):
                members = labels == k
                if not members.any():
                    continue
                feature = token.reshape(1, -1) / (np.linalg.norm(token) + 1e-8)
                logits = (CLIP_LOGIT_SCALE * feature @ coarse.fgbg_text_features.T)[0]
                rows.append({
                    "cat": category, "img": i, "k": k, "n": int(members.sum()), "gt": float(gt[members].mean()),
                    "gap": float(logits[:fg_num].max() - logits[fg_num:].max()),
                    "fg_prompt": prompts[int(np.argmax(logits[:fg_num]))],
                    "bg_prompt": prompts[fg_num + int(np.argmax(logits[fg_num:]))],
                })
    return rows


def report(rows):
    n = np.array([r["n"] for r in rows], float)
    gt = np.array([r["gt"] for r in rows])
    pred = np.array([r["gap"] > 0 for r in rows])
    fp = (n * (1 - gt) * pred).sum() / n.sum() * 100
    fn = (n * gt * ~pred).sum() / n.sum() * 100
    print(f"パッチ単位の誤り: 背景→前景 {fp:.2f}%  前景→背景 {fn:.2f}%  計 {fp + fn:.2f}%")
    for category in OBJECT_CATEGORIES:
        sel = np.array([r["cat"] == category for r in rows])
        err = (n * ((1 - gt) * pred + gt * ~pred))[sel].sum() / n[sel].sum() * 100
        print(f"  {category:<11}{err:6.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/ov_config.yaml")
    parser.add_argument("--prompts-dir", default=None)
    parser.add_argument("--mvtec", default="/home/llm/datasets/mvtec_anomaly_detection")
    parser.add_argument("--num", type=int, default=5, help="カテゴリごと・dev/holdout ごとの画像数")
    parser.add_argument("--json", default=None, help="クラスタごとの値を保存するファイル")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prompts_dir:
        config["prompts_dir"] = args.prompts_dir
    with contextlib.redirect_stdout(io.StringIO()):
        runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)

    indices = [i * 7 for i in range(args.num)] + [3 + i * 11 for i in range(args.num)]
    rows = cluster_rows(runner, Path(args.mvtec), indices)
    report(rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
