#!/usr/bin/env python
"""C++ 版の後処理（mask_utils / coarse_segmentation の純粋な関数）の確認用データを作る。
Python 版 (core/common) の関数を乱数で作った入力に適用し、入力と出力を .npy で書く。
tests/test_reference.cpp が同じ入力に C++ 版の関数を適用して、出力が一致することを確認する。

使い方: python cppsrc/scripts/gen_reference_cases.py <出力ディレクトリ> [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.common import mask_utils as M  # noqa: E402
from core.common.coarse_segmentation import CoarseSegmenter  # noqa: E402

CASES = {}


def save(out, group, index, name, array):
    np.save(out / f"{group}_{index:03d}_{name}.npy", np.ascontiguousarray(array))


def blob_mask(rng, h, w, n_blobs, speckle=0.0, min_r=0.04, max_r=0.25):
    """楕円を重ねた 2 値マスク + ごま塩ノイズ。"""
    mask = np.zeros((h, w), np.uint8)
    for _ in range(n_blobs):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (max(1, int(rng.uniform(min_r, max_r) * w)), max(1, int(rng.uniform(min_r, max_r) * h)))
        cv2.ellipse(mask, c, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    if speckle > 0:
        mask ^= (rng.random((h, w)) < speckle).astype(np.uint8)
    return mask


def class_map(rng, h, w, n_classes, speckle=0.0, bg_fraction=0.4):
    """Voronoi 風に領域分けしたクラス番号の画像（0=背景）。"""
    seeds = rng.random((n_classes + 3, 2)) * [w, h]
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx[..., None] - seeds[:, 0]) ** 2 + (yy[..., None] - seeds[:, 1]) ** 2)
    region = d.argmin(-1)
    labels = (region % (n_classes + 1)).astype(np.uint8)
    labels[(rng.random((h, w)) < 0.0)] = 0
    fg_blob = blob_mask(rng, h, w, 3, 0.0, 0.1, 0.4)
    labels = labels * fg_blob
    if speckle > 0:
        noise = rng.random((h, w)) < speckle
        labels[noise] = rng.integers(0, n_classes + 1, size=int(noise.sum()))
    return labels.astype(np.uint8)


def gen_fps(out, rng, n):
    for t in range(n):
        h, w = [(32, 32), (100, 80), (16, 16)][t % 3]
        m = int(rng.integers(1, 200))
        flat = rng.choice(h * w, size=min(m, h * w), replace=False)
        flat.sort()
        pts = np.stack([flat % w, flat // w], 1).astype(np.int32)  # 行優先 (x, y)
        k = int(rng.integers(0, len(pts) + 1))
        save(out, "fps", t, "points", pts)
        save(out, "fps", t, "k", np.array([k], np.int32))
        save(out, "fps", t, "out", M.farthest_point_sampling(pts, k).astype(np.int32))
    return n


def gen_prompt_points(out, rng, n):
    for t in range(n):
        mask = blob_mask(rng, 32, 32, int(rng.integers(1, 4)), 0.02)
        comps = M.connected_components(mask)
        if not comps:
            mask[16, 16] = 1
            comps = M.connected_components(mask)
        main = comps
        core = np.any([c.core for c in main], axis=0)
        peaks = np.array([c.peak for c in main], np.int32)
        k = int(rng.integers(1, 9))
        save(out, "prompt", t, "core", core.astype(np.uint8))
        save(out, "prompt", t, "peaks", peaks)
        save(out, "prompt", t, "k", np.array([k], np.int32))
        save(out, "prompt", t, "out", M.prompt_points(core, [tuple(p) for p in peaks], k).astype(np.int32))
    return n


def gen_background_points(out, rng, n):
    for t in range(n):
        fg = blob_mask(rng, 32, 32, int(rng.integers(1, 5)), 0.01).astype(bool)
        if t % 7 == 0:
            fg[:] = False
            fg[10:22, 10:22] = True  # 画像端に接しない前景
        if t % 11 == 0:
            fg |= np.pad(np.ones((30, 30), bool), 1) ^ True  # 外周が前景（背景が内側に閉じる）
        # 確信度: 連続値、または量子化（同値が多い）
        fmap = rng.random((32, 32)).astype(np.float32)
        if t % 2:
            fmap = (np.round(fmap * 6) / 6).astype(np.float32)
        count = int(rng.choice([2, 4, 8]))
        save(out, "bgpoints", t, "fg", fg.astype(np.uint8))
        save(out, "bgpoints", t, "fmap", fmap)
        save(out, "bgpoints", t, "count", np.array([count], np.int32))
        save(out, "bgpoints", t, "out", M.select_background_points(fg, count, fmap).astype(np.int32).reshape(-1, 2))
    return n


def gen_components(out, rng, n):
    for t in range(n):
        mask = blob_mask(rng, 32, 32, int(rng.integers(1, 6)), 0.03, 0.03, 0.2)
        comps = M.connected_components(mask)
        save(out, "components", t, "mask", mask)
        save(out, "components", t, "peaks", np.array([c.peak for c in comps], np.int32).reshape(-1, 2))
        save(out, "components", t, "cores", np.array([c.core for c in comps], np.uint8).reshape(len(comps), 32, 32))
        save(out, "components", t, "regions", np.array([c.region for c in comps], np.uint8).reshape(len(comps), 32, 32))
    return n


def gen_clean_mask(out, rng, n):
    for t in range(n):
        h, w = int(rng.integers(40, 300)), int(rng.integers(40, 300))
        mask = blob_mask(rng, h, w, int(rng.integers(0, 5)), float(rng.choice([0, 0.002, 0.01, 0.05])))
        min_area = int(rng.choice([1, 4, 16, 60]))
        save(out, "clean", t, "mask", mask)
        save(out, "clean", t, "min_area", np.array([min_area], np.int32))
        save(out, "clean", t, "out", M.clean_mask(mask.astype(bool), min_area).astype(np.uint8))
    return n


def gen_remove_small(out, rng, n):
    for t in range(n):
        h, w = int(rng.integers(60, 320)), int(rng.integers(60, 320))
        labels = class_map(rng, h, w, int(rng.integers(1, 5)), float(rng.choice([0, 0.001, 0.01])))
        min_area = int(rng.choice([2, 8, 24, 100]))
        expected = labels.copy()
        M.remove_small_regions(expected, min_area)
        save(out, "remove", t, "map", labels)
        save(out, "remove", t, "min_area", np.array([min_area], np.int32))
        save(out, "remove", t, "out", expected)
    return n


def gen_fill_gaps(out, rng, n):
    changed = 0
    t = 0
    attempts = 0
    while t < n and attempts < n * 40:
        attempts += 1
        h, w = int(rng.integers(160, 420)), int(rng.integers(160, 420))
        grid = 32
        # 粗い前景（パッチ格子）: 大きな塊。作業解像度の分類マップは、その中で 2 クラスに分かれ、境界に細い隙間がある
        fg = blob_mask(rng, grid, grid, int(rng.integers(1, 3)), 0.0, 0.2, 0.45).astype(np.uint8)
        full = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
        seeds = rng.random((int(rng.integers(2, 4)), 2)) * [w, h]
        yy, xx = np.mgrid[0:h, 0:w]
        region = (((xx[..., None] - seeds[:, 0]) ** 2 + (yy[..., None] - seeds[:, 1]) ** 2).argmin(-1) + 1).astype(np.uint8)
        smap = region * full
        # クラス境界を細い隙間にする（1〜3 画素）
        edges = cv2.Canny(region, 0, 1) > 0
        gap = cv2.dilate(edges.astype(np.uint8), np.ones((int(rng.integers(1, 4)),) * 2, np.uint8)) > 0
        smap[gap] = 0
        # 前景の外側の細かな穴・欠けも少し
        smap[rng.random((h, w)) < 0.0005] = 0
        expected = smap.copy()
        M.fill_narrow_gaps(expected, fg.astype(bool))
        diff = int((expected != smap).sum())
        if diff == 0 and rng.random() < 0.85:
            continue  # 何も埋まらない場合は一部だけ残す
        changed += diff > 0
        save(out, "gaps", t, "map", smap)
        save(out, "gaps", t, "fg", fg)
        save(out, "gaps", t, "out", expected)
        t += 1
    print(f"  fill_narrow_gaps: {t} 件中 {changed} 件で実際に画素が埋まる")
    return t


def gen_filters(out, rng, n):
    for t in range(n):
        grid = rng.random((32, 32)).astype(np.float32)
        save(out, "pmean", t, "in", (rng.random((32, 32)) < 0.4).astype(np.float32))
        save(out, "pmean", t, "out", M.patch_mean(np.load(out / f"pmean_{t:03d}_in.npy") > 0))
        size = int(rng.choice([64, 128]))
        guide = rng.random((size, size)).astype(np.float32)
        src = rng.random((size, size)).astype(np.float32)
        radius = int(rng.choice([1, 2, 4]))
        eps = float(rng.choice([1e-2, 1e-3]))
        save(out, "guided", t, "guide", guide)
        save(out, "guided", t, "src", src)
        save(out, "guided", t, "params", np.array([radius, eps], np.float64))
        save(out, "guided", t, "out", M.guided_filter(guide, src, radius, eps).astype(np.float32))
    return n


def gen_propagation(out, rng, n):
    for t in range(n):
        c = int(rng.choice([32, 256]))
        h = w = int(rng.choice([8, 16, 32]))
        count = h * w
        k_clusters = int(rng.integers(2, 6))
        centers = rng.normal(size=(k_clusters, c)).astype(np.float32)
        lab = rng.integers(0, k_clusters, size=count)
        feats = centers[lab] + float(rng.choice([0.3, 1.0])) * rng.normal(size=(count, c)).astype(np.float32)
        embed = feats.T.reshape(1, c, h, w).astype(np.float32)
        # PE 風の確信度: クラスタごとに一定 + 一部のパッチだけ反転
        cluster_conf = rng.random(k_clusters).astype(np.float32)
        conf = cluster_conf[rng.integers(0, k_clusters, size=count)].astype(np.float32)
        num_fg = 5
        prompt = np.where(conf > 0.5, rng.integers(0, num_fg, size=count), rng.integers(num_fg, num_fg + 4, size=count)).astype(np.int64)
        obj = object.__new__(CoarseSegmenter)
        obj.foreground_prompts = ["x"] * num_fg
        obj.foreground_threshold = 0.5
        new_conf, new_prompt = obj._propagate_with_edgetam(conf, prompt, embed)
        save(out, "propagate", t, "embed", embed)
        save(out, "propagate", t, "conf", conf)
        save(out, "propagate", t, "prompt", prompt)
        save(out, "propagate", t, "num_fg", np.array([num_fg], np.int32))
        save(out, "propagate", t, "out_conf", np.asarray(new_conf, np.float32))
        save(out, "propagate", t, "out_prompt", np.asarray(new_prompt, np.int64))
        save(out, "propagate", t, "out_dtype", np.array([np.asarray(new_conf).dtype.itemsize], np.int32))
    return n


def gen_coarse(out, rng, n):
    obj = object.__new__(CoarseSegmenter)
    obj.feature_size = 32
    for t in range(n):
        grid = 32
        fg = blob_mask(rng, grid, grid, int(rng.integers(1, 5)), float(rng.choice([0, 0.01, 0.03])), 0.05, 0.35)
        # 前景の中に穴を開ける
        if t % 3 == 0:
            fg[rng.integers(8, 24), rng.integers(8, 24)] = 0
        labels = rng.integers(0, int(rng.integers(2, 6)), size=(grid, grid)).astype(np.int64)
        if t % 2 == 0:  # 滑らかなラベル（領域単位）
            seeds = rng.random((int(rng.integers(2, 6)), 2)) * grid
            yy, xx = np.mgrid[0:grid, 0:grid]
            labels = (((xx[..., None] - seeds[:, 0]) ** 2 + (yy[..., None] - seeds[:, 1]) ** 2).argmin(-1)).astype(np.int64)
            noise = rng.random((grid, grid)) < 0.04
            labels[noise] = rng.integers(0, 6, size=int(noise.sum()))
        out_mask, out_labels = obj.refine_coarse_segmentation(labels, fg.astype(np.int64))
        save(out, "coarse", t, "labels", labels.astype(np.int32))
        save(out, "coarse", t, "fg", fg)
        save(out, "coarse", t, "out_mask", out_mask.astype(np.uint8))
        save(out, "coarse", t, "out_labels", out_labels.astype(np.int32))
        # クラスタ -> ラベルの付与
        n_clusters = int(rng.integers(1, 12))
        cluster_labels = rng.integers(0, n_clusters, size=grid * grid)
        cluster_index = rng.integers(0, 50, size=cluster_labels.max() + 1)
        labels_list, sem = obj.assign_semantic_labels(cluster_labels, list(cluster_index))
        save(out, "assign", t, "cluster_labels", cluster_labels.astype(np.int32))
        save(out, "assign", t, "cluster_index", cluster_index.astype(np.int32))
        save(out, "assign", t, "labels_list", np.asarray(labels_list, np.int32))
        save(out, "assign", t, "semantic", sem.astype(np.int32))
    return n


def gen_nearest(out, rng, n):
    for t in range(n):
        h, w = int(rng.integers(5, 80)), int(rng.integers(5, 80))
        feature = rng.random((h, w)) < float(rng.choice([0.01, 0.05, 0.3]))
        if not feature.any():
            feature[rng.integers(0, h), rng.integers(0, w)] = True
        dist, (iy, ix) = distance_transform_edt(~feature, return_indices=True)
        save(out, "nearest", t, "feature", feature.astype(np.uint8))
        save(out, "nearest", t, "dist2", (dist ** 2).round().astype(np.int32))
    return n


def gen_numpy_ops(out, rng, n):
    """NumPy / BLAS の丸め方の再現の確認: 二乗和・ノルムで割る・ベクトル @ 行列・X @ X.T。"""
    for t in range(n):
        rows, cols = int(rng.choice([1, 3, 20])), int(rng.choice([7, 64, 128, 130, 256, 512, 1000]))
        x = (rng.normal(size=(rows, cols)) * float(rng.choice([0.05, 1.0, 30.0]))).astype(np.float32)
        eps = float(rng.choice([0.0, 1e-8]))
        save(out, "numpy", t, "x", x)
        save(out, "numpy", t, "eps", np.array([eps], np.float32))
        save(out, "numpy", t, "sumsq", np.add.reduce((x.conj() * x).real, axis=-1).astype(np.float32))
        save(out, "numpy", t, "normalized", (x / (np.linalg.norm(x, axis=-1, keepdims=True) + np.float32(eps))).astype(np.float32))
    return n


def gen_gemv(out, rng, n):
    for t in range(n):
        m = 1 + (t % 70)
        d = int(rng.choice([512, 512, 512, 64, 24]))
        f = rng.normal(size=(1, d)).astype(np.float32)
        f /= np.linalg.norm(f, axis=-1, keepdims=True)
        text = rng.normal(size=(m, d)).astype(np.float32)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        save(out, "gemv", t, "feature", f)
        save(out, "gemv", t, "text", text)
        save(out, "gemv", t, "out", (100.0 * f @ text.T).astype(np.float32))
    return n


def gen_gram(out, rng, n):
    for t in range(n):
        count, c = [(1024, 256), (256, 256), (100, 64), (37, 24), (1600, 256), (8, 8), (9, 33)][t % 7]
        f = rng.normal(size=(count, c)).astype(np.float32)
        f /= (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)
        save(out, "gram", t, "x", f)
        save(out, "gram", t, "out", (f @ f.T).astype(np.float32))
    return n


def gen_classify(out, rng, n):
    """クラスタ分類（_classify_cluster）の全体: 正規化・logits・softmax・プロンプト番号。"""
    for t in range(n):
        num_fg, num_bg, d = int(rng.integers(1, 40)), int(rng.integers(1, 40)), 512
        text = rng.normal(size=(num_fg + num_bg, d)).astype(np.float32)
        text /= np.linalg.norm(text, axis=-1, keepdims=True)
        feature = rng.normal(size=(d,)).astype(np.float32) * float(rng.choice([0.3, 1.0, 5.0]))
        # 前景に近い / 背景に近いトークンを混ぜる
        pick = int(rng.integers(0, num_fg + num_bg))
        feature = feature * 0.3 + text[pick] * float(rng.choice([0.0, 1.0, 3.0]))
        feature = feature.astype(np.float32)
        thr = float(rng.choice([0.5, 0.3]))
        obj = object.__new__(CoarseSegmenter)
        obj.foreground_prompts = ["x"] * num_fg
        obj.fgbg_text_features = text
        obj.foreground_threshold = thr
        conf, idx = obj._classify_cluster(feature)
        save(out, "classify", t, "feature", feature)
        save(out, "classify", t, "text", text)
        save(out, "classify", t, "meta", np.array([num_fg, num_bg], np.int32))
        save(out, "classify", t, "thr", np.array([thr], np.float64))
        save(out, "classify", t, "out", np.array([conf], np.float64))
        save(out, "classify", t, "idx", np.array([idx], np.int32))
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    counts = {}
    for name, fn, n in [("fps", gen_fps, 60), ("prompt", gen_prompt_points, 40), ("bgpoints", gen_background_points, 60),
                        ("components", gen_components, 40), ("clean", gen_clean_mask, 40), ("remove", gen_remove_small, 40),
                        ("gaps", gen_fill_gaps, 30), ("pmean", gen_filters, 10), ("propagate", gen_propagation, 24),
                        ("coarse", gen_coarse, 80), ("nearest", gen_nearest, 30), ("numpy", gen_numpy_ops, 40), ("gemv", gen_gemv, 140),
                        ("gram", gen_gram, 14), ("classify", gen_classify, 300)]:
        counts[name] = fn(out, rng, n)
        print(f"{name}: {counts[name]} 件")
    # 件数（C++ 側はこれを読む）
    np.save(out / "counts.npy", np.array([counts[k] for k in ["fps", "prompt", "bgpoints", "components", "clean", "remove", "gaps", "pmean", "propagate", "coarse", "nearest", "numpy", "gemv", "gram", "classify"]], np.int32))


if __name__ == "__main__":
    main()
