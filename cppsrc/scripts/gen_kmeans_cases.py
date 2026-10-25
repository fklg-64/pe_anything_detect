#!/usr/bin/env python
"""KMeans の確認用データを作る: faiss.rand_perm と faiss.Kmeans（Python 版が使うもの）の出力を正解にして、指定ディレクトリに .npy で書く。
C++ の pes::kmeans_fit_predict（tests/test_kmeans.cpp）が同じクラスタ番号を返すことを確認する。

使い方: python cppsrc/scripts/gen_kmeans_cases.py <出力ディレクトリ>
"""
import sys
from pathlib import Path

import faiss
import numpy as np


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    for n, seed in [(10, 43), (1024, 43), (1024, 42), (5000, 7)]:
        perm = np.zeros(n, dtype=np.int32)
        faiss.rand_perm(faiss.swig_ptr(perm), n, seed)
        np.save(out / f"perm_{n}_{seed}.npy", perm.astype(np.int64))

    rng = np.random.default_rng(0)

    def blobs(n, d, k, noise):
        centers = rng.normal(size=(k, d)).astype(np.float32)
        lab = rng.integers(0, k, size=n)
        x = centers[lab] + noise * rng.normal(size=(n, d)).astype(np.float32)
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    # 名前: (点数, 次元, クラスタ数, ノイズ)。e, f は学習点が k×256 を超える（サブサンプルされる）場合
    cases = {"a": (1024, 192, 8, 0.6), "b": (1024, 576, 8, 1.2), "c": (1024, 192, 12, 2.0), "d": (900, 64, 3, 0.5),
             "e": (1024, 192, 1, 1.0), "f": (1024, 192, 2, 1.0)}
    for name, (n, d, k, noise) in cases.items():
        x = blobs(n, d, k if k > 1 else 3, noise)
        faiss.omp_set_num_threads(2)
        km = faiss.Kmeans(d=d, k=k, niter=10, spherical=False, verbose=False, seed=42)  # Python 版 fit_predict_kmeans と同じ設定
        km.train(x)
        _, lab = km.index.search(x, 1)
        np.save(out / f"x_{name}.npy", x)
        np.save(out / f"labels_{name}.npy", lab.reshape(-1).astype(np.int32))
        np.save(out / f"meta_{name}.npy", np.array([n, d, k], dtype=np.int32))
        print(name, n, d, k, np.bincount(lab.reshape(-1), minlength=k))


if __name__ == "__main__":
    main()
