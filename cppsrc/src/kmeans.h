// KMeans（faiss.Kmeans(spherical=False) と同じ手順: ランダムな点で初期化 -> 割り当てと平均の更新を niter 回 -> 最後に割り当て）
// 初期化の乱数（mt19937 による rand_perm）と空クラスタの分割も faiss と同じにしてあるので、同じ入力・同じ seed なら同じクラスタ番号になる
#pragma once

#include <cstdint>
#include <vector>

namespace pes {

// faiss::rand_perm: 0..n-1 の順列（Fisher-Yates。mt19937(seed) を使う）
std::vector<int> faiss_rand_perm(int n, int64_t seed);

// x: n × d（行優先）-> n 個のクラスタ番号（0〜k-1）
std::vector<int> kmeans_fit_predict(const float* x, int n, int d, int k, int niter, int seed);

}  // namespace pes
