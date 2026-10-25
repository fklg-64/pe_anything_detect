#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include "common.h"
#include "simd.h"

namespace pes {

std::vector<int> faiss_rand_perm(int n, int64_t seed) {
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::mt19937 mt(static_cast<unsigned int>(seed));
    for (int i = 0; i + 1 < n; ++i) {
        const int i2 = i + static_cast<int>(mt() % static_cast<unsigned>(n - i));
        std::swap(perm[i], perm[i2]);
    }
    return perm;
}

namespace {

// 各点に最も近い重心の番号（IndexFlatL2.search(k=1) 相当: |x|² + |c|² - 2 x·c が最小のもの。同値なら番号の小さいほう）
void assign_nearest(const float* x, const float* x_norms, int n, const float* centroids, int k, int d, int* assign, std::vector<float>& scratch) {
    std::vector<float> c_norms(k);
    for (int j = 0; j < k; ++j) c_norms[j] = simd::norm_sq(centroids + static_cast<size_t>(j) * d, d);

    scratch.resize(static_cast<size_t>(n) * k);
    // 内積 x_i · c_j。faiss は BLAS (sgemm) で求め、その丸め（各要素が k 順の直列な FMA の連鎖）に合わせないと、
    // 距離がほぼ等しい点の割り当てが入れ替わり、KMeans の結果（クラスタ番号）が faiss と食い違うことがある
    simd::gemm_nt_chain(x, n, d, centroids, k, d, d, scratch.data(), k);
    for (int i = 0; i < n; ++i) {
        const float* ip = scratch.data() + static_cast<size_t>(i) * k;
        int best = 0;
        float best_dis = std::numeric_limits<float>::infinity();
        for (int j = 0; j < k; ++j) {
            float dis = x_norms[i] + c_norms[j] - 2.0f * ip[j];
            if (dis < 0.0f) dis = 0.0f;  // 同一ベクトルで丸め誤差により負になるのを防ぐ
            if (dis < best_dis) {
                best_dis = dis;
                best = j;
            }
        }
        assign[i] = best;
    }
}

// faiss::compute_centroids: 割り当てた点の平均。空のクラスタの重心は 0 のまま
void compute_centroids(const float* x, int n, int d, int k, const int* assign, std::vector<float>& hassign, float* centroids) {
    std::fill(centroids, centroids + static_cast<size_t>(k) * d, 0.0f);
    std::fill(hassign.begin(), hassign.end(), 0.0f);
    for (int i = 0; i < n; ++i) {
        const int c = assign[i];
        hassign[c] += 1.0f;
        simd::add_inplace(centroids + static_cast<size_t>(c) * d, x + static_cast<size_t>(i) * d, d);
    }
    for (int c = 0; c < k; ++c) {
        if (hassign[c] == 0.0f) continue;
        const float norm = 1.0f / hassign[c];
        simd::scale_inplace(centroids + static_cast<size_t>(c) * d, d, norm);
    }
}

// faiss::split_clusters: 空のクラスタの重心を、大きいクラスタを少しずらして二分した重心で置き換える
void split_clusters(int d, int k, int n, std::vector<float>& hassign, float* centroids) {
    constexpr double EPS = 1.0 / 1024.0;
    std::mt19937 rng(1234);
    for (int ci = 0; ci < k; ++ci) {
        if (hassign[ci] != 0.0f) continue;
        int cj = 0;
        for (;; cj = (cj + 1) % k) {
            // このクラスタを分割する確率
            const float p = static_cast<float>((static_cast<double>(hassign[cj]) - 1.0) / static_cast<double>(static_cast<float>(n - k)));
            const float r = static_cast<float>(rng()) / static_cast<float>(std::mt19937::max());
            if (r < p) break;
        }
        std::memcpy(centroids + static_cast<size_t>(ci) * d, centroids + static_cast<size_t>(cj) * d, sizeof(float) * d);
        for (int j = 0; j < d; ++j) {  // 小さな対称の摂動
            float& a = centroids[static_cast<size_t>(ci) * d + j];
            float& b = centroids[static_cast<size_t>(cj) * d + j];
            if (j % 2 == 0) {
                a = static_cast<float>(static_cast<double>(a) * (1 + EPS));
                b = static_cast<float>(static_cast<double>(b) * (1 - EPS));
            } else {
                a = static_cast<float>(static_cast<double>(a) * (1 - EPS));
                b = static_cast<float>(static_cast<double>(b) * (1 + EPS));
            }
        }
        hassign[ci] = hassign[cj] / 2;  // 均等に分割したとみなす
        hassign[cj] -= hassign[ci];
    }
}

}  // namespace

std::vector<int> kmeans_fit_predict(const float* x_in, int n_in, int d, int k, int niter, int seed) {
    if (k <= 0 || d <= 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "KMeans のクラスタ数・次元は 1 以上にしてください");
    if (n_in < k) throw Error(PES_ERROR_INVALID_ARGUMENT, "KMeans の学習点の数 (" + std::to_string(n_in) + ") がクラスタ数 (" + std::to_string(k) + ") より少ないです");

    // 学習点が k × 256 を超えるときは、ランダムに k × 256 点を選んで学習する（faiss の既定 max_points_per_centroid）
    constexpr int kMaxPointsPerCentroid = 256;
    std::vector<float> subsampled;
    const float* x = x_in;
    int n = n_in;
    if (n > k * kMaxPointsPerCentroid) {
        const std::vector<int> perm = faiss_rand_perm(n, seed);
        n = k * kMaxPointsPerCentroid;
        subsampled.resize(static_cast<size_t>(n) * d);
        for (int i = 0; i < n; ++i) std::memcpy(subsampled.data() + static_cast<size_t>(i) * d, x_in + static_cast<size_t>(perm[i]) * d, sizeof(float) * d);
        x = subsampled.data();
    }

    std::vector<float> x_norms(n);
    for (int i = 0; i < n; ++i) x_norms[i] = simd::norm_sq(x + static_cast<size_t>(i) * d, d);

    // 初期の重心: ランダムに選んだ k 点
    const std::vector<int> perm = faiss_rand_perm(n, static_cast<int64_t>(seed) + 1);
    std::vector<float> centroids(static_cast<size_t>(k) * d);
    for (int i = 0; i < k; ++i) std::memcpy(centroids.data() + static_cast<size_t>(i) * d, x + static_cast<size_t>(perm[i]) * d, sizeof(float) * d);

    std::vector<int> assign(n);
    std::vector<float> hassign(k), scratch;
    for (int it = 0; it < niter; ++it) {
        assign_nearest(x, x_norms.data(), n, centroids.data(), k, d, assign.data(), scratch);
        compute_centroids(x, n, d, k, assign.data(), hassign, centroids.data());
        split_clusters(d, k, n, hassign, centroids.data());
    }

    // 最終の重心で、元の全点（サブサンプル前）を割り当てる
    std::vector<int> labels(n_in);
    std::vector<float> all_norms(n_in);
    for (int i = 0; i < n_in; ++i) all_norms[i] = simd::norm_sq(x_in + static_cast<size_t>(i) * d, d);
    assign_nearest(x_in, all_norms.data(), n_in, centroids.data(), k, d, labels.data(), scratch);
    return labels;
}

}  // namespace pes
