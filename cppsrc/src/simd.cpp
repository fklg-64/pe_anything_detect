// スカラー実装と、AVX2 / スカラーの振り分け
#include "simd.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace pes::simd {

// =============================================================================
// スカラー実装（AVX2 非対応 CPU 用・AVX2 実装の確認用）
// =============================================================================
namespace {

void normalize_to_chw_scalar(const uint8_t* src, size_t npix, bool src_is_bgr, const NormalizeTarget* targets, int num_targets) {
    for (int t = 0; t < num_targets; ++t) {
        const NormalizeTarget& tg = targets[t];
        for (int c = 0; c < 3; ++c) {
            const int sc = src_is_bgr ? 2 - c : c;
            float* dst = tg.dst + static_cast<size_t>(c) * npix;
            for (size_t i = 0; i < npix; ++i) {
                const float v = static_cast<float>(src[i * 3 + sc]) / 255.0f;
                dst[i] = (v - tg.mean[c]) / tg.stdv[c];
            }
        }
    }
}

void gemm_nt_chain_scalar(const float* A, int M, int lda, const float* B, int N, int ldb, int D, float* C, int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            const float* a = A + static_cast<size_t>(i) * lda;
            const float* b = B + static_cast<size_t>(j) * ldb;
            for (int t = 0; t < D; ++t) s = std::fma(a[t], b[t], s);
            C[static_cast<size_t>(i) * ldc + j] = s;
        }
}

// 8 レーンの FMA で足し込み（レーン l は l, l+8, ... 番目の要素）、(l と l+4 を足す) -> (隣どうし) -> (もう一度隣どうし) の順で畳む
float norm_sq_scalar(const float* x, int n) {
    float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; ++l) lane[l] = std::fma(x[i + l], x[i + l], lane[l]);
    const float q0 = lane[0] + lane[4], q1 = lane[1] + lane[5], q2 = lane[2] + lane[6], q3 = lane[3] + lane[7];
    float s = (q0 + q1) + (q2 + q3);
    for (; i < n; ++i) s += x[i] * x[i];
    return s;
}

void gram_symmetric_scalar(const float* X, int N, int D, float* S) {
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            float v = 0.0f;
            const float* a = X + static_cast<size_t>(i) * D;
            const float* b = X + static_cast<size_t>(j) * D;
            for (int t = 0; t < D; ++t) v = std::fma(a[t], b[t], v);
            S[static_cast<size_t>(i) * N + j] = v;
            S[static_cast<size_t>(j) * N + i] = v;
        }
}

// レーン（8 本）を (l と l+4 を足す) -> (隣どうし) -> (もう一度隣どうし) の順で畳む。AVX2 の「上下 128bit を足す -> hadd -> hadd」と同じ
inline float fold8(const float lane[8]) {
    const float q0 = lane[0] + lane[4], q1 = lane[1] + lane[5], q2 = lane[2] + lane[6], q3 = lane[3] + lane[7];
    return (q0 + q1) + (q2 + q3);
}

// OpenBLAS sgemv_t (Haswell) の再現。D は 8 の倍数を想定（そうでない分は最後に直列に足す）
void gemv_t_blas_scalar(const float* x, const float* A, int M, int D, float* y) {
    const int groups = M & ~3;
    const int d8 = D & ~7, d4 = D & ~3;
    for (int j = 0; j < M; ++j) {
        const float* a = A + static_cast<size_t>(j) * D;
        float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float result;
        if (j < groups) {  // 4 行ずつ: 8 レーンの FMA
            for (int k = 0; k < d8; k += 8)
                for (int l = 0; l < 8; ++l) lane[l] = std::fma(x[k + l], a[k + l], lane[l]);
            result = fold8(lane);
            for (int k = d8; k < D; ++k) result += x[k] * a[k];
        } else if (j < groups + (M & 2)) {  // 余りの 2 行: 4 レーンの mul + add
            for (int k = 0; k < d4; k += 4)
                for (int l = 0; l < 4; ++l) {
                    const float p = x[k + l] * a[k + l];
                    lane[l] = lane[l] + p;
                }
            result = (lane[0] + lane[1]) + (lane[2] + lane[3]);
            for (int k = d4; k < D; ++k) result += x[k] * a[k];
        } else {  // 余りの 1 行: 8 レーンの mul + add
            for (int k = 0; k < d8; k += 8)
                for (int l = 0; l < 8; ++l) {
                    const float p = x[k + l] * a[k + l];
                    lane[l] = lane[l] + p;
                }
            result = fold8(lane);
            for (int k = d8; k < D; ++k) result += x[k] * a[k];
        }
        y[j] = result;
    }
}

// NumPy の pairwise summation（np.add.reduce の float32）。要素は x[i] * x[i] を float32 に丸めたもの
float pairwise_sum_squares(const float* a, long n) {
    if (n < 8) {
        float res = 0.0f;
        for (long i = 0; i < n; ++i) {
            const float sq = a[i] * a[i];
            res += sq;
        }
        return res;
    }
    if (n <= 128) {  // PW_BLOCKSIZE
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j] * a[j];
        long i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) {
                const float sq = a[i + j] * a[i + j];
                r[j] += sq;
            }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            const float sq = a[i] * a[i];
            res += sq;
        }
        return res;
    }
    long n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum_squares(a, n2) + pairwise_sum_squares(a + n2, n - n2);
}

float sum_squares_numpy_scalar(const float* x, int n) { return pairwise_sum_squares(x, n); }

// 行ごとの上位 k 個。値が同じなら列番号の小さいほうを優先する（AVX2 実装と同じ）
void topk_rows_scalar(const float* S, int N, int k, bool exclude_diagonal, int* out) {
    std::vector<std::pair<float, int>> best;
    for (int i = 0; i < N; ++i) {
        const float* row = S + static_cast<size_t>(i) * N;
        best.clear();
        float threshold = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
            if (exclude_diagonal && j == i) continue;
            const float v = row[j];
            if (static_cast<int>(best.size()) < k) {
                best.emplace_back(v, j);
                if (static_cast<int>(best.size()) == k) {
                    threshold = std::min_element(best.begin(), best.end())->first;
                }
            } else if (v > threshold) {
                auto worst = std::min_element(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                *worst = {v, j};
                threshold = std::min_element(best.begin(), best.end())->first;
            }
        }
        for (int t = 0; t < k; ++t) out[static_cast<size_t>(i) * k + t] = t < static_cast<int>(best.size()) ? best[t].second : -1;
    }
}

void transpose_scalar(const float* src, int rows, int cols, float* dst) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
}

void normalize_rows_scalar(float* x, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<size_t>(r) * cols;
        const float denom = std::sqrt(pairwise_sum_squares(row, cols)) + eps;
        for (int c = 0; c < cols; ++c) row[c] = row[c] / denom;
    }
}

void sigmoid_scalar(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

void clamp_scalar(const float* x, float* y, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; ++i) y[i] = std::min(std::max(x[i], lo), hi);
}

void greater_than_scalar(const float* x, uint8_t* out, size_t n, float t) {
    for (size_t i = 0; i < n; ++i) out[i] = x[i] > t ? 1 : 0;
}

void add_inplace_scalar(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

void scale_inplace_scalar(float* x, size_t n, float s) {
    for (size_t i = 0; i < n; ++i) x[i] *= s;
}

void score_from_logit_scalar(const float* logit, const float* prior, float* score, size_t n, float w) {
    const float one_minus_w = 1.0f - w;
    for (size_t i = 0; i < n; ++i) {
        const float conf = 1.0f / (1.0f + std::exp(-logit[i]));
        const float wp = w * prior[i];
        const float u = one_minus_w + wp;
        score[i] = conf * u;
    }
}

void update_best_scalar(const uint8_t* binary, const float* score, float* best, uint8_t* map, size_t n, uint8_t class_id) {
    for (size_t i = 0; i < n; ++i) {
        if (binary[i] && score[i] > best[i]) {
            map[i] = class_id;
            best[i] = score[i];
        }
    }
}

}  // namespace

const Kernels kScalarKernels = {
    normalize_to_chw_scalar, gemm_nt_chain_scalar, gram_symmetric_scalar, gemv_t_blas_scalar, norm_sq_scalar, sum_squares_numpy_scalar,
    topk_rows_scalar,        transpose_scalar,     normalize_rows_scalar, sigmoid_scalar,      clamp_scalar,    greater_than_scalar,
    add_inplace_scalar,      scale_inplace_scalar, score_from_logit_scalar, update_best_scalar,
};

// =============================================================================
// 振り分け
// =============================================================================
namespace {
std::atomic<bool> g_force_scalar{false};

const Kernels& kernels() {
    static const bool avx2 = cpu_has_avx2();
    return (avx2 && !g_force_scalar.load(std::memory_order_relaxed)) ? kAvx2Kernels : kScalarKernels;
}
}  // namespace

bool cpu_has_avx2() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

void set_force_scalar(bool on) { g_force_scalar.store(on); }
bool using_avx2() { return &kernels() == &kAvx2Kernels; }

void normalize_to_chw(const uint8_t* src, size_t npix, bool src_is_bgr, const NormalizeTarget* targets, int n) {
    kernels().normalize_to_chw(src, npix, src_is_bgr, targets, n);
}
void gemm_nt_chain(const float* A, int M, int lda, const float* B, int N, int ldb, int D, float* C, int ldc) {
    kernels().gemm_nt_chain(A, M, lda, B, N, ldb, D, C, ldc);
}
void gram_symmetric(const float* X, int N, int D, float* S) { kernels().gram_symmetric(X, N, D, S); }
void gemv_t_blas(const float* x, const float* A, int M, int D, float* y) { kernels().gemv_t_blas(x, A, M, D, y); }
float norm_sq(const float* x, int n) { return kernels().norm_sq(x, n); }
float sum_squares_numpy(const float* x, int n) { return kernels().sum_squares_numpy(x, n); }
void topk_rows(const float* S, int N, int k, bool exclude_diagonal, int* out) { kernels().topk_rows(S, N, k, exclude_diagonal, out); }
void transpose(const float* src, int rows, int cols, float* dst) { kernels().transpose(src, rows, cols, dst); }
void normalize_rows(float* x, int rows, int cols, float eps) { kernels().normalize_rows(x, rows, cols, eps); }
void sigmoid(const float* x, float* y, size_t n) { kernels().sigmoid(x, y, n); }
void clamp(const float* x, float* y, size_t n, float lo, float hi) { kernels().clamp(x, y, n, lo, hi); }
void greater_than(const float* x, uint8_t* out01, size_t n, float t) { kernels().greater_than(x, out01, n, t); }
void add_inplace(float* dst, const float* src, size_t n) { kernels().add_inplace(dst, src, n); }
void scale_inplace(float* x, size_t n, float s) { kernels().scale_inplace(x, n, s); }
void score_from_logit(const float* logit, const float* prior, float* score, size_t n, float w) {
    kernels().score_from_logit(logit, prior, score, n, w);
}
void update_best(const uint8_t* binary, const float* score, float* best, uint8_t* map, size_t n, uint8_t class_id) {
    kernels().update_best(binary, score, best, map, n, class_id);
}

}  // namespace pes::simd
