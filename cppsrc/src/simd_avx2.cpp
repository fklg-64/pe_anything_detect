// AVX2 + FMA 実装。このファイルだけ -mavx2 -mfma でビルドする（他のファイルは汎用の命令のまま）。
// 呼び出し側は simd.h の振り分けを通すので、AVX2 非対応の CPU でこのファイルの命令が実行されることはない。
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "simd.h"

namespace pes::simd {
namespace {

// -----------------------------------------------------------------------------
// 画像の正規化: インターリーブ u8 -> CHW float (複数の正規化を同時に)
// -----------------------------------------------------------------------------

// 8 ピクセル（24 バイト）の sc 番目のチャンネルを int32 x 8 にする。p から 28 バイト先まで読む
inline __m256i load_channel8(const uint8_t* p, __m128i mask) {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));        // ピクセル 0〜4
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 12));   // ピクセル 4〜7（+ 余分）
    const __m128i a = _mm_shuffle_epi8(lo, mask);  // ピクセル 0〜3 のチャンネル値を下位 4 バイトへ
    const __m128i b = _mm_shuffle_epi8(hi, mask);  // ピクセル 4〜7
    return _mm256_cvtepu8_epi32(_mm_unpacklo_epi32(a, b));
}

void normalize_to_chw_avx2(const uint8_t* src, size_t npix, bool src_is_bgr, const NormalizeTarget* targets, int num_targets) {
    __m128i masks[3];
    for (int c = 0; c < 3; ++c) {
        const char sc = static_cast<char>(src_is_bgr ? 2 - c : c);
        masks[c] = _mm_setr_epi8(sc, static_cast<char>(sc + 3), static_cast<char>(sc + 6), static_cast<char>(sc + 9), -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1);
    }
    const __m256 v255 = _mm256_set1_ps(255.0f);

    size_t i = 0;
    // 16 バイトのロードが 28 バイト先まで届くので、ピクセル i から 10 個分が確保されている間だけベクトル処理する
    for (; i + 10 <= npix; i += 8) {
        for (int c = 0; c < 3; ++c) {
            const __m256 x = _mm256_div_ps(_mm256_cvtepi32_ps(load_channel8(src + i * 3, masks[c])), v255);
            for (int t = 0; t < num_targets; ++t) {
                const NormalizeTarget& tg = targets[t];
                const __m256 y = _mm256_div_ps(_mm256_sub_ps(x, _mm256_set1_ps(tg.mean[c])), _mm256_set1_ps(tg.stdv[c]));
                _mm256_storeu_ps(tg.dst + static_cast<size_t>(c) * npix + i, y);
            }
        }
    }
    for (; i < npix; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = static_cast<float>(src[i * 3 + (src_is_bgr ? 2 - c : c)]) / 255.0f;
            for (int t = 0; t < num_targets; ++t) {
                const NormalizeTarget& tg = targets[t];
                tg.dst[static_cast<size_t>(c) * npix + i] = (v - tg.mean[c]) / tg.stdv[c];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 内積・行列積（C = A Bᵀ）
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 内積・行列積（BLAS と同じ丸め方）
// -----------------------------------------------------------------------------

// C = A Bᵀ を、各要素 k = 0, 1, ... の順の直列な FMA の連鎖で求める。B を 8 行ずつ (D × 8) に転置して詰め、
// 8 個の行 A について同時に（FMA の遅延を隠すため独立な連鎖を 8 本並べて）進める。各レーンは独立なので、結果は要素ごとの直列な FMA と同じ
void gemm_nt_chain_avx2(const float* A, int M, int lda, const float* B, int N, int ldb, int D, float* C, int ldc) {
    std::vector<float> packed(static_cast<size_t>(D) * 8);
    for (int j0 = 0; j0 < N; j0 += 8) {
        const int nr = std::min(8, N - j0);
        for (int t = 0; t < D; ++t) {
            float* dst = packed.data() + static_cast<size_t>(t) * 8;
            for (int l = 0; l < 8; ++l) dst[l] = l < nr ? B[static_cast<size_t>(j0 + l) * ldb + t] : 0.0f;
        }
        for (int i0 = 0; i0 < M; i0 += 8) {
            const int mr = std::min(8, M - i0);
            const float* rows[8];
            for (int r = 0; r < 8; ++r) rows[r] = A + static_cast<size_t>(i0 + (r < mr ? r : 0)) * lda;  // 余った行は 0 行目の複製（結果は捨てる）
            __m256 acc[8];
            for (int r = 0; r < 8; ++r) acc[r] = _mm256_setzero_ps();
            for (int t = 0; t < D; ++t) {
                const __m256 b = _mm256_loadu_ps(packed.data() + static_cast<size_t>(t) * 8);
                for (int r = 0; r < 8; ++r) acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(rows[r] + t), b, acc[r]);
            }
            for (int r = 0; r < mr; ++r) {
                float lane[8];
                _mm256_storeu_ps(lane, acc[r]);
                for (int l = 0; l < nr; ++l) C[static_cast<size_t>(i0 + r) * ldc + j0 + l] = lane[l];
            }
        }
    }
}

// 二乗ノルム: 8 レーンの FMA -> 上下 128bit を足す -> hadd を 2 回（simd.cpp のスカラー版と同じ順序）
float norm_sq_avx2(const float* x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    float s = _mm_cvtss_f32(q);
    for (; i < n; ++i) s += x[i] * x[i];
    return s;
}

// S = X Xᵀ。gemm_nt_chain_avx2 と同じ丸め方（要素ごとに直列な FMA の連鎖）で、8×8 のタイルのうち上三角（対角のタイルを含む）だけ計算して、下三角に写す
void gram_symmetric_avx2(const float* X, int N, int D, float* S) {
    std::vector<float> packed(static_cast<size_t>(D) * 8);
    for (int j0 = 0; j0 < N; j0 += 8) {
        const int nr = std::min(8, N - j0);
        for (int t = 0; t < D; ++t) {
            float* dst = packed.data() + static_cast<size_t>(t) * 8;
            for (int l = 0; l < 8; ++l) dst[l] = l < nr ? X[static_cast<size_t>(j0 + l) * D + t] : 0.0f;
        }
        for (int i0 = 0; i0 <= j0; i0 += 8) {
            const int mr = std::min(8, N - i0);
            const float* rows[8];
            for (int r = 0; r < 8; ++r) rows[r] = X + static_cast<size_t>(i0 + (r < mr ? r : 0)) * D;
            __m256 acc[8];
            for (int r = 0; r < 8; ++r) acc[r] = _mm256_setzero_ps();
            for (int t = 0; t < D; ++t) {
                const __m256 b = _mm256_loadu_ps(packed.data() + static_cast<size_t>(t) * 8);
                for (int r = 0; r < 8; ++r) acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(rows[r] + t), b, acc[r]);
            }
            for (int r = 0; r < mr; ++r) {
                float lane[8];
                _mm256_storeu_ps(lane, acc[r]);
                for (int l = 0; l < nr; ++l) {
                    S[static_cast<size_t>(i0 + r) * N + (j0 + l)] = lane[l];
                    S[static_cast<size_t>(j0 + l) * N + (i0 + r)] = lane[l];
                }
            }
        }
    }
}

// 8 レーンを (上下 128bit を足す) -> hadd -> hadd の順に畳む（simd.cpp の fold8 と同じ）
inline float fold8_avx2(__m256 v) {
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    return _mm_cvtss_f32(q);
}

// y[j] = x · A[j]。OpenBLAS sgemv_t (Haswell) の再現（simd.cpp のスカラー版と同じ丸め方）
void gemv_t_blas_avx2(const float* x, const float* A, int M, int D, float* y) {
    const int groups = M & ~3;
    const int d8 = D & ~7, d4 = D & ~3;
    int j = 0;
    // 4 行ずつ: 8 レーンの FMA（4 行を同時に進めて FMA の遅延を隠す）
    for (; j < groups; j += 4) {
        const float* a0 = A + static_cast<size_t>(j) * D;
        const float* a1 = a0 + D;
        const float* a2 = a1 + D;
        const float* a3 = a2 + D;
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        for (int k = 0; k < d8; k += 8) {
            const __m256 xv = _mm256_loadu_ps(x + k);
            c0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a0 + k), c0);
            c1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a1 + k), c1);
            c2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a2 + k), c2);
            c3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(a3 + k), c3);
        }
        float r[4] = {fold8_avx2(c0), fold8_avx2(c1), fold8_avx2(c2), fold8_avx2(c3)};
        for (int k = d8; k < D; ++k)
            for (int c = 0; c < 4; ++c) r[c] += x[k] * a0[c * D + k];
        for (int c = 0; c < 4; ++c) y[j + c] = r[c];
    }
    // 余りの 2 行: SSE 4 レーンの mul + add
    if (M & 2) {
        for (int c = 0; c < 2; ++c, ++j) {
            const float* a = A + static_cast<size_t>(j) * D;
            __m128 acc = _mm_setzero_ps();
            for (int k = 0; k < d4; k += 4) acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(x + k), _mm_loadu_ps(a + k)));
            acc = _mm_hadd_ps(acc, acc);
            acc = _mm_hadd_ps(acc, acc);
            float result = _mm_cvtss_f32(acc);
            for (int k = d4; k < D; ++k) result += x[k] * a[k];
            y[j] = result;
        }
    }
    // 余りの 1 行: AVX 8 レーンの mul + add
    if (M & 1) {
        const float* a = A + static_cast<size_t>(j) * D;
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < d8; k += 8) acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(a + k)));
        float result = fold8_avx2(acc);
        for (int k = d8; k < D; ++k) result += x[k] * a[k];
        y[j] = result;
    }
}

// NumPy の pairwise summation（float32）。要素は a[i] * a[i] を float32 に丸めたもの。葉（8〜128 要素）は 8 レーンの add でそのまま書ける
float pairwise_sum_squares_avx2(const float* a, long n) {
    if (n < 8) {
        float res = 0.0f;
        for (long i = 0; i < n; ++i) {
            const float sq = a[i] * a[i];
            res += sq;
        }
        return res;
    }
    if (n <= 128) {
        const __m256 first = _mm256_loadu_ps(a);
        __m256 r = _mm256_mul_ps(first, first);
        long i;
        for (i = 8; i < n - (n % 8); i += 8) {
            const __m256 v = _mm256_loadu_ps(a + i);
            r = _mm256_add_ps(r, _mm256_mul_ps(v, v));
        }
        float lane[8];
        _mm256_storeu_ps(lane, r);
        float res = ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        for (; i < n; ++i) {
            const float sq = a[i] * a[i];
            res += sq;
        }
        return res;
    }
    long n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum_squares_avx2(a, n2) + pairwise_sum_squares_avx2(a + n2, n - n2);
}

float sum_squares_numpy_avx2(const float* x, int n) { return pairwise_sum_squares_avx2(x, n); }

// -----------------------------------------------------------------------------
// 各行の上位 k 個
// -----------------------------------------------------------------------------

void topk_rows_avx2(const float* S, int N, int k, bool exclude_diagonal, int* out) {
    constexpr int kMaxK = 64;
    const int kk = std::min(k, kMaxK);
    float vals[kMaxK];
    int idxs[kMaxK];
    for (int i = 0; i < N; ++i) {
        const float* row = S + static_cast<size_t>(i) * N;
        int count = 0;
        float threshold = -std::numeric_limits<float>::infinity();
        int worst = 0;

        auto consider = [&](int j) {
            if (exclude_diagonal && j == i) return;
            const float v = row[j];
            if (count < kk) {
                vals[count] = v;
                idxs[count] = j;
                ++count;
                if (count == kk) {
                    worst = 0;
                    for (int t = 1; t < kk; ++t) if (vals[t] < vals[worst]) worst = t;
                    threshold = vals[worst];
                }
            } else if (v > threshold) {
                vals[worst] = v;
                idxs[worst] = j;
                worst = 0;
                for (int t = 1; t < kk; ++t) if (vals[t] < vals[worst]) worst = t;
                threshold = vals[worst];
            }
        };

        int j = 0;
        // 先頭の kk 個ぶんは無条件に入れる必要があるので、閾値が決まるまではスカラーで進める
        for (; j < N && count < kk; ++j) consider(j);
        const int stop = N - 7;
        for (; j < stop; j += 8) {
            const __m256 v = _mm256_loadu_ps(row + j);
            unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_set1_ps(threshold), _CMP_GT_OQ)));
            while (mask) {
                const int bit = __builtin_ctz(mask);
                mask &= mask - 1;
                consider(j + bit);  // 閾値が途中で上がるので、consider の中でも改めて比較する
            }
        }
        for (; j < N; ++j) consider(j);
        for (int t = 0; t < k; ++t) out[static_cast<size_t>(i) * k + t] = t < count ? idxs[t] : -1;
    }
}

// -----------------------------------------------------------------------------
// 転置（8x8 ブロック）
// -----------------------------------------------------------------------------

inline void transpose8x8(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)), s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)), s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0)), s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0)), s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

void transpose_avx2(const float* src, int rows, int cols, float* dst) {
    const int rows8 = rows & ~7, cols8 = cols & ~7;
    for (int r = 0; r < rows8; r += 8) {
        for (int c = 0; c < cols8; c += 8) {
            const float* s = src + static_cast<size_t>(r) * cols + c;
            __m256 v0 = _mm256_loadu_ps(s), v1 = _mm256_loadu_ps(s + cols), v2 = _mm256_loadu_ps(s + 2 * cols),
                   v3 = _mm256_loadu_ps(s + 3 * cols), v4 = _mm256_loadu_ps(s + 4 * cols), v5 = _mm256_loadu_ps(s + 5 * cols),
                   v6 = _mm256_loadu_ps(s + 6 * cols), v7 = _mm256_loadu_ps(s + 7 * cols);
            transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7);
            float* d = dst + static_cast<size_t>(c) * rows + r;
            _mm256_storeu_ps(d, v0);
            _mm256_storeu_ps(d + rows, v1);
            _mm256_storeu_ps(d + 2 * rows, v2);
            _mm256_storeu_ps(d + 3 * rows, v3);
            _mm256_storeu_ps(d + 4 * rows, v4);
            _mm256_storeu_ps(d + 5 * rows, v5);
            _mm256_storeu_ps(d + 6 * rows, v6);
            _mm256_storeu_ps(d + 7 * rows, v7);
        }
    }
    // 端（8 の倍数でない行・列）
    for (int r = 0; r < rows; ++r)
        for (int c = (r < rows8 ? cols8 : 0); c < cols; ++c) dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
}

// -----------------------------------------------------------------------------
// 要素ごとの演算
// -----------------------------------------------------------------------------

void normalize_rows_avx2(float* x, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<size_t>(r) * cols;
        const float denom = std::sqrt(pairwise_sum_squares_avx2(row, cols)) + eps;
        const __m256 vd = _mm256_set1_ps(denom);
        int c = 0;
        for (; c + 8 <= cols; c += 8) _mm256_storeu_ps(row + c, _mm256_div_ps(_mm256_loadu_ps(row + c), vd));
        for (; c < cols; ++c) row[c] = row[c] / denom;
    }
}

// exp(x)（Cephes の単精度版。相対誤差 約 1e-7）
inline __m256 exp256(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    __m256i n = _mm256_cvttps_epi32(fx);
    n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

void sigmoid_avx2(const float* x, float* y, size_t n) {
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 e = exp256(_mm256_sub_ps(zero, _mm256_loadu_ps(x + i)));
        _mm256_storeu_ps(y + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    }
    for (; i < n; ++i) y[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

void clamp_avx2(const float* x, float* y, size_t n, float lo, float hi) {
    const __m256 vlo = _mm256_set1_ps(lo), vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(x + i), vlo), vhi));
    for (; i < n; ++i) y[i] = std::min(std::max(x[i], lo), hi);
}

void greater_than_avx2(const float* x, uint8_t* out, size_t n, float t) {
    const __m256 vt = _mm256_set1_ps(t);
    const __m256i one = _mm256_set1_epi8(1);
    size_t i = 0;
    // 32 要素ずつ: 比較 -> 32bit のマスク x4 -> 8bit に詰める
    for (; i + 32 <= n; i += 32) {
        const __m256i m0 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(x + i), vt, _CMP_GT_OQ));
        const __m256i m1 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(x + i + 8), vt, _CMP_GT_OQ));
        const __m256i m2 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(x + i + 16), vt, _CMP_GT_OQ));
        const __m256i m3 = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(x + i + 24), vt, _CMP_GT_OQ));
        // 32bit -> 16bit -> 8bit（レーンが入れ替わるので最後に並べ直す）
        const __m256i p01 = _mm256_packs_epi32(m0, m1);
        const __m256i p23 = _mm256_packs_epi32(m2, m3);
        __m256i p = _mm256_packs_epi16(p01, p23);
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_and_si256(p, one));
    }
    for (; i < n; ++i) out[i] = x[i] > t ? 1 : 0;
}

void add_inplace_avx2(float* dst, const float* src, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    for (; i < n; ++i) dst[i] += src[i];
}

void scale_inplace_avx2(float* x, size_t n, float s) {
    const __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    for (; i < n; ++i) x[i] *= s;
}

void score_from_logit_avx2(const float* logit, const float* prior, float* score, size_t n, float w) {
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    const __m256 vw = _mm256_set1_ps(w), vomw = _mm256_set1_ps(1.0f - w);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 e = exp256(_mm256_sub_ps(zero, _mm256_loadu_ps(logit + i)));
        const __m256 conf = _mm256_div_ps(one, _mm256_add_ps(one, e));
        const __m256 u = _mm256_add_ps(vomw, _mm256_mul_ps(vw, _mm256_loadu_ps(prior + i)));
        _mm256_storeu_ps(score + i, _mm256_mul_ps(conf, u));
    }
    for (; i < n; ++i) {
        const float conf = 1.0f / (1.0f + std::exp(-logit[i]));
        const float wp = w * prior[i];
        const float u = (1.0f - w) + wp;
        score[i] = conf * u;
    }
}

// 8 画素分の勝敗ビット（下位 8 ビット）を、画素ごとの 0x00 / 0xFF のバイトマスク（64bit）に展開する表
const uint64_t* byte_mask_table() {
    static const auto table = [] {
        std::array<uint64_t, 256> t{};
        for (int bits = 0; bits < 256; ++bits) {
            uint64_t m = 0;
            for (int b = 0; b < 8; ++b)
                if (bits & (1 << b)) m |= uint64_t(0xFF) << (8 * b);
            t[bits] = m;
        }
        return t;
    }();
    return table.data();
}

void update_best_avx2(const uint8_t* binary, const float* score, float* best, uint8_t* map, size_t n, uint8_t class_id) {
    const uint64_t* byte_mask = byte_mask_table();
    const uint64_t class_bytes = 0x0101010101010101ULL * class_id;
    const __m256i vzero = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t b8;
        std::memcpy(&b8, binary + i, 8);
        if (b8 == 0) continue;  // この 8 画素に前景が無ければ何もしない（ほとんどの画素）
        const __m256i b32 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(b8)));
        const __m256 is_fg = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(b32, vzero), _mm256_set1_epi32(-1)));  // binary != 0
        const __m256 s = _mm256_loadu_ps(score + i), bst = _mm256_loadu_ps(best + i);
        const __m256 win = _mm256_and_ps(is_fg, _mm256_cmp_ps(s, bst, _CMP_GT_OQ));
        const unsigned wmask = static_cast<unsigned>(_mm256_movemask_ps(win));
        if (!wmask) continue;
        _mm256_storeu_ps(best + i, _mm256_blendv_ps(bst, s, win));
        uint64_t old;
        std::memcpy(&old, map + i, 8);
        const uint64_t m = byte_mask[wmask];
        const uint64_t updated = (old & ~m) | (class_bytes & m);
        std::memcpy(map + i, &updated, 8);
    }
    for (; i < n; ++i) {
        if (binary[i] && score[i] > best[i]) {
            map[i] = class_id;
            best[i] = score[i];
        }
    }
}

}  // namespace

const Kernels kAvx2Kernels = {
    normalize_to_chw_avx2, gemm_nt_chain_avx2, gram_symmetric_avx2, gemv_t_blas_avx2, norm_sq_avx2, sum_squares_numpy_avx2,
    topk_rows_avx2,        transpose_avx2,     normalize_rows_avx2, sigmoid_avx2,     clamp_avx2,    greater_than_avx2,
    add_inplace_avx2,      scale_inplace_avx2, score_from_logit_avx2, update_best_avx2,
};

}  // namespace pes::simd
