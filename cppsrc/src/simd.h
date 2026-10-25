// AVX2 + FMA で高速化した数値カーネル。実行時に CPU を調べ、非対応ならスカラー実装に切り替える。
// AVX2 の実装は simd_avx2.cpp（-mavx2 -mfma でビルド）、スカラーの実装は simd.cpp。両者は同じ結果を返す
// （sigmoid / score_from_logit だけは AVX2 が exp を近似するので ~1e-7 の誤差。他はビット単位で一致。tests/test_simd.cpp で確認）。
#pragma once

#include <cstddef>
#include <cstdint>

namespace pes::simd {

// 実行中の CPU が AVX2 + FMA に対応しているか
bool cpu_has_avx2();
// true にするとスカラー実装を使う（AVX2 実装との一致確認・比較用）。false で CPU の対応に従う
void set_force_scalar(bool on);
// 現在 AVX2 実装を使っているか
bool using_avx2();

// --- 画像の前処理 ---
// インターリーブの 8bit 3 チャンネル画像を、((x / 255) - mean) / std で正規化した CHW (RGB 順) の float にする。
// 複数の正規化（CLIP と EdgeTAM）を、画像を 1 回読むだけで同時に書き出す。
struct NormalizeTarget {
    float mean[3];
    float stdv[3];
    float* dst;  // 3 * npix 個（チャンネルごとに npix 個が連続）
};
void normalize_to_chw(const uint8_t* src, size_t npix, bool src_is_bgr, const NormalizeTarget* targets, int num_targets);

// --- 内積・行列積 ---
// 以下のカーネルは、Python 版が使う NumPy / OpenBLAS / faiss の丸め方（足し込みの順序・FMA の有無）をそろえてある。
// 丸め方が違うと、前景確信度に ~1e-6 の差が出て、確信度がほぼ同じパッチが並ぶとき（背景の負例点の選択など）に
// Python 版と別の結果になることがあるため。
//
// C = A Bᵀ（A: M × D、B: N × D）。各要素は k = 0, 1, ... の順の直列な FMA の連鎖（OpenBLAS の sgemm と同じ。KMeans の割り当て用）
void gemm_nt_chain(const float* A, int M, int lda, const float* B, int N, int ldb, int D, float* C, int ldc);
// S = X Xᵀ（N × N、行優先。X: N × D）。要素の丸め方は gemm_nt_chain と同じ。対称性を使って上三角だけ計算する。
// NumPy の X @ X.T は、大きい行列（OpenBLAS のタイル内の足し込みの順序が違う）では 1〜2 ulp 違う要素がある（tests/test_reference.cpp）
void gram_symmetric(const float* X, int N, int D, float* S);
// y[j] = x · A の第 j 行（A: M × D の行優先、x: 長さ D）。NumPy の「ベクトル @ 行列」（OpenBLAS の sgemv）と同じ丸め方:
// 4 行ずつは 8 レーンの FMA、余りの 2 行は SSE 4 レーンの mul+add、余りの 1 行は AVX 8 レーンの mul+add
void gemv_t_blas(const float* x, const float* A, int M, int D, float* y);
// 二乗ノルム。8 レーンの FMA で足し込み、レーンを畳む順序も固定（faiss の fvec_norm_L2sqr と同じ）
float norm_sq(const float* x, int n);
// np.add.reduce(x * x) と同じ丸め（x * x を float32 に丸めてから pairwise summation）
float sum_squares_numpy(const float* x, int n);

// 各行の値が大きい上位 k 個の列番号を out（N × k、順不同）に書く。exclude_diagonal なら対角（行番号と同じ列）を除く
void topk_rows(const float* S, int N, int k, bool exclude_diagonal, int* out);

// --- 配列の変形・要素ごとの演算 ---
void transpose(const float* src, int rows, int cols, float* dst);  // rows × cols -> cols × rows
// 行ごとに x / (np.linalg.norm(x) + eps)。NumPy と同じ丸め（sum_squares_numpy の sqrt）
void normalize_rows(float* x, int rows, int cols, float eps);
void sigmoid(const float* x, float* y, size_t n);                  // 1 / (1 + exp(-x))
void clamp(const float* x, float* y, size_t n, float lo, float hi);
void greater_than(const float* x, uint8_t* out01, size_t n, float threshold);  // out = (x > threshold) ? 1 : 0
void add_inplace(float* dst, const float* src, size_t n);          // dst += src
void scale_inplace(float* x, size_t n, float s);                   // x *= s
// score = sigmoid(logit) * ((1 - w) + w * prior)（EdgeTAM の確信度と粗い前景らしさの積。重なり解決用）
void score_from_logit(const float* logit, const float* prior, float* score, size_t n, float w);
// 重なり解決: binary が立ち、かつ score > best の画素を、semantic_map = class_id、best = score に更新する
void update_best(const uint8_t* binary, const float* score, float* best, uint8_t* semantic_map, size_t n, uint8_t class_id);

// --- 実装（simd.cpp が振り分ける） ---
struct Kernels {
    void (*normalize_to_chw)(const uint8_t*, size_t, bool, const NormalizeTarget*, int);
    void (*gemm_nt_chain)(const float*, int, int, const float*, int, int, int, float*, int);
    void (*gram_symmetric)(const float*, int, int, float*);
    void (*gemv_t_blas)(const float*, const float*, int, int, float*);
    float (*norm_sq)(const float*, int);
    float (*sum_squares_numpy)(const float*, int);
    void (*topk_rows)(const float*, int, int, bool, int*);
    void (*transpose)(const float*, int, int, float*);
    void (*normalize_rows)(float*, int, int, float);
    void (*sigmoid)(const float*, float*, size_t);
    void (*clamp)(const float*, float*, size_t, float, float);
    void (*greater_than)(const float*, uint8_t*, size_t, float);
    void (*add_inplace)(float*, const float*, size_t);
    void (*scale_inplace)(float*, size_t, float);
    void (*score_from_logit)(const float*, const float*, float*, size_t, float);
    void (*update_best)(const uint8_t*, const float*, float*, uint8_t*, size_t, uint8_t);
};
extern const Kernels kScalarKernels;
extern const Kernels kAvx2Kernels;  // simd_avx2.cpp

}  // namespace pes::simd
