// AVX2 実装とスカラー実装の一致確認（乱数データ）と、AVX2 の速度の目安
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "simd.h"

using namespace pes::simd;

static int g_failed = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++g_failed;                                   \
            std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

static std::vector<float> rand_vec(size_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static double max_rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double)a[i] - b[i]) / (1e-6 + std::abs((double)b[i])));
    return m;
}
[[maybe_unused]] static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double)a[i] - b[i]));
    return m;
}

template <class F>
static double time_ms(F f, int reps = 5) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

int main() {
    std::printf("AVX2 対応: %s\n", cpu_has_avx2() ? "あり" : "なし（スカラー同士の比較になる）");

    // --- normalize_to_chw: 中途半端な画素数・BGR ---
    for (int npix : {1, 7, 9, 10, 17, 100, 1001}) {
        for (bool bgr : {false, true}) {
            std::mt19937 rng(npix);
            std::vector<uint8_t> img(npix * 3);
            for (auto& v : img) v = rng() & 255;
            std::vector<float> a0(npix * 3), a1(npix * 3), b0(npix * 3), b1(npix * 3);
            NormalizeTarget ta[2] = {{{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, a0.data()}, {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}, a1.data()}};
            NormalizeTarget tb[2] = {{{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, b0.data()}, {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}, b1.data()}};
            set_force_scalar(false);
            normalize_to_chw(img.data(), npix, bgr, ta, 2);
            set_force_scalar(true);
            normalize_to_chw(img.data(), npix, bgr, tb, 2);
            CHECK(a0 == b0 && a1 == b1, "normalize_to_chw npix=%d bgr=%d (ビット単位で一致するはず)", npix, (int)bgr);
        }
    }
    set_force_scalar(false);

    // --- gemm_nt_chain / norm_sq（AVX2 とスカラーでビット単位に一致する）---
    for (auto [M, N, D] : {std::array<int, 3>{1, 1, 1}, {9, 3, 7}, {8, 8, 16}, {17, 12, 192}, {1024, 8, 192}, {33, 20, 576}, {5, 9, 10}}) {
        auto A = rand_vec((size_t)M * D, 21), B = rand_vec((size_t)N * D, 22);
        std::vector<float> c1((size_t)M * N), c2((size_t)M * N);
        set_force_scalar(false);
        gemm_nt_chain(A.data(), M, D, B.data(), N, D, D, c1.data(), N);
        set_force_scalar(true);
        gemm_nt_chain(A.data(), M, D, B.data(), N, D, D, c2.data(), N);
        set_force_scalar(false);
        CHECK(c1 == c2, "gemm_nt_chain %dx%dx%d (ビット一致)", M, N, D);
        double err = 0;  // 倍精度の内積との差（丸め誤差の範囲）
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double ref = 0;
                for (int k = 0; k < D; ++k) ref += (double)A[(size_t)i * D + k] * B[(size_t)j * D + k];
                err = std::max(err, std::abs(ref - c1[(size_t)i * N + j]));
            }
        CHECK(err < 1e-3, "gemm_nt_chain は内積と近い値 err=%g", err);
    }
    for (int D : {1, 7, 8, 9, 100, 192, 576}) {
        auto a = rand_vec(D, 23);
        set_force_scalar(false);
        const float x = norm_sq(a.data(), D);
        set_force_scalar(true);
        const float y = norm_sq(a.data(), D);
        set_force_scalar(false);
        CHECK(x == y, "norm_sq D=%d (ビット一致) %.9g vs %.9g", D, x, y);
    }

    // --- gram_symmetric ---
    for (auto [N, D] : {std::array<int, 2>{1, 8}, {7, 9}, {13, 64}, {100, 192}, {1024, 256}}) {
        auto X = rand_vec((size_t)N * D, 5);
        std::vector<float> s1((size_t)N * N), s2((size_t)N * N);
        set_force_scalar(false);
        gram_symmetric(X.data(), N, D, s1.data());
        set_force_scalar(true);
        gram_symmetric(X.data(), N, D, s2.data());
        set_force_scalar(false);
        CHECK(s1 == s2, "gram N=%d D=%d (AVX2 とスカラーがビット一致)", N, D);
        std::vector<float> chain((size_t)N * N);
        gemm_nt_chain(X.data(), N, D, X.data(), N, D, D, chain.data(), N);
        CHECK(s1 == chain, "gram N=%d D=%d は gemm_nt_chain と同じ値", N, D);
        bool sym = true;
        for (int i = 0; i < N && sym; ++i)
            for (int j = 0; j < N; ++j)
                if (s1[(size_t)i * N + j] != s1[(size_t)j * N + i]) { sym = false; break; }
        CHECK(sym, "gram N=%d は対称であること", N);
    }

    // --- topk_rows（上位 k の集合が一致する） ---
    for (auto [N, k] : {std::array<int, 2>{9, 8}, {20, 8}, {100, 8}, {1024, 8}, {50, 3}}) {
        auto S = rand_vec((size_t)N * N, 6);
        for (bool excl : {true, false}) {
            std::vector<int> o1((size_t)N * k), o2((size_t)N * k);
            set_force_scalar(false);
            topk_rows(S.data(), N, k, excl, o1.data());
            set_force_scalar(true);
            topk_rows(S.data(), N, k, excl, o2.data());
            set_force_scalar(false);
            bool ok = true;
            for (int i = 0; i < N && ok; ++i) {
                std::set<int> a(o1.begin() + (size_t)i * k, o1.begin() + (size_t)(i + 1) * k);
                std::set<int> b(o2.begin() + (size_t)i * k, o2.begin() + (size_t)(i + 1) * k);
                ok = a == b && (!excl || !a.count(i));
                // 正解（ソート）とも比べる
                std::vector<std::pair<float, int>> all;
                for (int j = 0; j < N; ++j) if (!(excl && j == i)) all.emplace_back(S[(size_t)i * N + j], j);
                std::sort(all.begin(), all.end(), [](auto& x, auto& y) { return x.first > y.first; });
                std::set<int> ref;
                for (int t = 0; t < std::min<int>(k, all.size()); ++t) ref.insert(all[t].second);
                ok = ok && a == ref;
            }
            CHECK(ok, "topk N=%d k=%d excl=%d", N, k, (int)excl);
        }
    }

    // --- transpose ---
    for (auto [R, C] : {std::array<int, 2>{1, 1}, {8, 8}, {9, 17}, {256, 1024}, {32, 7}}) {
        auto X = rand_vec((size_t)R * C, 7);
        std::vector<float> t1((size_t)R * C), t2((size_t)R * C);
        set_force_scalar(false);
        transpose(X.data(), R, C, t1.data());
        set_force_scalar(true);
        transpose(X.data(), R, C, t2.data());
        set_force_scalar(false);
        CHECK(t1 == t2, "transpose %dx%d", R, C);
    }

    // --- normalize_rows ---
    {
        auto X = rand_vec(100 * 77, 8);
        auto Y = X;
        set_force_scalar(false);
        normalize_rows(X.data(), 100, 77, 1e-8f);
        set_force_scalar(true);
        normalize_rows(Y.data(), 100, 77, 1e-8f);
        set_force_scalar(false);
        CHECK(X == Y, "normalize_rows (ビット一致)");
    }
    for (int n : {1, 7, 8, 9, 31, 128, 129, 256, 257, 512, 1000, 1023}) {
        auto a = rand_vec(n, 30);
        set_force_scalar(false);
        const float x = sum_squares_numpy(a.data(), n);
        set_force_scalar(true);
        const float y = sum_squares_numpy(a.data(), n);
        set_force_scalar(false);
        CHECK(x == y, "sum_squares_numpy n=%d (ビット一致) %.9g vs %.9g", n, x, y);
        double ref = 0;
        for (float v : a) ref += (double)v * v;
        CHECK(std::abs(x - ref) < 1e-3 * (1 + ref), "sum_squares_numpy n=%d は二乗和と近い値", n);
    }
    for (auto [M, D] : {std::array<int, 2>{1, 512}, {2, 512}, {3, 512}, {4, 512}, {5, 512}, {29, 512}, {52, 512}, {23, 64}, {7, 24}, {6, 13}}) {
        auto x = rand_vec(D, 31), A = rand_vec((size_t)M * D, 32);
        std::vector<float> y1(M), y2(M);
        set_force_scalar(false);
        gemv_t_blas(x.data(), A.data(), M, D, y1.data());
        set_force_scalar(true);
        gemv_t_blas(x.data(), A.data(), M, D, y2.data());
        set_force_scalar(false);
        CHECK(y1 == y2, "gemv_t_blas M=%d D=%d (ビット一致)", M, D);
        double err = 0;
        for (int j = 0; j < M; ++j) {
            double ref = 0;
            for (int k = 0; k < D; ++k) ref += (double)x[k] * A[(size_t)j * D + k];
            err = std::max(err, std::abs(ref - y1[j]));
        }
        CHECK(err < 1e-3, "gemv_t_blas は内積と近い値 err=%g", err);
    }

    // --- sigmoid / clamp / greater_than / add / scale / score / update_best ---
    {
        auto X = rand_vec(1003, 9, -40.f, 40.f);
        std::vector<float> y1(X.size()), y2(X.size());
        set_force_scalar(false);
        sigmoid(X.data(), y1.data(), X.size());
        set_force_scalar(true);
        sigmoid(X.data(), y2.data(), X.size());
        set_force_scalar(false);
        CHECK(max_rel_err(y1, y2) < 5e-6, "sigmoid rel err=%g", max_rel_err(y1, y2));

        clamp(X.data(), y1.data(), X.size(), -32.f, 32.f);
        bool ok = true;
        for (size_t i = 0; i < X.size(); ++i) ok = ok && y1[i] == std::min(std::max(X[i], -32.f), 32.f);
        CHECK(ok, "clamp");

        std::vector<uint8_t> g1(X.size()), g2(X.size());
        set_force_scalar(false);
        greater_than(X.data(), g1.data(), X.size(), 0.0f);
        set_force_scalar(true);
        greater_than(X.data(), g2.data(), X.size(), 0.0f);
        set_force_scalar(false);
        CHECK(g1 == g2, "greater_than");

        auto Z = rand_vec(X.size(), 10);
        auto W = X;
        add_inplace(W.data(), Z.data(), W.size());
        ok = true;
        for (size_t i = 0; i < W.size(); ++i) ok = ok && W[i] == X[i] + Z[i];
        CHECK(ok, "add_inplace");
        scale_inplace(W.data(), W.size(), 0.25f);
        for (size_t i = 0; i < W.size(); ++i) ok = ok && W[i] == (X[i] + Z[i]) * 0.25f;
        CHECK(ok, "scale_inplace");

        auto P = rand_vec(X.size(), 11, 0.f, 1.f);
        auto L = rand_vec(X.size(), 12, -32.f, 32.f);
        set_force_scalar(false);
        score_from_logit(L.data(), P.data(), y1.data(), L.size(), 0.5f);
        set_force_scalar(true);
        score_from_logit(L.data(), P.data(), y2.data(), L.size(), 0.5f);
        set_force_scalar(false);
        CHECK(max_rel_err(y1, y2) < 5e-6, "score_from_logit rel err=%g", max_rel_err(y1, y2));

        // update_best: まばらな前景
        const size_t n = 5003;
        std::mt19937 rng(13);
        std::vector<uint8_t> bin(n, 0), m1(n, 0), m2(n, 0);
        for (size_t i = 0; i < n; ++i) bin[i] = (rng() % 5 == 0) || (i > 1000 && i < 1500) ? 1 : 0;
        auto sc = rand_vec(n, 14, 0.f, 1.f);
        auto b1 = rand_vec(n, 15, 0.f, 1.f), b2 = b1;
        set_force_scalar(false);
        update_best(bin.data(), sc.data(), b1.data(), m1.data(), n, 3);
        set_force_scalar(true);
        update_best(bin.data(), sc.data(), b2.data(), m2.data(), n, 3);
        set_force_scalar(false);
        CHECK(m1 == m2 && b1 == b2, "update_best");
    }

    // --- 速度の目安（実際のパイプラインの大きさ） ---
    {
        const int N = 1024, D = 256;
        auto X = rand_vec((size_t)N * D, 20);
        std::vector<float> S((size_t)N * N);
        std::vector<int> idx((size_t)N * 8);
        for (bool sc : {true, false}) {
            set_force_scalar(sc);
            const double t_gram = time_ms([&] { gram_symmetric(X.data(), N, D, S.data()); });
            const double t_topk = time_ms([&] { topk_rows(S.data(), N, 8, true, idx.data()); });
            std::vector<uint8_t> img(512 * 512 * 3, 100);
            std::vector<float> o1(512 * 512 * 3), o2(512 * 512 * 3);
            NormalizeTarget tg[2] = {{{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, o1.data()}, {{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}, o2.data()}};
            const double t_norm = time_ms([&] { normalize_to_chw(img.data(), 512 * 512, false, tg, 2); });
            std::vector<float> E((size_t)D * N), ET((size_t)N * D);
            const double t_tr = time_ms([&] { transpose(E.data(), D, N, ET.data()); });
            std::printf("[%s] gram 1024x256: %.2f ms / topk8: %.2f ms / 画像正規化 512x512 x2: %.2f ms / 転置 256x1024: %.2f ms\n",
                        sc ? "scalar" : "avx2  ", t_gram, t_topk, t_norm, t_tr);
        }
        set_force_scalar(false);
    }

    std::printf(g_failed ? "test_simd: %d 件失敗\n" : "test_simd: すべて成功\n", g_failed);
    return g_failed ? 1 : 0;
}
