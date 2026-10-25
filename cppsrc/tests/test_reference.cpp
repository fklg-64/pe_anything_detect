// C++ 版の後処理（mask_utils / coarse_segmentation）が Python 版（core/common）と同じ結果を返すことの確認。
// 正解データは scripts/gen_reference_cases.py で作る。使い方: test_reference <データのディレクトリ>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

#include <opencv2/imgproc.hpp>

#include "coarse_segmentation.h"
#include "mask_utils.h"
#include "npy.h"
#include "simd.h"

using namespace pes;

static std::string g_dir;
static int g_total_failed = 0;

static npy::Array load(const char* group, int index, const char* name) {
    char file[256];
    std::snprintf(file, sizeof(file), "/%s_%03d_%s.npy", group, index, name);
    return npy::load(g_dir + file);
}

// 2 次元の npy -> cv::Mat（コピー）
static cv::Mat to_mat(const npy::Array& a, int rows, int cols, int type) {
    cv::Mat m(rows, cols, type);
    std::memcpy(m.data, a.data.data(), a.data.size());
    return m;
}
static cv::Mat to_mat2d(const npy::Array& a, int type) { return to_mat(a, static_cast<int>(a.shape[0]), static_cast<int>(a.shape[1]), type); }

static bool equal_mat(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    cv::Mat diff;
    cv::compare(a.reshape(1), b.reshape(1), diff, cv::CMP_NE);
    return cv::countNonZero(diff) == 0;
}

static std::vector<cv::Point> to_points(const npy::Array& a) {
    std::vector<cv::Point> pts;
    for (size_t i = 0; i + 1 < a.size(); i += 2) pts.emplace_back(a.ptr<int32_t>()[i], a.ptr<int32_t>()[i + 1]);
    return pts;
}

static bool equal_points(const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) { return a == b; }

// 1 グループ分の実行。failed に不一致の添字を数える
static void run_group(const char* name, int count, const std::function<bool(int, std::string*)>& fn, double allowed_fraction = 0.0) {
    int failed = 0;
    std::string first_message;
    for (int t = 0; t < count; ++t) {
        std::string message;
        if (!fn(t, &message)) {
            if (failed == 0) first_message = "#" + std::to_string(t) + " " + message;
            ++failed;
        }
    }
    const bool ok = failed <= allowed_fraction * count;
    std::printf("%-12s %3d 件中 %3d 件が不一致 %s%s\n", name, count, failed, ok ? "" : "  <-- NG  ", failed ? first_message.c_str() : "");
    if (!ok) ++g_total_failed;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("使い方: test_reference <データのディレクトリ>\n");
        return 2;
    }
    g_dir = argv[1];
    try {
        const npy::Array counts_arr = npy::load(g_dir + "/counts.npy");
        const int32_t* n = counts_arr.ptr<int32_t>();

        run_group("fps", n[0], [&](int t, std::string*) {
            const auto pts = to_points(load("fps", t, "points"));
            const int k = load("fps", t, "k").ptr<int32_t>()[0];
            return equal_points(farthest_point_sampling(pts, k), to_points(load("fps", t, "out")));
        });

        run_group("prompt", n[1], [&](int t, std::string*) {
            const cv::Mat core = to_mat2d(load("prompt", t, "core"), CV_8U);
            const auto peaks = to_points(load("prompt", t, "peaks"));
            const int k = load("prompt", t, "k").ptr<int32_t>()[0];
            return equal_points(prompt_points(core, peaks, k), to_points(load("prompt", t, "out")));
        });

        run_group("bgpoints", n[2], [&](int t, std::string* msg) {
            const cv::Mat fg = to_mat2d(load("bgpoints", t, "fg"), CV_8U);
            const cv::Mat fmap = to_mat2d(load("bgpoints", t, "fmap"), CV_32F);
            const int count = load("bgpoints", t, "count").ptr<int32_t>()[0];
            const auto got = select_background_points(fg, count, fmap);
            const auto want = to_points(load("bgpoints", t, "out"));
            if (got != want) *msg = "点の数 " + std::to_string(got.size()) + " / " + std::to_string(want.size());
            return got == want;
        });

        run_group("components", n[3], [&](int t, std::string* msg) {
            const cv::Mat mask = to_mat2d(load("components", t, "mask"), CV_8U);
            const auto peaks = to_points(load("components", t, "peaks"));
            const npy::Array cores = load("components", t, "cores"), regions = load("components", t, "regions");
            const auto got = connected_components(mask);
            if (got.size() != peaks.size()) { *msg = "成分数"; return false; }
            for (size_t i = 0; i < got.size(); ++i) {
                cv::Mat want_core(32, 32, CV_8U, const_cast<char*>(cores.data.data()) + i * 32 * 32);
                cv::Mat want_region(32, 32, CV_8U, const_cast<char*>(regions.data.data()) + i * 32 * 32);
                if (got[i].peak != peaks[i]) { *msg = "peak"; return false; }
                if (!equal_mat(got[i].core, want_core)) { *msg = "core"; return false; }
                if (!equal_mat(got[i].region, want_region)) { *msg = "region"; return false; }
            }
            return true;
        });

        run_group("clean_mask", n[4], [&](int t, std::string*) {
            const cv::Mat mask = to_mat2d(load("clean", t, "mask"), CV_8U);
            const int min_area = load("clean", t, "min_area").ptr<int32_t>()[0];
            return equal_mat(clean_mask(mask, min_area), to_mat2d(load("clean", t, "out"), CV_8U));
        });

        run_group("remove_small", n[5], [&](int t, std::string*) {
            cv::Mat map = to_mat2d(load("remove", t, "map"), CV_8U);
            const int min_area = load("remove", t, "min_area").ptr<int32_t>()[0];
            remove_small_regions(map, min_area);
            return equal_mat(map, to_mat2d(load("remove", t, "out"), CV_8U));
        });

        // 距離が同じ画素のうちどれを「最寄り」にするかは scipy と実装が違うので、ごく一部の画素の違いは許す
        run_group("fill_gaps", n[6], [&](int t, std::string* msg) {
            cv::Mat map = to_mat2d(load("gaps", t, "map"), CV_8U);
            const cv::Mat fg = to_mat2d(load("gaps", t, "fg"), CV_8U);
            fill_narrow_gaps(map, fg);
            const cv::Mat want = to_mat2d(load("gaps", t, "out"), CV_8U);
            cv::Mat different;
            cv::compare(map, want, different, cv::CMP_NE);
            const int diff = cv::countNonZero(different);
            if (diff) *msg = std::to_string(diff) + " 画素の違い（全 " + std::to_string(map.total()) + "）";
            return diff == 0;
        });

        run_group("patch_mean", n[7], [&](int t, std::string*) {
            const cv::Mat in = to_mat2d(load("pmean", t, "in"), CV_32F);
            const cv::Mat want = to_mat2d(load("pmean", t, "out"), CV_32F);
            return cv::norm(patch_mean(in), want, cv::NORM_INF) < 1e-6;
        });

        run_group("guided", n[7], [&](int t, std::string* msg) {
            const cv::Mat guide = to_mat2d(load("guided", t, "guide"), CV_32F), src = to_mat2d(load("guided", t, "src"), CV_32F);
            const npy::Array params = load("guided", t, "params");
            const cv::Mat got = guided_filter(guide, src, static_cast<int>(params.ptr<double>()[0]), static_cast<float>(params.ptr<double>()[1]));
            const double err = cv::norm(got, to_mat2d(load("guided", t, "out"), CV_32F), cv::NORM_INF);
            if (err >= 1e-4) *msg = "最大誤差 " + std::to_string(err);
            return err < 1e-4;
        });

        // kNN の近傍（上位 8 個）は、8 位と 9 位の類似度がほぼ同じとき、足し算の順序の違い（BLAS と自前のカーネル）で入れ替わりうる。その分は少数なら許す
        run_group("propagate", n[8], [&](int t, std::string* msg) {
            const npy::Array embed = load("propagate", t, "embed"), conf = load("propagate", t, "conf"), prompt = load("propagate", t, "prompt");
            const int num_fg = load("propagate", t, "num_fg").ptr<int32_t>()[0];
            const int c = static_cast<int>(embed.shape[1]), count = static_cast<int>(embed.shape[2] * embed.shape[3]);
            std::vector<float> confidence(conf.ptr<float>(), conf.ptr<float>() + count);
            std::vector<int> prompts(count);
            for (int i = 0; i < count; ++i) prompts[i] = static_cast<int>(prompt.ptr<int64_t>()[i]);
            CoarseSegmenter::propagate_with_edgetam(confidence, prompts, embed.ptr<float>(), c, count, num_fg, 0.5);
            const npy::Array want_conf = load("propagate", t, "out_conf"), want_prompt = load("propagate", t, "out_prompt");
            double err = 0;
            for (int i = 0; i < count; ++i) err = std::max(err, std::abs(static_cast<double>(confidence[i]) - want_conf.ptr<float>()[i]));
            int prompt_diff = 0;
            for (int i = 0; i < count; ++i) prompt_diff += prompts[i] != want_prompt.ptr<int64_t>()[i];
            if (err > 1e-5 || prompt_diff) *msg = "確信度の最大誤差 " + std::to_string(err) + " / プロンプトの違い " + std::to_string(prompt_diff);
            return err <= 1e-5 && prompt_diff == 0;
        }, 0.1);

        run_group("coarse", n[9], [&](int t, std::string* msg) {
            const cv::Mat labels = to_mat2d(load("coarse", t, "labels"), CV_32S);
            const cv::Mat fg = to_mat2d(load("coarse", t, "fg"), CV_8U);
            cv::Mat out_mask, out_labels;
            CoarseSegmenter::refine_coarse_segmentation(labels, fg, &out_mask, &out_labels);
            if (!equal_mat(out_mask, to_mat2d(load("coarse", t, "out_mask"), CV_8U))) { *msg = "前景マスク"; return false; }
            if (!equal_mat(out_labels, to_mat2d(load("coarse", t, "out_labels"), CV_32S))) { *msg = "ラベル"; return false; }
            return true;
        });

        run_group("assign", n[9], [&](int t, std::string* msg) {
            const npy::Array cl = load("assign", t, "cluster_labels"), ci = load("assign", t, "cluster_index");
            std::vector<int> cluster_labels(cl.ptr<int32_t>(), cl.ptr<int32_t>() + cl.size()), index_list(ci.ptr<int32_t>(), ci.ptr<int32_t>() + ci.size());
            std::vector<int> labels_list;
            cv::Mat semantic;
            CoarseSegmenter::assign_labels(32, cluster_labels, index_list, &labels_list, &semantic);
            const npy::Array ll = load("assign", t, "labels_list");
            if (labels_list != std::vector<int>(ll.ptr<int32_t>(), ll.ptr<int32_t>() + ll.size())) { *msg = "labels_list"; return false; }
            return equal_mat(semantic, to_mat2d(load("assign", t, "semantic"), CV_32S));
        });

        run_group("nearest", n[10], [&](int t, std::string* msg) {
            const npy::Array fa = load("nearest", t, "feature");
            const cv::Mat feature = to_mat2d(fa, CV_8U);
            const cv::Mat want_dist2 = to_mat2d(load("nearest", t, "dist2"), CV_32S);
            const cv::Mat index = nearest_feature_index(feature);
            for (int y = 0; y < feature.rows; ++y)
                for (int x = 0; x < feature.cols; ++x) {
                    const int idx = index.at<int>(y, x);
                    if (idx < 0 || !feature.at<uint8_t>(idx / feature.cols, idx % feature.cols)) { *msg = "特徴画素を指していない"; return false; }
                    const int dy = y - idx / feature.cols, dx = x - idx % feature.cols;
                    if (dy * dy + dx * dx != want_dist2.at<int>(y, x)) { *msg = "距離が最短でない"; return false; }
                }
            return true;
        });

        // --- NumPy / BLAS の丸め方の再現（ビット単位で一致するはず）---
        run_group("numpy_ops", n[11], [&](int t, std::string* msg) {
            const npy::Array x = load("numpy", t, "x");
            const int rows = static_cast<int>(x.shape[0]), cols = static_cast<int>(x.shape[1]);
            const float eps = load("numpy", t, "eps").ptr<float>()[0];
            const npy::Array want_sum = load("numpy", t, "sumsq"), want_norm = load("numpy", t, "normalized");
            for (int r = 0; r < rows; ++r)
                if (simd::sum_squares_numpy(x.ptr<float>() + static_cast<size_t>(r) * cols, cols) != want_sum.ptr<float>()[r]) { *msg = "二乗和"; return false; }
            std::vector<float> normalized(x.ptr<float>(), x.ptr<float>() + x.size());
            simd::normalize_rows(normalized.data(), rows, cols, eps);
            for (size_t i = 0; i < normalized.size(); ++i)
                if (normalized[i] != want_norm.ptr<float>()[i]) { *msg = "ノルムで割る"; return false; }
            return true;
        });

        // M=1 は NumPy が sgemv ではなく sdot を使う（本パイプラインでは前景・背景が 1 つ以上ずつなので M >= 2）。M >= 2 はビット単位で一致する
        run_group("gemv", n[12], [&](int t, std::string* msg) {
            const npy::Array f = load("gemv", t, "feature"), text = load("gemv", t, "text"), want = load("gemv", t, "out");
            const int m = static_cast<int>(text.shape[0]), d = static_cast<int>(text.shape[1]);
            if (m < 2) return true;
            std::vector<float> scaled(f.ptr<float>(), f.ptr<float>() + d), y(m);
            for (float& v : scaled) v *= 100.0f;
            simd::gemv_t_blas(scaled.data(), text.ptr<float>(), m, d, y.data());
            int diff = 0;
            for (int j = 0; j < m; ++j) diff += y[j] != want.ptr<float>()[j];
            if (diff) *msg = "M=" + std::to_string(m) + " D=" + std::to_string(d) + " で " + std::to_string(diff) + " 列が不一致";
            return diff == 0;
        });

        // X @ X.T は、大きい行列では OpenBLAS の sgemm の内部（タイルごとの足し込みの順序）が直列な FMA の連鎖と異なり、1〜2 ulp ずれる要素がある。
        // 小さい行列（K が小さい・行数が少ない）では一致する。ここでは誤差が 1e-6 以内であることを確認する（kNN の上位 8 個の選択にしか効かない）
        run_group("gram", n[13], [&](int t, std::string* msg) {
            const npy::Array x = load("gram", t, "x"), want = load("gram", t, "out");
            const int count = static_cast<int>(x.shape[0]), c = static_cast<int>(x.shape[1]);
            std::vector<float> s(static_cast<size_t>(count) * count);
            simd::gram_symmetric(x.ptr<float>(), count, c, s.data());
            double err = 0;
            for (size_t i = 0; i < s.size(); ++i) err = std::max(err, std::abs(static_cast<double>(s[i]) - want.ptr<float>()[i]));
            if (err >= 1e-6) *msg = "N=" + std::to_string(count) + " C=" + std::to_string(c) + " 最大誤差 " + std::to_string(err);
            return err < 1e-6;
        });

        // 確信度は softmax の exp が NumPy の SIMD 実装と最後の 1〜2 ulp 異なりうるので、完全一致ではなく誤差で見る
        int exact_confidence = 0;
        run_group("classify", n[14], [&](int t, std::string* msg) {
            const npy::Array feature = load("classify", t, "feature"), text = load("classify", t, "text");
            const npy::Array meta = load("classify", t, "meta"), thr = load("classify", t, "thr"), want = load("classify", t, "out"), idx = load("classify", t, "idx");
            const int num_fg = meta.ptr<int32_t>()[0], num_bg = meta.ptr<int32_t>()[1], d = static_cast<int>(feature.size());
            const auto got = classify_cluster_feature(feature.ptr<float>(), d, text.ptr<float>(), num_fg, num_bg, thr.ptr<double>()[0]);
            const double want_conf = want.ptr<double>()[0];
            exact_confidence += static_cast<double>(got.first) == want_conf;
            const bool ok = std::abs(got.first - want_conf) < 2e-7 && got.second == idx.ptr<int32_t>()[0];
            if (!ok) *msg = "確信度 " + std::to_string(got.first) + " vs " + std::to_string(want_conf) + " / 番号 " + std::to_string(got.second) + " vs " + std::to_string(idx.ptr<int32_t>()[0]);
            return ok;
        });
        std::printf("  (classify: 確信度が NumPy とビット単位で一致したのは %d / %d 件)\n", exact_confidence, n[14]);
    } catch (const std::exception& e) {
        std::printf("エラー: %s\n", e.what());
        return 2;
    }
    std::printf(g_total_failed ? "test_reference: %d グループで不一致\n" : "test_reference: すべて一致\n", g_total_failed);
    return g_total_failed ? 1 : 0;
}
