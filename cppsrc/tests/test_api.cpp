// C API (pe_segmentation.h) の通しの確認。モデルを読み込んで実際に動かす。
//   - モデル読み込み・テキスト特徴量抽出・処理実行
//   - プロンプトの追加・削除（追加削除のたびに特徴量が更新され、元に戻せば同じ結果になる）
//   - エラーの戻り値
// 使い方: test_api <リポジトリのルート> [画像]   （既定の画像は MVTec の capsule。無ければ合成画像）
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pe_segmentation.h"

static int g_failed = 0;
#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            ++g_failed;                                           \
            std::printf("  FAIL %s:%d  ", __FILE__, __LINE__);    \
            std::printf(__VA_ARGS__);                             \
            std::printf("\n");                                    \
        }                                                         \
    } while (0)

static bool same_result(const PesResult& a, const PesResult& b) {
    return a.width == b.width && a.height == b.height && a.num_classes == b.num_classes &&
           std::memcmp(a.semantic_map, b.semantic_map, static_cast<size_t>(a.width) * a.height) == 0;
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    std::string image_path = argc > 2 ? argv[2] : "/home/llm/datasets/mvtec_anomaly_detection/capsule/test/good/000.png";

    PesModelParams mp;
    pes_default_model_params(&mp);
    std::snprintf(mp.model_dir, sizeof(mp.model_dir), "%s/weights/openvino_models_l6", root.c_str());
    if (!std::ifstream(std::string(mp.model_dir) + "/edgetam_decoder.xml")) {
        std::printf("test_api: モデルが見つからないのでスキップ (%s)\n", mp.model_dir);
        return 0;
    }
    PesPromptParams pp;
    pes_default_prompt_params(&pp);
    std::snprintf(pp.prompts_dir, sizeof(pp.prompts_dir), "%s/prompts/candidates/color_free_bg", root.c_str());
    PesRunParams rp;
    pes_default_run_params(&rp);

    // 画像: 無ければ合成（暗い背景に明るい円盤を 2 つ）
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        bgr = cv::Mat(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
        cv::circle(bgr, cv::Point(200, 240), 80, cv::Scalar(200, 220, 230), cv::FILLED);
        cv::circle(bgr, cv::Point(450, 250), 60, cv::Scalar(120, 160, 220), cv::FILLED);
    }

    // --- エラーの戻り値（モデル読み込み前） ---
    PesContext* ctx = nullptr;
    CHECK(pes_create(nullptr, &ctx) == PES_ERROR_INVALID_ARGUMENT && ctx == nullptr, "params=NULL");
    PesModelParams bad = mp;
    std::snprintf(bad.model_dir, sizeof(bad.model_dir), "/nonexistent/dir");
    CHECK(pes_create(&bad, &ctx) == PES_ERROR_LOAD && ctx == nullptr, "存在しないモデルのディレクトリ");
    CHECK(std::strlen(pes_last_error(nullptr)) > 0, "pes_last_error にメッセージ");
    bad = mp;
    bad.image_size = 500;  // パッチサイズ (16) の倍数でない
    CHECK(pes_create(&bad, &ctx) != PES_OK && ctx == nullptr, "image_size がパッチサイズの倍数でない");

    // --- 1. モデル読み込み ---
    CHECK(pes_create(&mp, &ctx) == PES_OK && ctx, "pes_create: %s", pes_last_error(nullptr));
    if (!ctx) return 1;

    PesImage image{bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step), PES_PIXEL_BGR};
    PesResult result;
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_ERROR_NO_PROMPTS, "プロンプトが無いとき");
    CHECK(std::strlen(pes_last_error(ctx)) > 0, "エラーメッセージ");

    // --- 2. テキスト特徴量抽出 ---
    CHECK(pes_extract_text_features(ctx, &pp) == PES_OK, "pes_extract_text_features: %s", pes_last_error(ctx));
    const int num_fg = pes_prompt_count(ctx, PES_PROMPT_FOREGROUND), num_bg = pes_prompt_count(ctx, PES_PROMPT_BACKGROUND);
    CHECK(num_fg == 29 && num_bg == 23, "プロンプト数 %d / %d", num_fg, num_bg);
    CHECK(std::string(pes_prompt_text(ctx, PES_PROMPT_FOREGROUND, 0)) == "a photo of a screw.", "先頭の前景プロンプト");
    CHECK(pes_prompt_text(ctx, PES_PROMPT_FOREGROUND, num_fg) == nullptr, "範囲外の番号は NULL");

    // 存在しない prompts_dir はエラーで、今のプロンプトはそのまま
    {
        PesPromptParams missing = pp;
        std::snprintf(missing.prompts_dir, sizeof(missing.prompts_dir), "/nonexistent/prompts");
        CHECK(pes_extract_text_features(ctx, &missing) == PES_ERROR_LOAD, "存在しない prompts_dir");
        CHECK(pes_prompt_count(ctx, PES_PROMPT_FOREGROUND) == num_fg && pes_prompt_count(ctx, PES_PROMPT_BACKGROUND) == num_bg, "失敗してもプロンプトは変わらない");
    }

    // --- 3. 処理実行（BGR / RGB / ファイルのどれでも同じ結果） ---
    PesResult base;
    CHECK(pes_run(ctx, &rp, &image, &base) == PES_OK, "pes_run: %s", pes_last_error(ctx));
    CHECK(base.width == bgr.cols && base.height == bgr.rows && base.semantic_map, "結果の大きさ");
    CHECK(base.num_classes >= 0 && base.timings.total_ms > 0, "結果");
    for (int i = 0; i < base.num_classes; ++i) CHECK(base.class_prompts[i] && std::strlen(base.class_prompts[i]) > 0, "class_prompts[%d]", i);
    for (size_t i = 0; i < static_cast<size_t>(base.width) * base.height; ++i)
        if (base.semantic_map[i] > base.num_classes) { CHECK(false, "semantic_map の値が num_classes を超える"); break; }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    PesImage image_rgb{rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), PES_PIXEL_RGB};
    CHECK(pes_run(ctx, &rp, &image_rgb, &result) == PES_OK && same_result(base, result), "RGB でも同じ結果");
    pes_result_free(&result);

    // stride が width*3 より大きい画像（ROI）
    {
        cv::Mat padded(bgr.rows, bgr.cols + 37, CV_8UC3, cv::Scalar(0, 0, 0));
        bgr.copyTo(padded(cv::Rect(0, 0, bgr.cols, bgr.rows)));
        PesImage roi{padded.data, bgr.cols, bgr.rows, static_cast<int>(padded.step), PES_PIXEL_BGR};
        CHECK(pes_run(ctx, &rp, &roi, &result) == PES_OK && same_result(base, result), "stride つきの画像でも同じ結果");
        pes_result_free(&result);
    }
    if (std::ifstream(image_path)) {
        CHECK(pes_run_file(ctx, &rp, image_path.c_str(), &result) == PES_OK && same_result(base, result), "pes_run_file でも同じ結果");
        pes_result_free(&result);
    }
    CHECK(pes_run_file(ctx, &rp, "/nonexistent.png", &result) == PES_ERROR_LOAD, "存在しない画像");

    // 同じ入力なら何度でも同じ結果（決定的）
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_OK && same_result(base, result), "2 回目も同じ結果");
    pes_result_free(&result);

    // --- プロンプトの追加・削除 ---
    const char* kSpoon = "a photo of a spoon.";
    CHECK(pes_add_prompt(ctx, PES_PROMPT_FOREGROUND, kSpoon) == PES_OK, "追加: %s", pes_last_error(ctx));
    CHECK(pes_prompt_count(ctx, PES_PROMPT_FOREGROUND) == num_fg + 1, "追加で 1 増える");
    CHECK(std::string(pes_prompt_text(ctx, PES_PROMPT_FOREGROUND, num_fg)) == kSpoon, "末尾に追加される");
    CHECK(pes_add_prompt(ctx, PES_PROMPT_FOREGROUND, kSpoon) == PES_ERROR_ALREADY_EXISTS, "重複の追加");
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_OK, "追加後の処理");
    pes_result_free(&result);
    CHECK(pes_remove_prompt(ctx, PES_PROMPT_FOREGROUND, kSpoon) == PES_OK, "削除");
    CHECK(pes_prompt_count(ctx, PES_PROMPT_FOREGROUND) == num_fg, "削除で元に戻る");
    CHECK(pes_remove_prompt(ctx, PES_PROMPT_FOREGROUND, kSpoon) == PES_ERROR_NOT_FOUND, "無いプロンプトの削除");
    CHECK(pes_remove_prompt_at(ctx, PES_PROMPT_BACKGROUND, 9999) == PES_ERROR_NOT_FOUND, "範囲外の番号の削除");
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_OK && same_result(base, result), "追加して削除したら元と同じ結果");
    pes_result_free(&result);

    const char* more[] = {"a photo of a fork.", "a photo of a spoon.", "a photo of a fork."};
    int added = -1;
    CHECK(pes_add_prompts(ctx, PES_PROMPT_BACKGROUND, more, 3, &added) == PES_OK && added == 2, "まとめて追加（重複は飛ばす）: %d", added);
    CHECK(pes_prompt_count(ctx, PES_PROMPT_BACKGROUND) == num_bg + 2, "背景が 2 増える");
    CHECK(pes_remove_prompt_at(ctx, PES_PROMPT_BACKGROUND, num_bg) == PES_OK && pes_remove_prompt_at(ctx, PES_PROMPT_BACKGROUND, num_bg) == PES_OK, "番号で削除");
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_OK && same_result(base, result), "背景も元に戻る");
    pes_result_free(&result);

    // 背景を空にすると処理できない。再抽出で復活する
    CHECK(pes_clear_prompts(ctx, PES_PROMPT_BACKGROUND) == PES_OK && pes_prompt_count(ctx, PES_PROMPT_BACKGROUND) == 0, "背景を空に");
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_ERROR_NO_PROMPTS, "背景が無いとき");
    CHECK(pes_extract_text_features(ctx, &pp) == PES_OK && pes_prompt_count(ctx, PES_PROMPT_BACKGROUND) == num_bg, "再抽出");
    CHECK(pes_run(ctx, &rp, &image, &result) == PES_OK && same_result(base, result), "再抽出後は元と同じ結果");
    pes_result_free(&result);

    // --- 実行パラメタを変える ---
    PesRunParams rp2 = rp;
    rp2.use_edgetam_clustering = 0;
    rp2.edgetam_refine_iterations = 1;
    rp2.semantic_max_side = 256;
    rp2.k_clusters = 5;
    CHECK(pes_run(ctx, &rp2, &image, &result) == PES_OK && result.width == bgr.cols && result.height == bgr.rows, "パラメタ違い: %s", pes_last_error(ctx));
    pes_result_free(&result);
    PesRunParams rp_noiter = rp;
    rp_noiter.edgetam_refine_iterations = 0;  // EdgeTAM を実行しない（粗いマスクがそのまま結果になる）
    CHECK(pes_run(ctx, &rp_noiter, &image, &result) == PES_OK && result.width == bgr.cols, "edgetam_refine_iterations=0: %s", pes_last_error(ctx));
    pes_result_free(&result);
    PesRunParams rp3 = rp;
    rp3.k_clusters = 0;
    CHECK(pes_run(ctx, &rp3, &image, &result) == PES_ERROR_INVALID_ARGUMENT, "k_clusters=0");
    CHECK(pes_run(ctx, nullptr, &image, &result) == PES_OK, "params=NULL は既定値");
    pes_result_free(&result);

    // --- 小さい・縦長の画像 ---
    for (cv::Size size : {cv::Size(33, 21), cv::Size(300, 900), cv::Size(1, 1)}) {
        cv::Mat small;
        cv::resize(bgr, small, size, 0, 0, cv::INTER_AREA);
        PesImage im{small.data, small.cols, small.rows, static_cast<int>(small.step), PES_PIXEL_BGR};
        const int status = pes_run(ctx, &rp, &im, &result);
        CHECK(status == PES_OK && result.width == size.width && result.height == size.height, "%dx%d の画像: %s", size.width, size.height, pes_last_error(ctx));
        pes_result_free(&result);
    }

    pes_result_free(&base);
    pes_destroy(ctx);
    std::printf(g_failed ? "test_api: %d 件失敗\n" : "test_api: すべて成功\n", g_failed);
    return g_failed ? 1 : 0;
}
