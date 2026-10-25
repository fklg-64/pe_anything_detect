// pe_segmentation_sample: ライブラリ (libpe_segmentation) を extern "C" の関数で呼ぶサンプル。
//
//   1. pes_create                  モデル読み込み
//   2. pes_extract_text_features   テキスト特徴量抽出（prompts_dir の yaml）。pes_add_prompt / pes_remove_prompt で追加・削除もできる
//   3. pes_run_file                処理実行（画像のパスを渡す）
//
// 画像のパス以外の引数は、既定では Python 版 (configs/ov_config.yaml) と同じ値を使う。
// 使い方の例（リポジトリのルートで実行）:
//   ./cppsrc/build/bin/pe_segmentation_sample image.png
//   ./cppsrc/build/bin/pe_segmentation_sample --add-fg "a photo of a spoon." --remove-bg "a photo of a shadow." image1.png image2.png
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pe_segmentation.h"

namespace fs = std::filesystem;

namespace {

// --- 可視化（core/common/visualization.py の ResultSaver 相当を OpenCV だけで） ---

// セマンティック可視化の色 (RGB)。0番は背景で常にシアン
const uint8_t kSemanticColors[13][3] = {
    {0, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {255, 128, 0},
    {128, 0, 255}, {255, 255, 255}, {128, 64, 0}, {255, 128, 192}, {0, 128, 0}, {128, 128, 128}};
constexpr double kForegroundOverlayAlpha = 0.3;  // 前景オーバーレイの赤の不透明度
constexpr int kPointRadius = 20;                 // プロンプト点の半径(px)

cv::Vec3b semantic_color_bgr(int class_id) {
    const uint8_t* c = class_id == 0 ? kSemanticColors[0] : kSemanticColors[1 + (class_id - 1) % 12];
    return cv::Vec3b(c[2], c[1], c[0]);
}

// "a photo of a " などの固有表現と末尾のピリオドを除いたクラス名にする
std::string clean_prompt(std::string prompt) {
    const char* heads[] = {"a photo of an ", "a photo of a ", "a photo of "};
    for (const char* h : heads) {
        if (prompt.rfind(h, 0) == 0) {
            prompt = prompt.substr(std::strlen(h));
            break;
        }
    }
    while (!prompt.empty() && (prompt.back() == '.' || prompt.back() == ' ')) prompt.pop_back();
    return prompt;
}

cv::Mat label_grid_to_color(const int* labels, int grid, const std::vector<int>& palette, cv::Size size) {
    cv::Mat color(grid, grid, CV_8UC3);
    for (int y = 0; y < grid; ++y)
        for (int x = 0; x < grid; ++x) {
            const int idx = (labels[y * grid + x] * 3) % std::max<int>(1, static_cast<int>(palette.size()) - 2);
            color.at<cv::Vec3b>(y, x) = cv::Vec3b(palette[idx + 2], palette[idx + 1], palette[idx]);
        }
    cv::Mat out;
    cv::resize(color, out, size, 0, 0, cv::INTER_NEAREST);
    return out;
}

void save_results(const fs::path& dir, const cv::Mat& image_bgr, const PesResult& r, const PesOutputParams& out, PesContext* ctx) {
    fs::create_directories(dir);
    const cv::Size size(r.width, r.height);
    const cv::Mat semantic(r.height, r.width, CV_8UC1, r.semantic_map);
    const std::vector<int> palette(out.color_palette, out.color_palette + out.color_palette_size);

    // 最終結果: クラス番号の画像・色分け・元画像への重畳・凡例つきの図
    cv::imwrite((dir / "semantic_map.png").string(), semantic);  // 画素値 = クラス番号
    cv::Mat color(r.height, r.width, CV_8UC3);
    for (int y = 0; y < r.height; ++y)
        for (int x = 0; x < r.width; ++x) color.at<cv::Vec3b>(y, x) = semantic_color_bgr(semantic.at<uint8_t>(y, x));
    cv::imwrite((dir / "semantic_color.png").string(), color);
    cv::Mat overlay;
    cv::addWeighted(image_bgr, 1.0 - out.overlay_alpha, color, out.overlay_alpha, 0, overlay);
    cv::imwrite((dir / "semantic_overlay.png").string(), overlay);

    std::vector<std::string> names{"background"};
    for (int i = 0; i < r.num_classes; ++i) names.push_back(clean_prompt(r.class_prompts[i]));
    std::vector<double> ratio(names.size(), 0.0);
    for (int y = 0; y < r.height; ++y)
        for (int x = 0; x < r.width; ++x) {
            const int id = semantic.at<uint8_t>(y, x);
            if (id < static_cast<int>(ratio.size())) ratio[id] += 1.0;
        }
    const int row_h = 26, legend_w = 380;
    cv::Mat figure(std::max(r.height, row_h * static_cast<int>(names.size()) + 12), r.width + legend_w, CV_8UC3, cv::Scalar(255, 255, 255));
    overlay.copyTo(figure(cv::Rect(0, 0, r.width, r.height)));
    for (size_t i = 0; i < names.size(); ++i) {
        const int y = 8 + static_cast<int>(i) * row_h;
        cv::rectangle(figure, cv::Rect(r.width + 10, y, 18, 18), cv::Scalar(semantic_color_bgr(static_cast<int>(i))), cv::FILLED);
        cv::rectangle(figure, cv::Rect(r.width + 10, y, 18, 18), cv::Scalar(0, 0, 0), 1);
        char text[256];
        std::snprintf(text, sizeof(text), "%zu: %s (%.1f%%)", i, names[i].c_str(), 100.0 * ratio[i] / (static_cast<double>(r.width) * r.height));
        cv::putText(figure, text, cv::Point(r.width + 36, y + 14), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    cv::imwrite((dir / "semantic_result.png").string(), figure);

    // クラスごとの最終マスクと、最終マスクの前景を赤で重畳した画像
    for (int id = 1; id <= r.num_classes; ++id) {
        cv::Mat mask;
        cv::compare(semantic, id, mask, cv::CMP_EQ);
        cv::imwrite((dir / ("foreground_mask_n" + std::to_string(id) + ".png")).string(), mask);
    }
    cv::Mat final_mask, red_final = cv::Mat::zeros(image_bgr.size(), CV_8UC3);
    cv::compare(semantic, 0, final_mask, cv::CMP_GT);
    red_final.setTo(cv::Scalar(0, 0, 255), final_mask);
    cv::Mat final_overlay;
    cv::addWeighted(image_bgr, 1.0 - kForegroundOverlayAlpha, red_final, kForegroundOverlayAlpha, 0, final_overlay);
    cv::imwrite((dir / "foreground_overlay_edgetam.png").string(), final_overlay);

    // 粗い推定の中間結果
    const int grid = r.feature_size;
    cv::imwrite((dir / "layer_fused_k.png").string(), label_grid_to_color(r.cluster_labels, grid, palette, size));
    cv::imwrite((dir / "coarse_semantic_segmentation.png").string(), label_grid_to_color(r.semantic_labels, grid, palette, size));
    cv::imwrite((dir / "coarse_semantic_segmentation_raw.png").string(), label_grid_to_color(r.raw_semantic_labels, grid, palette, size));

    cv::Mat fmap(grid, grid, CV_8UC1), fmask(grid, grid, CV_8UC1);
    for (int i = 0; i < grid * grid; ++i) {
        fmap.data[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, r.foreground_map[i] * 255.0f)));
        fmask.data[i] = r.foreground_mask[i] ? 255 : 0;
    }
    cv::Mat fmap_big, fmask_big;
    cv::resize(fmap, fmap_big, size, 0, 0, cv::INTER_LINEAR);
    cv::resize(fmask, fmask_big, size, 0, 0, cv::INTER_NEAREST);
    cv::imwrite((dir / "foreground_map.png").string(), fmap_big);
    cv::imwrite((dir / "foreground_mask.png").string(), fmask_big);

    cv::Mat red_coarse = cv::Mat::zeros(image_bgr.size(), CV_8UC3);
    red_coarse.setTo(cv::Scalar(0, 0, 255), fmask_big);
    cv::Mat coarse_overlay;
    cv::addWeighted(image_bgr, 1.0 - kForegroundOverlayAlpha, red_coarse, kForegroundOverlayAlpha, 0, coarse_overlay);
    if (cv::countNonZero(fmask_big) > 0) {
        for (int i = 0; i < r.num_points; ++i)
            cv::circle(coarse_overlay, cv::Point(static_cast<int>(r.point_coords[i * 2]), static_cast<int>(r.point_coords[i * 2 + 1])), kPointRadius,
                       r.point_labels[i] == 1 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), cv::FILLED);
    }
    cv::imwrite((dir / "foreground_overlay.png").string(), coarse_overlay);

    // クラスタごとの分類結果（対応するプロンプトの文）
    FILE* fp = std::fopen((dir / "cluster_prompts.txt").string().c_str(), "w");
    if (fp) {
        std::vector<int> counts(r.num_clusters, 0);
        for (int i = 0; i < grid * grid; ++i) ++counts[r.cluster_labels[i]];
        const int num_fg = pes_prompt_count(ctx, PES_PROMPT_FOREGROUND);
        for (int c = 0; c < r.num_clusters; ++c) {
            const int p = r.cluster_prompt_index[c];
            const char* text = p < num_fg ? pes_prompt_text(ctx, PES_PROMPT_FOREGROUND, p) : pes_prompt_text(ctx, PES_PROMPT_BACKGROUND, p - num_fg);
            std::fprintf(fp, "クラスタ%d (%dパッチ): %s\n", c, counts[c], text ? text : "?");
        }
        std::fclose(fp);
    }
}

// 検証用: 配列を .npy で書く（Python の np.load で読める）
template <typename T>
void write_npy(const fs::path& path, const char* descr, const std::vector<size_t>& shape, const T* data) {
    std::string shape_text = "(";
    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_text += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
        count *= shape[i];
        if (i + 1 < shape.size()) shape_text += " ";
    }
    shape_text += ")";
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape_text + ", }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    FILE* fp = std::fopen(path.string().c_str(), "wb");
    if (!fp) return;
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::fwrite(magic, 1, 8, fp);
    const uint16_t len = static_cast<uint16_t>(header.size());
    std::fwrite(&len, 2, 1, fp);
    std::fwrite(header.data(), 1, header.size(), fp);
    std::fwrite(data, sizeof(T), count, fp);
    std::fclose(fp);
}

void dump_npy(const fs::path& dir, const PesResult& r) {
    fs::create_directories(dir);
    const size_t grid = static_cast<size_t>(r.feature_size);
    write_npy(dir / "semantic_map.npy", "|u1", {static_cast<size_t>(r.height), static_cast<size_t>(r.width)}, r.semantic_map);
    write_npy(dir / "prompt_indices.npy", "<i4", {static_cast<size_t>(r.num_classes)}, r.prompt_indices);
    write_npy(dir / "foreground_map.npy", "<f4", {grid, grid}, r.foreground_map);
    write_npy(dir / "foreground_mask.npy", "|u1", {grid, grid}, r.foreground_mask);
    write_npy(dir / "cluster_labels.npy", "<i4", {grid * grid}, r.cluster_labels);
    write_npy(dir / "cluster_index_list.npy", "<i4", {static_cast<size_t>(r.num_clusters)}, r.cluster_prompt_index);
    write_npy(dir / "raw_semantic_labels.npy", "<i4", {grid, grid}, r.raw_semantic_labels);
    write_npy(dir / "semantic_labels.npy", "<i4", {grid, grid}, r.semantic_labels);
    write_npy(dir / "point_coords.npy", "<f4", {static_cast<size_t>(r.num_points), 2}, r.point_coords);
    write_npy(dir / "point_labels.npy", "<i4", {static_cast<size_t>(r.num_points)}, r.point_labels);
}

void print_timings(const PesTimings& t) {
    std::printf("---- 処理時間 [ms] ----\n");
    std::printf("  画像前処理                     %8.2f\n", t.preprocess_ms);
    std::printf("  EdgeTAM エンコーダ             %8.2f\n", t.edgetam_encode_ms);
    std::printf("  PE 画像特徴量                  %8.2f\n", t.clip_features_ms);
    std::printf("  粗い前景マスク取得             %8.2f  (KMeans %.2f / クラスタトークン %.2f / 伝播 %.2f)\n", t.coarse_foreground_ms, t.kmeans_ms,
                t.cluster_tokens_ms, t.propagate_ms);
    std::printf("  セマンティックラベル           %8.2f\n", t.semantic_labels_ms);
    std::printf("  粗い結果の整形                 %8.2f\n", t.coarse_refine_ms);
    std::printf("  EdgeTAM による詳細化           %8.2f\n", t.refine_ms);
    std::printf("  合計                           %8.2f\n", t.total_ms);
}

void usage(const char* argv0) {
    std::printf(
        "使い方: %s [オプション] <画像のパス> [<画像のパス> ...]\n"
        "\n"
        "画像のパス以外は、既定では Python 版 (configs/ov_config.yaml) と同じ値を使います（リポジトリのルートで実行してください）。\n"
        "\n"
        "オプション:\n"
        "  --config <yaml>        設定 yaml を読む（ov_config.yaml と同じ形式。指定したキーだけ上書き）\n"
        "  --model-dir <dir>      OpenVINO モデルのディレクトリ\n"
        "  --prompts-dir <dir>    foreground.yaml / background.yaml のディレクトリ\n"
        "  --output-dir <dir>     結果の出力先\n"
        "  --add-fg <文>          前景プロンプトを追加（繰り返し可）\n"
        "  --add-bg <文>          背景プロンプトを追加（繰り返し可）\n"
        "  --remove-fg <文>       前景プロンプトを削除（繰り返し可）\n"
        "  --remove-bg <文>       背景プロンプトを削除（繰り返し可）\n"
        "  --repeat <N>           各画像を N 回処理する（時間の測定用。保存は最後の 1 回）\n"
        "  --no-save              結果の画像を保存しない\n"
        "  --dump-npy <dir>       検証用: 結果の配列を <dir>/<番号>/ に .npy で書く（Python 版との比較用。<dir>/00, 01, ... は画像の引数の順番）\n"
        "  -v, --verbose          各段階の処理時間をその都度表示\n"
        "  -h, --help             このヘルプ\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    // 既定値（Python 版の configs/ov_config.yaml と同じ）
    PesModelParams model_params;
    PesPromptParams prompt_params;
    PesRunParams run_params;
    PesOutputParams output_params;
    pes_default_model_params(&model_params);
    pes_default_prompt_params(&prompt_params);
    pes_default_run_params(&run_params);
    pes_default_output_params(&output_params);

    std::vector<std::string> images, add_fg, add_bg, remove_fg, remove_bg;
    std::string config_path;
    std::string model_dir, prompts_dir, output_dir, dump_dir;
    int repeat = 1;
    bool save = true, verbose = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s には値が必要です\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--config") config_path = value("--config");
        else if (a == "--model-dir") model_dir = value("--model-dir");
        else if (a == "--prompts-dir") prompts_dir = value("--prompts-dir");
        else if (a == "--output-dir") output_dir = value("--output-dir");
        else if (a == "--add-fg") add_fg.push_back(value("--add-fg"));
        else if (a == "--add-bg") add_bg.push_back(value("--add-bg"));
        else if (a == "--remove-fg") remove_fg.push_back(value("--remove-fg"));
        else if (a == "--remove-bg") remove_bg.push_back(value("--remove-bg"));
        else if (a == "--repeat") repeat = std::max(1, std::atoi(value("--repeat").c_str()));
        else if (a == "--no-save") save = false;
        else if (a == "--dump-npy") dump_dir = value("--dump-npy");
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "不明なオプション: %s\n", a.c_str()); usage(argv[0]); return 2; }
        else images.push_back(a);
    }
    if (images.empty()) { usage(argv[0]); return 2; }

    if (!config_path.empty() &&
        pes_load_config_yaml(config_path.c_str(), &model_params, &prompt_params, &run_params, &output_params) != PES_OK) {
        std::fprintf(stderr, "設定の読み込みに失敗: %s\n", pes_last_error(nullptr));
        return 1;
    }
    auto set_string = [](char* dst, size_t cap, const std::string& s) { std::snprintf(dst, cap, "%s", s.c_str()); };
    if (!model_dir.empty()) set_string(model_params.model_dir, sizeof(model_params.model_dir), model_dir);
    if (!prompts_dir.empty()) set_string(prompt_params.prompts_dir, sizeof(prompt_params.prompts_dir), prompts_dir);
    if (!output_dir.empty()) set_string(output_params.output_dir, sizeof(output_params.output_dir), output_dir);

    // 既定の相対パス（./weights/... など）が現在のディレクトリに無ければ、リポジトリのルート
    // （実行ファイルが cppsrc/build/bin/ にあるとして 3 つ上）からの相対パスとして探す
    {
        std::error_code ec;
        const fs::path root = fs::read_symlink("/proc/self/exe", ec).parent_path().parent_path().parent_path().parent_path();
        auto resolve = [&](char* path, size_t capacity) {
            if (ec || path[0] == '\0' || fs::path(path).is_absolute() || fs::exists(path)) return;
            const fs::path candidate = root / path;
            if (fs::exists(candidate)) std::snprintf(path, capacity, "%s", candidate.lexically_normal().string().c_str());
        };
        resolve(model_params.model_dir, sizeof(model_params.model_dir));
        resolve(prompt_params.prompts_dir, sizeof(prompt_params.prompts_dir));
    }

    std::printf("AVX2: %s\n", pes_cpu_has_avx2() ? "使用" : "非対応（スカラー実装）");

    // 1. モデル読み込み
    PesContext* ctx = nullptr;
    if (pes_create(&model_params, &ctx) != PES_OK) {
        std::fprintf(stderr, "モデルの読み込みに失敗: %s\n", pes_last_error(nullptr));
        return 1;
    }
    pes_set_verbose(ctx, verbose ? 1 : 0);

    // 2. テキスト特徴量抽出
    if (pes_extract_text_features(ctx, &prompt_params) != PES_OK) {
        std::fprintf(stderr, "テキスト特徴量の抽出に失敗: %s\n", pes_last_error(ctx));
        pes_destroy(ctx);
        return 1;
    }
    std::printf("プロンプト: 前景 %d 個 / 背景 %d 個\n", pes_prompt_count(ctx, PES_PROMPT_FOREGROUND), pes_prompt_count(ctx, PES_PROMPT_BACKGROUND));

    // プロンプトの追加・削除（そのたびに特徴量を更新する）
    auto edit = [&](const std::vector<std::string>& texts, PesPromptKind kind, bool add) {
        for (const std::string& t : texts) {
            const int status = add ? pes_add_prompt(ctx, kind, t.c_str()) : pes_remove_prompt(ctx, kind, t.c_str());
            if (status != PES_OK) std::fprintf(stderr, "  %s失敗 (%s): %s\n", add ? "追加" : "削除", t.c_str(), pes_last_error(ctx));
            else std::printf("  %s: [%s] %s\n", add ? "追加" : "削除", kind == PES_PROMPT_FOREGROUND ? "前景" : "背景", t.c_str());
        }
    };
    edit(add_fg, PES_PROMPT_FOREGROUND, true);
    edit(add_bg, PES_PROMPT_BACKGROUND, true);
    edit(remove_fg, PES_PROMPT_FOREGROUND, false);
    edit(remove_bg, PES_PROMPT_BACKGROUND, false);

    // 3. 処理実行
    int exit_code = 0;
    int image_index = 0;
    for (const std::string& path : images) {
        const int index = image_index++;
        if (!fs::exists(path)) {
            std::printf("====== 存在しません: %s ======\n", fs::path(path).filename().string().c_str());
            exit_code = 1;
            continue;
        }
        std::printf("====== 画像パス: %s ======\n", fs::path(path).filename().string().c_str());
        PesResult result;
        bool ok = true;
        PesTimings best{};  // repeat のとき、合計が最小だった回の各段階の時間（他のプロセスの影響を受けにくい）
        std::vector<double> totals;
        for (int n = 0; n < repeat && ok; ++n) {
            if (n > 0) pes_result_free(&result);
            ok = pes_run_file(ctx, &run_params, path.c_str(), &result) == PES_OK;
            if (!ok) {
                std::fprintf(stderr, "処理に失敗: %s\n", pes_last_error(ctx));
                exit_code = 1;
            } else {
                totals.push_back(result.timings.total_ms);
                if (n == 0 || result.timings.total_ms < best.total_ms) best = result.timings;
                if (repeat > 1) std::printf("  [%d/%d] %.2f ms\n", n + 1, repeat, result.timings.total_ms);
            }
        }
        if (!ok) continue;
        if (repeat > 1) {
            std::sort(totals.begin(), totals.end());
            std::printf("  合計 %d 回: 最小 %.2f ms / 中央値 %.2f ms（以下は最小だった回の内訳）\n", repeat, totals.front(), totals[totals.size() / 2]);
            result.timings = best;
        }

        print_timings(result.timings);
        std::printf("前景クラス: %d 個\n", result.num_classes);
        for (int i = 0; i < result.num_classes; ++i) std::printf("  クラス%d: %s\n", i + 1, result.class_prompts[i]);

        if (save) {
            const cv::Mat image_bgr = cv::imread(path, cv::IMREAD_COLOR);
            const fs::path save_dir = fs::path(output_params.output_dir) / fs::path(path).stem();
            save_results(save_dir, image_bgr, result, output_params, ctx);
            std::printf("Saving results to %s\n", save_dir.string().c_str());
        }
        if (!dump_dir.empty()) {
            char name[16];
            std::snprintf(name, sizeof(name), "%02d", index);  // 同じ名前の画像（別フォルダの 000.png など）が衝突しないよう、引数の順番で区別する
            dump_npy(fs::path(dump_dir) / name, result);
        }
        pes_result_free(&result);
    }

    pes_destroy(ctx);
    return exit_code;
}
