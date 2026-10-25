#include "segmenter.h"

#include <cstring>
#include <fstream>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "mask_utils.h"
#include "params.h"
#include "simd.h"

// CMake が data/clip_bpe_merges.txt から生成する埋め込みデータ（merges_data.cpp）
extern "C" const unsigned char pes_clip_bpe_merges[];
extern "C" const unsigned long pes_clip_bpe_merges_size;

namespace pes {

namespace {

std::unique_ptr<ClipTokenizer> make_tokenizer(const PesModelParams& params) {
    if (params.tokenizer_merges_path[0] != '\0') {
        std::ifstream in(params.tokenizer_merges_path, std::ios::binary);
        if (!in) throw Error(PES_ERROR_LOAD, std::string("BPE の merges ファイルを開けません: ") + params.tokenizer_merges_path);
        std::stringstream ss;
        ss << in.rdbuf();
        return std::make_unique<ClipTokenizer>(ss.str());
    }
    return std::make_unique<ClipTokenizer>(std::string(reinterpret_cast<const char*>(pes_clip_bpe_merges), pes_clip_bpe_merges_size));
}

void validate_model_params(const PesModelParams& p) {
    if (p.image_size <= 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "image_size は 1 以上にしてください");
    if (p.ov_num_thread < 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "ov_num_thread は 0 以上にしてください");
    if (p.edgetam_high_res_feature_downscale_factor <= 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "edgetam_high_res_feature_downscale_factor は 1 以上にしてください");
    if (p.model_dir[0] == '\0') throw Error(PES_ERROR_INVALID_ARGUMENT, "model_dir を指定してください");
}

void validate_run_params(const PesRunParams& p) {
    if (p.k_clusters <= 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "k_clusters は 1 以上にしてください");
    if (p.n_iter < 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "n_iter は 0 以上にしてください");
    if (p.edgetam_foreground_points_per_class < 0 || p.edgetam_background_point_count < 0)
        throw Error(PES_ERROR_INVALID_ARGUMENT, "EdgeTAM に入れる点の数は 0 以上にしてください");
    if (p.edgetam_refine_iterations < 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "edgetam_refine_iterations は 0 以上にしてください");
    if (p.semantic_max_side <= 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "semantic_max_side は 1 以上にしてください");
}

template <typename T>
T* alloc_array(size_t count) {
    T* p = static_cast<T*>(std::calloc(count > 0 ? count : 1, sizeof(T)));
    if (!p) throw std::bad_alloc();
    return p;
}

}  // namespace

// 引数の検査と、OpenCV のスレッド数の設定（メンバの初期化より前に行う）
static const PesModelParams& prepare(const PesModelParams& params) {
    validate_model_params(params);
    cv::setNumThreads(params::OPENCV_NUM_THREADS);
    return params;
}

Segmenter::Segmenter(const PesModelParams& params)
    : params_(prepare(params)),
      core_(),
      clip_(core_, params_, make_tokenizer(params_)),
      edgetam_(core_, params_),
      coarse_(clip_),
      refiner_(edgetam_, clip_.feature_size()) {
    const size_t count = static_cast<size_t>(3) * params_.image_size * params_.image_size;
    clip_tensor_ = make_aligned<float>(count);
    edgetam_tensor_ = make_aligned<float>(count);
}

void Segmenter::set_verbose(bool verbose) {
    verbose_ = verbose;
    coarse_.set_verbose(verbose);
    refiner_.set_verbose(verbose);
}

void Segmenter::segment(const PesImage& image, const PesRunParams& run, PesResult* result) {
    validate_run_params(run);
    if (!image.data || image.width <= 0 || image.height <= 0) throw Error(PES_ERROR_INVALID_ARGUMENT, "画像が空です");
    if (image.pixel_format != PES_PIXEL_RGB && image.pixel_format != PES_PIXEL_BGR) throw Error(PES_ERROR_INVALID_ARGUMENT, "pixel_format が不正です");
    const int stride = image.stride > 0 ? image.stride : image.width * 3;
    if (stride < image.width * 3) throw Error(PES_ERROR_INVALID_ARGUMENT, "stride が小さすぎます");
    if (!coarse_.ready()) throw Error(PES_ERROR_NO_PROMPTS, "前景と背景のプロンプトがありません（pes_extract_text_features / pes_add_prompt を先に呼んでください）");

    const int width = image.width, height = image.height;
    const int size = params_.image_size;
    const bool bgr = image.pixel_format == PES_PIXEL_BGR;
    PesTimings timings{};
    const Timer total("合計", false);

    // 画像前処理: リサイズは CLIP と EdgeTAM で共通（正規化だけがモデルごとに異なる）
    cv::Mat resized;
    {
        Timer t("画像前処理時間", verbose_, &timings.preprocess_ms);
        const cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(image.data), static_cast<size_t>(stride));
        cv::resize(src, resized, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);
        const simd::NormalizeTarget targets[2] = {
            {{ClipExtractor::kMean[0], ClipExtractor::kMean[1], ClipExtractor::kMean[2]}, {ClipExtractor::kStd[0], ClipExtractor::kStd[1], ClipExtractor::kStd[2]}, clip_tensor_.get()},
            {{EdgeTamRunner::kMean[0], EdgeTamRunner::kMean[1], EdgeTamRunner::kMean[2]}, {EdgeTamRunner::kStd[0], EdgeTamRunner::kStd[1], EdgeTamRunner::kStd[2]}, edgetam_tensor_.get()},
        };
        simd::normalize_to_chw(resized.data, static_cast<size_t>(size) * size, bgr, targets, 2);
    }

    // EdgeTAM の特徴量抽出（粗い判定で EdgeTAM の特徴を使うときは、ここで 1 回だけ実行して精緻化にも渡す）
    EdgeTamRunner::Features edgetam_features;
    bool have_edgetam_features = false;
    if (run.use_edgetam_clustering) {
        Timer t("EdgeTAM特徴量抽出処理時間", verbose_, &timings.edgetam_encode_ms);
        edgetam_features = edgetam_.encode(edgetam_tensor_.get());
        have_edgetam_features = true;
    }

    // 特徴量抽出
    ClipExtractor::ImageFeatures clip_features;
    {
        Timer t("特徴量抽出処理時間", verbose_, &timings.clip_features_ms);
        clip_features = clip_.extract_image_features(clip_tensor_.get());
    }

    // CLIP + KMeans による初期前景マップの推定
    CoarseForeground coarse;
    {
        Timer t("粗い前景マスク取得処理時間", verbose_, &timings.coarse_foreground_ms);
        const float* edge_embed = nullptr;
        int edge_channels = 0;
        if (have_edgetam_features) {
            const ov::Shape shape = edgetam_features.image_embed.get_shape();  // (1, C, h, w)
            if (shape.size() != 4 || static_cast<int>(shape[2] * shape[3]) != clip_.feature_size() * clip_.feature_size())
                throw Error(PES_ERROR_RUNTIME, "EdgeTAM の特徴マップ " + shape_to_string(shape) + " の大きさが PE のパッチ格子と合いません");
            edge_embed = edgetam_features.image_embed.data<float>();
            edge_channels = static_cast<int>(shape[1]);
        }
        coarse = coarse_.generate_foreground_map(clip_features, edge_embed, edge_channels, run);
        timings.kmeans_ms = coarse.kmeans_ms;
        timings.cluster_tokens_ms = coarse.cluster_tokens_ms;
        timings.propagate_ms = coarse.propagate_ms;
    }

    // セマンティックセグメンテーションのラベルを取得
    std::vector<int> labels_list;
    cv::Mat raw_semantic_labels;
    {
        Timer t("セマンティックセグメンテーション処理時間", verbose_, &timings.semantic_labels_ms);
        coarse_.assign_semantic_labels(coarse.clusters.labels, coarse.clusters.index_list, &labels_list, &raw_semantic_labels);
    }

    // パッチ単位の粗い結果を整形（穴埋め・小さな島の統合・境界パッチのラベル統一）
    cv::Mat foreground_mask, semantic_labels;
    {
        Timer t("粗いセグメンテーション整形処理時間", verbose_, &timings.coarse_refine_ms);
        CoarseSegmenter::refine_coarse_segmentation(raw_semantic_labels, coarse.foreground_mask, &foreground_mask, &semantic_labels);
    }

    // 粗いセマンティックセグメンテーションを EdgeTAM で精緻化し、1枚のセマンティックマップに統合
    RefineOutput refined;
    {
        Timer t("詳細セマンティックセグメンテーション処理時間", verbose_, &timings.refine_ms);
        if (!have_edgetam_features) {
            Timer te("    ┌ edgetam特徴量抽出", verbose_);
            edgetam_features = edgetam_.encode(edgetam_tensor_.get());
        }
        refined = refiner_.refine(resized, coarse.foreground_map, foreground_mask, semantic_labels, labels_list, width, height, edgetam_features, run);
    }

    // --- 結果の書き出し（C の配列） ---
    std::memset(result, 0, sizeof(*result));
    try {
        const int f = clip_.feature_size();
        const size_t patches = static_cast<size_t>(f) * f;
        result->width = width;
        result->height = height;
        result->semantic_map = alloc_array<uint8_t>(static_cast<size_t>(width) * height);
        for (int y = 0; y < height; ++y) std::memcpy(result->semantic_map + static_cast<size_t>(y) * width, refined.semantic_map.ptr<uint8_t>(y), width);

        result->num_classes = static_cast<int>(refined.prompt_indices.size());
        result->prompt_indices = alloc_array<int>(refined.prompt_indices.size());
        result->class_prompts = alloc_array<char*>(refined.prompt_indices.size());
        const auto& fg = coarse_.prompts(PES_PROMPT_FOREGROUND);
        const auto& bg = coarse_.prompts(PES_PROMPT_BACKGROUND);
        for (size_t i = 0; i < refined.prompt_indices.size(); ++i) {
            const int index = refined.prompt_indices[i];
            result->prompt_indices[i] = index;
            const std::string& text = index < static_cast<int>(fg.size()) ? fg[index] : bg[index - fg.size()];
            result->class_prompts[i] = alloc_array<char>(text.size() + 1);
            std::memcpy(result->class_prompts[i], text.c_str(), text.size() + 1);
        }

        result->num_points = static_cast<int>(refined.point_labels.size());
        result->point_coords = alloc_array<float>(refined.point_coords.size() * 2);
        result->point_labels = alloc_array<int>(refined.point_labels.size());
        for (size_t i = 0; i < refined.point_coords.size(); ++i) {
            result->point_coords[i * 2] = refined.point_coords[i][0];
            result->point_coords[i * 2 + 1] = refined.point_coords[i][1];
            result->point_labels[i] = refined.point_labels[i];
        }

        result->feature_size = f;
        result->foreground_map = alloc_array<float>(patches);
        result->foreground_mask = alloc_array<uint8_t>(patches);
        result->cluster_labels = alloc_array<int>(patches);
        result->raw_semantic_labels = alloc_array<int>(patches);
        result->semantic_labels = alloc_array<int>(patches);
        for (int y = 0; y < f; ++y)
            for (int x = 0; x < f; ++x) {
                const size_t i = static_cast<size_t>(y) * f + x;
                result->foreground_map[i] = coarse.foreground_map.at<float>(y, x);
                result->foreground_mask[i] = foreground_mask.at<uint8_t>(y, x);
                result->cluster_labels[i] = coarse.clusters.labels[i];
                result->raw_semantic_labels[i] = raw_semantic_labels.at<int>(y, x);
                result->semantic_labels[i] = semantic_labels.at<int>(y, x);
            }
        result->num_clusters = static_cast<int>(coarse.clusters.index_list.size());
        result->cluster_prompt_index = alloc_array<int>(coarse.clusters.index_list.size());
        for (size_t i = 0; i < coarse.clusters.index_list.size(); ++i) result->cluster_prompt_index[i] = coarse.clusters.index_list[i];
    } catch (...) {
        pes_result_free(result);
        throw;
    }

    timings.total_ms = total.elapsed_ms();
    result->timings = timings;
    if (verbose_) std::printf("Inference Time: %.2f ms\n", timings.total_ms);
}

}  // namespace pes
