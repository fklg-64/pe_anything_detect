#include "coarse_segmentation.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <opencv2/imgproc.hpp>

#include "common.h"
#include "config.h"
#include "kmeans.h"
#include "mask_utils.h"
#include "params.h"
#include "simd.h"

namespace pes {

// =============================================================================
// プロンプトとテキスト特徴量
// =============================================================================

void CoarseSegmenter::load_prompts(const std::string& prompts_dir) {
    std::string dir = prompts_dir;
    if (!dir.empty() && dir.back() != '/') dir += '/';
    std::vector<std::string> fg = load_color_shape_prompts(dir + "foreground.yaml");
    std::vector<std::string> bg = load_plain_prompts(dir + "background.yaml");

    // 特徴量の抽出に成功してから置き換える（失敗したら今のプロンプトのまま）
    std::vector<float> fg_features = extractor_.extract_text_features(fg);
    std::vector<float> bg_features = extractor_.extract_text_features(bg);
    text_dim_ = extractor_.text_feature_dim();
    fg_prompts_.swap(fg);
    bg_prompts_.swap(bg);
    fg_features_.swap(fg_features);
    bg_features_.swap(bg_features);
    text_all_dirty_ = true;
}

int CoarseSegmenter::find_prompt(PesPromptKind kind, const std::string& text) const {
    const std::vector<std::string>& list = prompts(kind);
    for (size_t i = 0; i < list.size(); ++i)
        if (list[i] == text) return static_cast<int>(i);
    return -1;
}

int CoarseSegmenter::add_prompts(PesPromptKind kind, const std::vector<std::string>& texts) {
    std::vector<std::string>& list = kind == PES_PROMPT_FOREGROUND ? fg_prompts_ : bg_prompts_;
    std::vector<float>& features = kind == PES_PROMPT_FOREGROUND ? fg_features_ : bg_features_;

    std::vector<std::string> fresh;
    for (const std::string& t : texts)
        if (find_prompt(kind, t) < 0 && std::find(fresh.begin(), fresh.end(), t) == fresh.end()) fresh.push_back(t);
    if (fresh.empty()) return 0;

    const std::vector<float> new_features = extractor_.extract_text_features(fresh);  // 追加した分だけ特徴量を抽出する
    text_dim_ = extractor_.text_feature_dim();
    list.insert(list.end(), fresh.begin(), fresh.end());
    features.insert(features.end(), new_features.begin(), new_features.end());
    text_all_dirty_ = true;
    return static_cast<int>(fresh.size());
}

void CoarseSegmenter::remove_prompt_at(PesPromptKind kind, int index) {
    std::vector<std::string>& list = kind == PES_PROMPT_FOREGROUND ? fg_prompts_ : bg_prompts_;
    std::vector<float>& features = kind == PES_PROMPT_FOREGROUND ? fg_features_ : bg_features_;
    if (index < 0 || index >= static_cast<int>(list.size())) throw Error(PES_ERROR_NOT_FOUND, "プロンプトの番号が範囲外です: " + std::to_string(index));
    list.erase(list.begin() + index);
    features.erase(features.begin() + static_cast<size_t>(index) * text_dim_, features.begin() + static_cast<size_t>(index + 1) * text_dim_);
    text_all_dirty_ = true;
}

void CoarseSegmenter::clear_prompts(PesPromptKind kind) {
    (kind == PES_PROMPT_FOREGROUND ? fg_prompts_ : bg_prompts_).clear();
    (kind == PES_PROMPT_FOREGROUND ? fg_features_ : bg_features_).clear();
    text_all_dirty_ = true;
}

// =============================================================================
// 前景マップ（KMeans -> クラスタごとにテキストと照合）
// =============================================================================

std::pair<float, int> classify_cluster_feature(const float* cluster_feature, int dim, const float* text_all, int num_fg, int num_bg, double threshold) {
    // 正規化してテキストとの類似度(logits)を得る。Python 版は (100 * feature) @ テキスト特徴量.T
    std::vector<float> feature(cluster_feature, cluster_feature + dim);
    simd::normalize_rows(feature.data(), 1, dim, 1e-8f);
    for (float& v : feature) v *= params::CLIP_LOGIT_SCALE;
    std::vector<float> logits(num_fg + num_bg);
    simd::gemv_t_blas(feature.data(), text_all, num_fg + num_bg, dim, logits.data());

    // グループスコアは最も近いプロンプトの類似度（平均ではなく最大。プロンプト数を増減しても判定が偏らない）
    int best_fg = 0, best_bg = 0;
    float fg_score = logits[0], bg_score = logits[num_fg];
    for (int j = 1; j < num_fg; ++j)
        if (logits[j] > fg_score) { fg_score = logits[j]; best_fg = j; }
    for (int j = 1; j < num_bg; ++j)
        if (logits[num_fg + j] > bg_score) { bg_score = logits[num_fg + j]; best_bg = j; }

    // FG vs BG の2クラス softmax
    const float m = std::max(fg_score, bg_score);
    const float e_fg = std::exp(fg_score - m), e_bg = std::exp(bg_score - m);
    const float confidence = e_fg / (e_fg + e_bg);

    if (static_cast<double>(confidence) > threshold) return {confidence, best_fg};
    return {confidence, num_fg + best_bg};
}

std::vector<int> CoarseSegmenter::prompt_per_cluster(const std::vector<int>& cluster_labels, const std::vector<int>& prompt_index) {
    int max_label = 0;
    for (int l : cluster_labels) max_label = std::max(max_label, l);
    // 現れないラベルは、np.argmax が全部 False のとき 0 を返すのと同じく 0 番目のパッチのプロンプトにする
    std::vector<int> out(max_label + 1, prompt_index.empty() ? 0 : prompt_index[0]);
    std::vector<uint8_t> seen(max_label + 1, 0);
    for (size_t i = 0; i < cluster_labels.size(); ++i) {
        const int l = cluster_labels[i];
        if (!seen[l]) {
            seen[l] = 1;
            out[l] = prompt_index[i];
        }
    }
    return out;
}

void CoarseSegmenter::propagate_with_edgetam(std::vector<float>& confidence, std::vector<int>& prompt_index, const float* edge_embed,
                                             int channels, int num_patches, int num_foreground_prompts, double foreground_threshold) {
    // (C, N) -> (N, C) にして、L2 正規化する
    std::vector<float> features(static_cast<size_t>(num_patches) * channels);
    simd::transpose(edge_embed, channels, num_patches, features.data());
    simd::normalize_rows(features.data(), num_patches, channels, 1e-8f);
    const KnnGraph graph = knn_graph(features.data(), num_patches, channels, params::EDGE_PROPAGATION_NEIGHBORS);
    const std::vector<float> propagated = propagate(confidence, graph, params::EDGE_PROPAGATION_ALPHA, params::EDGE_PROPAGATION_ITERATIONS);

    const float threshold = static_cast<float>(foreground_threshold);
    std::vector<uint8_t> old_fg(num_patches), new_fg(num_patches);
    for (int i = 0; i < num_patches; ++i) {
        old_fg[i] = confidence[i] > threshold;
        new_fg[i] = propagated[i] > threshold;
    }

    // 判定が反転したパッチのプロンプトは、反転後と同じ側に判定されていた近傍のプロンプトの最頻値にする（無ければ画像全体でその側の最頻値）
    auto most_frequent = [](const std::vector<int>& values) {
        int max_value = 0;
        for (int v : values) max_value = std::max(max_value, v);
        std::vector<int> counts(max_value + 1, 0);
        for (int v : values) ++counts[v];
        return static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());  // 同数なら小さい番号
    };
    for (int patch = 0; patch < num_patches; ++patch) {
        if (old_fg[patch] == new_fg[patch]) continue;
        std::vector<int> candidates;
        for (int jj = graph.indptr[patch]; jj < graph.indptr[patch + 1]; ++jj) {
            const int neighbor = graph.indices[jj];
            if (old_fg[neighbor] == new_fg[patch]) candidates.push_back(prompt_index[neighbor]);
        }
        if (candidates.empty()) {
            for (int p = 0; p < num_patches; ++p)
                if (old_fg[p] == new_fg[patch]) candidates.push_back(prompt_index[p]);
        }
        if (!candidates.empty()) prompt_index[patch] = most_frequent(candidates);
        else prompt_index[patch] = new_fg[patch] ? 0 : num_foreground_prompts;
    }
    confidence = propagated;
}

CoarseForeground CoarseSegmenter::generate_foreground_map(const ClipExtractor::ImageFeatures& features, const float* edge_embed,
                                                          int edge_channels, const PesRunParams& run) {
    if (!ready()) throw Error(PES_ERROR_NO_PROMPTS, "前景と背景のプロンプトが必要です（pes_extract_text_features / pes_add_prompt）");
    const int n = feature_size_ * feature_size_;
    if (features.num_tokens != n + 1) throw Error(PES_ERROR_RUNTIME, "PE の特徴量のトークン数が想定と違います");
    const int num_fg = static_cast<int>(fg_prompts_.size()), num_bg = static_cast<int>(bg_prompts_.size());
    const int num_prompts = num_fg + num_bg;
    if (text_all_dirty_) {
        text_all_ = fg_features_;
        text_all_.insert(text_all_.end(), bg_features_.begin(), bg_features_.end());
        text_all_dirty_ = false;
    }

    CoarseForeground out;
    const float* patch_tokens = features.fused + features.fused_dim;  // 先頭の CLS トークンを除く

    std::vector<int> cluster_labels;
    {
        Timer t("    ┌ Kmeansクラスタ処理時間", verbose_, &out.kmeans_ms);
        cluster_labels = kmeans_fit_predict(patch_tokens, n, features.fused_dim, run.k_clusters, run.n_iter, params::KMEANS_SEED);
    }

    // クラスタごとのトークンをテキストと照合し、パッチごとの (前景確信度, プロンプト番号) にする
    std::vector<float> confidence(n);
    std::vector<int> prompt_index(n);
    {
        Timer t("    ┌ クラスタごとのCLSトークン抽出処理時間", verbose_, &out.cluster_tokens_ms);
        std::vector<float> labels_f(cluster_labels.begin(), cluster_labels.end());
        int num_clusters = 0, dim = 0;
        const float* tokens = extractor_.extract_cluster_tokens(features, labels_f.data(), n, &num_clusters, &dim);
        if (dim != text_dim_) throw Error(PES_ERROR_RUNTIME, "クラスタのトークン (" + std::to_string(dim) + " 次元) とテキスト特徴量 (" + std::to_string(text_dim_) + " 次元) の次元が違います");
        std::vector<float> cluster_confidence(num_clusters);
        std::vector<int> cluster_prompt(num_clusters);
        for (int c = 0; c < num_clusters; ++c) {
            const auto [conf, prompt] = classify_cluster_feature(tokens + static_cast<size_t>(c) * dim, dim, text_all_.data(), num_fg, num_bg, run.foreground_threshold);
            cluster_confidence[c] = conf;
            cluster_prompt[c] = prompt;
        }
        for (int i = 0; i < n; ++i) {
            confidence[i] = cluster_confidence[cluster_labels[i]];
            prompt_index[i] = cluster_prompt[cluster_labels[i]];
        }
    }

    if (edge_embed) {
        out.has_pe_clusters = true;
        out.pe_clusters = {cluster_labels, prompt_per_cluster(cluster_labels, prompt_index)};
        {
            Timer t("    ┌ EdgeTAM特徴での確信度の伝播処理時間", verbose_, &out.propagate_ms);
            propagate_with_edgetam(confidence, prompt_index, edge_embed, edge_channels, n, num_fg, run.foreground_threshold);
        }
        // PE のクラスタの中でも、伝播で判定が変わったパッチはプロンプトが違うので、(PE のクラスタ, プロンプト) の組を新しいクラスタにする
        std::vector<int64_t> keys(n);
        for (int i = 0; i < n; ++i) keys[i] = static_cast<int64_t>(cluster_labels[i]) * num_prompts + prompt_index[i];
        std::vector<int64_t> unique_keys = keys;
        std::sort(unique_keys.begin(), unique_keys.end());
        unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());
        for (int i = 0; i < n; ++i)
            cluster_labels[i] = static_cast<int>(std::lower_bound(unique_keys.begin(), unique_keys.end(), keys[i]) - unique_keys.begin());
    }

    out.clusters.index_list = prompt_per_cluster(cluster_labels, prompt_index);
    out.clusters.labels = std::move(cluster_labels);

    out.foreground_map = cv::Mat(feature_size_, feature_size_, CV_32F);
    out.foreground_mask = cv::Mat(feature_size_, feature_size_, CV_8U);
    const float threshold = static_cast<float>(run.foreground_threshold);
    for (int i = 0; i < n; ++i) {
        out.foreground_map.at<float>(i / feature_size_, i % feature_size_) = confidence[i];
        out.foreground_mask.at<uint8_t>(i / feature_size_, i % feature_size_) = confidence[i] > threshold ? 1 : 0;
    }
    return out;
}

// =============================================================================
// 粗いセマンティックラベル
// =============================================================================

void CoarseSegmenter::assign_labels(int feature_size, const std::vector<int>& cluster_labels, const std::vector<int>& cluster_index_list,
                                    std::vector<int>* labels_list, cv::Mat* semantic_labels) {
    const size_t n = cluster_labels.size();
    std::vector<int> semantic_indices(n);
    for (size_t i = 0; i < n; ++i) semantic_indices[i] = cluster_index_list[cluster_labels[i]];
    std::vector<int> sorted = semantic_indices;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    *labels_list = sorted;
    *semantic_labels = cv::Mat(feature_size, feature_size, CV_32S);
    for (size_t i = 0; i < n; ++i)
        semantic_labels->at<int>(static_cast<int>(i) / feature_size, static_cast<int>(i) % feature_size) =
            static_cast<int>(std::lower_bound(sorted.begin(), sorted.end(), semantic_indices[i]) - sorted.begin());
}

cv::Mat CoarseSegmenter::fill_holes(const cv::Mat& fg, cv::Mat& labels, const cv::Mat& original_fg) {
    cv::Mat comp;
    const int n = cv::connectedComponents(mask_not(fg), comp, 4, CV_32S);
    std::vector<uint8_t> touches_border(n, 0);
    for (int x = 0; x < comp.cols; ++x) {
        touches_border[comp.at<int>(0, x)] = 1;
        touches_border[comp.at<int>(comp.rows - 1, x)] = 1;
    }
    for (int y = 0; y < comp.rows; ++y) {
        touches_border[comp.at<int>(y, 0)] = 1;
        touches_border[comp.at<int>(y, comp.cols - 1)] = 1;
    }

    cv::Mat holes = cv::Mat::zeros(comp.size(), CV_8U);
    bool any_hole = false;
    for (int y = 0; y < comp.rows; ++y)
        for (int x = 0; x < comp.cols; ++x) {
            const int c = comp.at<int>(y, x);
            if (c > 0 && !touches_border[c]) {
                holes.at<uint8_t>(y, x) = 1;
                any_hole = true;
            }
        }
    cv::Mat result = fg.clone();
    if (any_hole && cv::countNonZero(original_fg) > 0) {
        // 穴の画素には、最も近い（元の）前景画素のラベルを与える
        const cv::Mat nearest = nearest_feature_index(original_fg);
        const cv::Mat old_labels = labels.clone();
        for (int y = 0; y < comp.rows; ++y)
            for (int x = 0; x < comp.cols; ++x) {
                if (!holes.at<uint8_t>(y, x)) continue;
                const int idx = nearest.at<int>(y, x);
                labels.at<int>(y, x) = old_labels.at<int>(idx / comp.cols, idx % comp.cols);
                result.at<uint8_t>(y, x) = 1;
            }
    }
    return result;
}

void CoarseSegmenter::merge_thin_regions(const cv::Mat& fg, cv::Mat& labels) {
    struct Region {
        int area;
        int class_id;
        cv::Mat mask;
    };
    std::set<int> class_ids;
    for (int y = 0; y < fg.rows; ++y)
        for (int x = 0; x < fg.cols; ++x)
            if (fg.at<uint8_t>(y, x)) class_ids.insert(labels.at<int>(y, x));

    std::vector<Region> regions;
    for (int class_id : class_ids) {
        cv::Mat of_class = cv::Mat::zeros(fg.size(), CV_8U);
        for (int y = 0; y < fg.rows; ++y)
            for (int x = 0; x < fg.cols; ++x) of_class.at<uint8_t>(y, x) = (fg.at<uint8_t>(y, x) && labels.at<int>(y, x) == class_id) ? 1 : 0;
        cv::Mat comp, stats, centroids;
        const int n = cv::connectedComponentsWithStats(of_class, comp, stats, centroids, 4, CV_32S);
        for (int i = 1; i < n; ++i) {
            cv::Mat mask;
            cv::compare(comp, i, mask, cv::CMP_EQ);
            regions.push_back({stats.at<int>(i, cv::CC_STAT_AREA), class_id, to_binary01(mask)});
        }
    }
    std::stable_sort(regions.begin(), regions.end(), [](const Region& a, const Region& b) { return a.area < b.area; });  // 小さい領域から順に

    for (Region& r : regions) {
        // 前の統合でラベルが変わっているかもしれないので、今もそのクラスの画素だけに絞る
        for (int y = 0; y < fg.rows; ++y)
            for (int x = 0; x < fg.cols; ++x)
                if (r.mask.at<uint8_t>(y, x) && labels.at<int>(y, x) != r.class_id) r.mask.at<uint8_t>(y, x) = 0;
        if (cv::countNonZero(r.mask) == 0) continue;
        // 1パッチ縮めて残るなら細くない
        cv::Mat eroded;
        cv::erode(r.mask.clone(), eroded, cv::Mat::ones(params::PATCH_NEIGHBORHOOD, params::PATCH_NEIGHBORHOOD, CV_8U));
        if (cv::countNonZero(eroded) > 0) continue;

        cv::Mat dilated;
        cv::dilate(r.mask.clone(), dilated, cv::Mat::ones(params::PATCH_NEIGHBORHOOD, params::PATCH_NEIGHBORHOOD, CV_8U));
        std::map<int, int> counts;
        for (int y = 0; y < fg.rows; ++y)
            for (int x = 0; x < fg.cols; ++x)
                if (dilated.at<uint8_t>(y, x) && !r.mask.at<uint8_t>(y, x) && fg.at<uint8_t>(y, x) && labels.at<int>(y, x) != r.class_id)
                    ++counts[labels.at<int>(y, x)];
        if (counts.empty()) continue;
        int best_label = counts.begin()->first, best_count = -1;
        for (const auto& [label, count] : counts)  // 同数なら小さいラベル
            if (count > best_count) {
                best_count = count;
                best_label = label;
            }
        for (int y = 0; y < fg.rows; ++y)
            for (int x = 0; x < fg.cols; ++x)
                if (r.mask.at<uint8_t>(y, x)) labels.at<int>(y, x) = best_label;
    }
}

void CoarseSegmenter::refine_coarse_segmentation(const cv::Mat& semantic_labels, const cv::Mat& foreground_mask, cv::Mat* out_mask,
                                                 cv::Mat* out_labels) {
    const cv::Mat original_fg = to_binary01(foreground_mask);
    cv::Mat labels = semantic_labels.clone();
    cv::Mat fg = fill_holes(original_fg, labels, original_fg);
    merge_thin_regions(fg, labels);
    *out_mask = fg;
    *out_labels = labels;
}

}  // namespace pes
