#include "semantic_refiner.h"

#include <algorithm>
#include <cmath>
#include <set>

#include <opencv2/imgproc.hpp>

#include "common.h"
#include "mask_utils.h"
#include "params.h"
#include "simd.h"

namespace pes {

SemanticRefiner::SemanticRefiner(EdgeTamRunner& edgetam, int feature_size)
    : edgetam_(edgetam), feature_size_(feature_size), high_res_feature_size_(edgetam.mask_size()) {
    guide_radius_ = std::max(1, high_res_feature_size_ / feature_size_);
}

std::vector<std::array<double, 2>> SemanticRefiner::transform_coords(const std::vector<cv::Point>& pts, int width, int height) const {
    std::vector<std::array<double, 2>> out;
    out.reserve(pts.size());
    for (const cv::Point& p : pts)
        out.push_back({(p.x + 0.5) * width / feature_size_, (p.y + 0.5) * height / feature_size_});
    return out;
}

cv::Mat SemanticRefiner::make_guide(const cv::Mat& resized) const {
    cv::Mat small;
    cv::resize(resized, small, cv::Size(high_res_feature_size_, high_res_feature_size_), 0, 0, cv::INTER_AREA);
    cv::Mat guide(small.rows, small.cols, CV_32F);
    for (int y = 0; y < small.rows; ++y) {
        const cv::Vec3b* s = small.ptr<cv::Vec3b>(y);
        float* g = guide.ptr<float>(y);
        for (int x = 0; x < small.cols; ++x) {
            const float sum = static_cast<float>(s[x][0]) + static_cast<float>(s[x][1]) + static_cast<float>(s[x][2]);
            g[x] = (sum / 3.0f) / 255.0f;
        }
    }
    return guide;
}

RefineOutput SemanticRefiner::refine(const cv::Mat& resized, const cv::Mat& foreground_map, const cv::Mat& foreground_mask,
                                     const cv::Mat& semantic_labels, const std::vector<int>& labels_list, int width, int height,
                                     const EdgeTamRunner::Features& feats, const PesRunParams& run) {
    const cv::Mat foreground_binary = to_binary01(foreground_mask);
    std::set<int> id_set;
    for (int y = 0; y < foreground_binary.rows; ++y)
        for (int x = 0; x < foreground_binary.cols; ++x)
            if (foreground_binary.at<uint8_t>(y, x)) id_set.insert(semantic_labels.at<int>(y, x));
    const std::vector<int> semantic_ids(id_set.begin(), id_set.end());

    RefineOutput out;
    for (int id : semantic_ids) out.prompt_indices.push_back(labels_list[id]);
    if (semantic_ids.empty()) {
        out.semantic_map = cv::Mat::zeros(height, width, CV_8U);
        return out;
    }

    // マスク演算は長辺が semantic_max_side 以下の作業解像度で行い、最後に元画像サイズへ戻す
    const double scale = std::min(1.0, static_cast<double>(run.semantic_max_side) / std::max(width, height));
    const int work_w = std::max(1, static_cast<int>(std::nearbyint(width * scale)));
    const int work_h = std::max(1, static_cast<int>(std::nearbyint(height * scale)));

    Context ctx;
    ctx.feats = &feats;
    // 背景の負例点は全クラス共通
    ctx.background_coords = transform_coords(select_background_points(foreground_binary, run.edgetam_background_point_count, foreground_map), width, height);
    ctx.foreground_map = foreground_map;
    ctx.width = width;
    ctx.height = height;
    ctx.size = cv::Size(work_w, work_h);
    // 粗いマスクの信頼限界は1パッチ。1パッチ未満の領域はノイズとして扱う
    ctx.min_area = std::max(1, static_cast<int>(params::MIN_REGION_PATCH_AREA * work_w * work_h / (static_cast<double>(feature_size_) * feature_size_)));
    ctx.guide = make_guide(resized);
    ctx.run = &run;
    ctx.semantic_map = cv::Mat::zeros(work_h, work_w, CV_8U);
    ctx.best_score = cv::Mat(work_h, work_w, CV_32F, cv::Scalar(-1.0f));

    const auto class_components = refine_classes(ctx, semantic_ids, semantic_labels, foreground_binary);
    rescue_uncovered(ctx, class_components);

    // 重ね合わせ後のノイズ除去
    {
        Timer t("    ┌ 小領域のノイズ除去", verbose_);
        remove_small_regions(ctx.semantic_map, ctx.min_area);
    }
    // クラス境界の割れ（前景の内部の細い隙間）を埋める
    {
        Timer t("    ┌ 隙間埋め", verbose_);
        fill_narrow_gaps(ctx.semantic_map, foreground_binary);
    }

    out.semantic_map = ctx.semantic_map;
    if (ctx.size != cv::Size(width, height)) cv::resize(ctx.semantic_map, out.semantic_map, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    out.point_coords = std::move(ctx.point_coords);
    out.point_labels = std::move(ctx.point_labels);
    return out;
}

std::vector<std::vector<Component>> SemanticRefiner::refine_classes(Context& ctx, const std::vector<int>& semantic_ids, const cv::Mat& semantic_labels,
                                                                    const cv::Mat& foreground_binary) {
    std::vector<std::vector<Component>> class_components;  // 添字は class_id - 1
    const size_t total = static_cast<size_t>(ctx.size.width) * ctx.size.height;
    for (size_t k = 0; k < semantic_ids.size(); ++k) {
        const int class_id = static_cast<int>(k) + 1;
        cv::Mat coarse_mask(semantic_labels.size(), CV_8U);
        for (int y = 0; y < coarse_mask.rows; ++y)
            for (int x = 0; x < coarse_mask.cols; ++x)
                coarse_mask.at<uint8_t>(y, x) = (semantic_labels.at<int>(y, x) == semantic_ids[k] && foreground_binary.at<uint8_t>(y, x)) ? 1 : 0;
        std::vector<Component> comps = connected_components(coarse_mask);

        // 孤立した極小成分（背景クラスタの斑点）はクラスの推定に混ぜない。混ぜると、斑点に置いた点を
        // 経由して物体のマスクが周囲の背景へ広がる。斑点は後段で個別に（範囲を制限して）扱う
        std::vector<Component> main;
        for (const Component& c : comps)
            if (cv::countNonZero(c.region) >= params::MIN_ISOLATED_PATCHES) main.push_back(c);
        if (main.empty()) main = comps;
        cv::Mat coarse_main = cv::Mat::zeros(coarse_mask.size(), CV_8U);
        for (const Component& c : main) cv::bitwise_or(coarse_main, c.region, coarse_main);

        RegionResult r = refine_region(ctx, coarse_main, main, "クラス" + std::to_string(class_id));

        // 重なり部分はスコアの高いクラスを優先
        simd::update_best(r.binary.data, r.score.ptr<float>(), ctx.best_score.ptr<float>(), ctx.semantic_map.data, total, static_cast<uint8_t>(class_id));
        class_components.push_back(std::move(comps));
    }
    return class_components;
}

void SemanticRefiner::rescue_uncovered(Context& ctx, const std::vector<std::vector<Component>>& class_components) {
    const int work_w = ctx.size.width, work_h = ctx.size.height;
    for (size_t k = 0; k < class_components.size(); ++k) {
        const int class_id = static_cast<int>(k) + 1;
        for (const Component& component : class_components[k]) {
            const int px = std::min(static_cast<int>((component.peak.x + 0.5) * work_w / feature_size_), work_w - 1);
            const int py = std::min(static_cast<int>((component.peak.y + 0.5) * work_h / feature_size_), work_h - 1);
            if (ctx.semantic_map.at<uint8_t>(py, px) != 0) continue;

            RegionResult r = refine_region(ctx, component.region, {component}, "クラス" + std::to_string(class_id) + "(再推定)");
            if (cv::countNonZero(component.region) < params::MIN_ISOLATED_PATCHES) {
                // 極小成分は、成分を1パッチ広げた範囲までに制限する
                cv::Mat allowed;
                cv::dilate(component.region.clone(), allowed, cv::Mat::ones(params::PATCH_NEIGHBORHOOD, params::PATCH_NEIGHBORHOOD, CV_8U));
                const cv::Mat allowed_full = resize_map(allowed, ctx.size, cv::INTER_NEAREST);
                for (int y = 0; y < work_h; ++y) {
                    uint8_t* b = r.binary.ptr<uint8_t>(y);
                    const float* a = allowed_full.ptr<float>(y);
                    for (int x = 0; x < work_w; ++x) b[x] = (b[x] && a[x] > 0.0f) ? 1 : 0;
                }
            }
            for (int y = 0; y < work_h; ++y) {
                const uint8_t* b = r.binary.ptr<uint8_t>(y);
                uint8_t* m = ctx.semantic_map.ptr<uint8_t>(y);
                for (int x = 0; x < work_w; ++x)
                    if (b[x] && m[x] == 0) m[x] = static_cast<uint8_t>(class_id);
            }
        }
    }
}

SemanticRefiner::RegionResult SemanticRefiner::refine_region(Context& ctx, const cv::Mat& coarse_mask, const std::vector<Component>& comps,
                                                             const std::string& tag) {
    cv::Mat core = cv::Mat::zeros(coarse_mask.size(), CV_8U);
    std::vector<cv::Point> peaks;
    for (const Component& c : comps) {
        cv::bitwise_or(core, c.core, core);
        peaks.push_back(c.peak);
    }
    const cv::Mat coarse_map = patch_mean(coarse_mask);
    cv::Mat prior_map;
    {
        Timer t("    ┌ 初期マスク作成", verbose_);
        prior_map = build_prior_map(ctx, coarse_mask, coarse_map, core);
    }

    const std::vector<cv::Point> fg_points = prompt_points(core, peaks, ctx.run->edgetam_foreground_points_per_class);
    const std::vector<float> logit = estimate_class_logit(ctx, prior_map, fg_points, tag);

    // 粗い前景らしさ（粗いマスクの周辺平均 × 前景確信度）
    const cv::Mat coarse_foreground_score = coarse_map.mul(ctx.foreground_map);

    RegionResult result;
    result.core = core;
    if (logit.empty()) {
        const cv::Mat coarse_full = resize_map(coarse_mask, ctx.size, cv::INTER_NEAREST);
        cv::Mat positive;
        cv::compare(coarse_full, 0.0, positive, cv::CMP_GT);
        result.binary = to_binary01(positive);
        result.score = resize_map(coarse_foreground_score, ctx.size, cv::INTER_LINEAR);
        return result;
    }

    Timer post("    ┌ マスク後処理", verbose_);
    const int m = high_res_feature_size_;
    const cv::Mat logit_low(m, m, CV_32F, const_cast<float*>(logit.data()));
    cv::Mat logit_full;
    cv::resize(logit_low, logit_full, ctx.size, 0, 0, cv::INTER_LINEAR);
    cv::Mat positive(ctx.size, CV_8U);
    simd::greater_than(logit_full.ptr<float>(), positive.data, static_cast<size_t>(ctx.size.width) * ctx.size.height, params::MASK_LOGIT_THRESHOLD);
    result.binary = clean_mask(positive, ctx.min_area);

    // 重なり解決用スコア = EdgeTAMの確信度 × ((1-w) + w × 粗い前景らしさ)
    cv::Mat prior_low;
    cv::resize(coarse_foreground_score, prior_low, cv::Size(m, m), 0, 0, cv::INTER_LINEAR);
    cv::Mat score_low(m, m, CV_32F);
    simd::score_from_logit(logit.data(), prior_low.ptr<float>(), score_low.ptr<float>(), static_cast<size_t>(m) * m, params::SCORE_PRIOR_WEIGHT);
    result.score = resize_map(score_low, ctx.size, cv::INTER_LINEAR);
    return result;
}

cv::Mat SemanticRefiner::build_prior_map(const Context& ctx, const cv::Mat& coarse_mask, const cv::Mat& coarse_map, const cv::Mat& core) const {
    const cv::Mat core_map = patch_mean(core);
    cv::Mat dilated;
    cv::dilate(coarse_mask.clone(), dilated, cv::Mat::ones(params::PATCH_NEIGHBORHOOD, params::PATCH_NEIGHBORHOOD, CV_8U));
    cv::Mat dilated_f;
    dilated.convertTo(dilated_f, CV_32F);
    const cv::Mat unknown = cv::max(coarse_map, dilated_f);
    const cv::Mat prior_map = cv::max(core_map, params::PRIOR_NON_CORE_PROBABILITY * unknown);

    cv::Mat upsampled;
    cv::resize(prior_map, upsampled, cv::Size(high_res_feature_size_, high_res_feature_size_), 0, 0, cv::INTER_LINEAR);
    cv::Mat filtered = guided_filter(ctx.guide, upsampled, guide_radius_, params::GUIDED_FILTER_EPS);
    simd::clamp(filtered.ptr<float>(), filtered.ptr<float>(), static_cast<size_t>(filtered.total()), 0.0f, 1.0f);
    return filtered;
}

std::vector<float> SemanticRefiner::estimate_class_logit(Context& ctx, const cv::Mat& prior_map, const std::vector<cv::Point>& foreground_points,
                                                         const std::string& tag) {
    std::vector<std::array<double, 2>> coords = transform_coords(foreground_points, ctx.width, ctx.height);
    std::vector<int32_t> labels(coords.size(), 1);
    coords.insert(coords.end(), ctx.background_coords.begin(), ctx.background_coords.end());
    labels.insert(labels.end(), ctx.background_coords.size(), 0);
    for (const auto& c : coords) ctx.point_coords.push_back({static_cast<float>(c[0]), static_cast<float>(c[1])});
    for (int32_t l : labels) ctx.point_labels.push_back(l);

    const int m = high_res_feature_size_;
    const size_t pixels = static_cast<size_t>(m) * m;
    // 初期マスク: ロジットは (2p-1)*PRIOR_LOGIT_RANGE（0.5=不明）
    std::vector<float> mask_input(pixels), best_logit(pixels);
    for (int y = 0; y < m; ++y)
        for (int x = 0; x < m; ++x) {
            const float p = prior_map.at<float>(y, x);
            mask_input[static_cast<size_t>(y) * m + x] = (2.0f * p - 1.0f) * params::PRIOR_LOGIT_RANGE;
        }

    std::vector<float> model_coords(coords.size() * 2);
    edgetam_.to_model_coords(coords, ctx.width, ctx.height, model_coords.data());

    // 前回のロジットを入力にして再推定を繰り返す
    bool have_logit = false;
    for (int it = 0; it < ctx.run->edgetam_refine_iterations; ++it) {
        Timer t("    ┌ edgetam " + tag + " " + std::to_string(it + 1) + "回目マスク推定", verbose_);
        const int best = edgetam_.decode_best(*ctx.feats, model_coords.data(), labels.data(), static_cast<int>(labels.size()), mask_input.data(), best_logit.data());
        if (best < 0) break;
        mask_input = best_logit;
        have_logit = true;
    }
    return have_logit ? best_logit : std::vector<float>();
}

}  // namespace pes
