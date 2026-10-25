#include "mask_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "common.h"
#include "params.h"
#include "simd.h"

namespace pes {

namespace {

const cv::Mat& structure3x3() {
    static const cv::Mat kernel = cv::Mat::ones(params::PATCH_NEIGHBORHOOD, params::PATCH_NEIGHBORHOOD, CV_8U);
    return kernel;
}

// 0 / 1 の u8 マスクの erode / dilate（3x3）。境界は scipy.ndimage.binary_erosion(border_value=1) / binary_dilation(border_value=0) と同じ
// （cv::erode は既定で境界の外を +inf、cv::dilate は -inf として扱う）
cv::Mat erode3x3(const cv::Mat& m) {
    cv::Mat out;
    cv::erode(m.clone(), out, structure3x3());
    return out;
}
cv::Mat dilate3x3(const cv::Mat& m) {
    cv::Mat out;
    cv::dilate(m.clone(), out, structure3x3());
    return out;
}

int count_nonzero(const cv::Mat& m) { return cv::countNonZero(m); }

// 行優先で真の画素の座標 (x, y)
std::vector<cv::Point> argwhere_xy(const cv::Mat& mask) {
    std::vector<cv::Point> points;
    for (int y = 0; y < mask.rows; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < mask.cols; ++x)
            if (row[x]) points.emplace_back(x, y);
    }
    return points;
}

// 1 次元の二乗ユークリッド距離変換（Felzenszwalb & Huttenlocher）。arg[q] は q に最も近い特徴点（f が有限の点）の位置
void distance_transform_1d(const double* f, int n, double* d, int* arg, int* v, double* z) {
    int k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < n; ++q) {
        double s = ((f[q] + static_cast<double>(q) * q) - (f[v[k]] + static_cast<double>(v[k]) * v[k])) / (2.0 * q - 2.0 * v[k]);
        while (s <= z[k]) {
            --k;
            s = ((f[q] + static_cast<double>(q) * q) - (f[v[k]] + static_cast<double>(v[k]) * v[k])) / (2.0 * q - 2.0 * v[k]);
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = std::numeric_limits<double>::infinity();
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        d[q] = static_cast<double>(q - v[k]) * (q - v[k]) + f[v[k]];
        arg[q] = v[k];
    }
}

}  // namespace

cv::Mat to_binary01(const cv::Mat& mask) {
    cv::Mat out;
    cv::threshold(mask, out, 0, 1, cv::THRESH_BINARY);
    return out;
}

cv::Mat mask_not(const cv::Mat& a) {
    cv::Mat eq;
    cv::compare(a, 0, eq, cv::CMP_EQ);
    return to_binary01(eq);
}

// =============================================================================
// 点のサンプリング
// =============================================================================

std::vector<cv::Point> farthest_point_sampling(const std::vector<cv::Point>& points, int k) {
    const int m = static_cast<int>(points.size());
    if (k < 0 || k > m) throw Error(PES_ERROR_INVALID_ARGUMENT, "farthest_point_sampling: k は 0 以上 点の数 以下にしてください");
    if (k == 0) return {};
    if (k == m) return points;

    // 最初は重心に最も近い点を選ぶ
    double cx = 0.0, cy = 0.0;
    for (const cv::Point& p : points) {
        cx += p.x;
        cy += p.y;
    }
    cx /= m;
    cy /= m;
    int first = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i) {
        const double dx = points[i].x - cx, dy = points[i].y - cy;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best) {
            best = d2;
            first = i;
        }
    }

    std::vector<int> selected{first};
    // 各点から、選択済みの点までの最小二乗距離
    std::vector<double> min_dist2(m);
    for (int i = 0; i < m; ++i) {
        const double dx = points[i].x - points[first].x, dy = points[i].y - points[first].y;
        min_dist2[i] = dx * dx + dy * dy;
    }
    min_dist2[first] = -1.0;

    for (int i = 1; i < k; ++i) {
        // 選択済みの点から最も遠い候補を選ぶ（同じなら先頭）
        int idx = 0;
        for (int j = 1; j < m; ++j)
            if (min_dist2[j] > min_dist2[idx]) idx = j;
        selected.push_back(idx);
        for (int j = 0; j < m; ++j) {
            const double dx = points[j].x - points[idx].x, dy = points[j].y - points[idx].y;
            min_dist2[j] = std::min(min_dist2[j], dx * dx + dy * dy);
        }
        for (int s : selected) min_dist2[s] = -1.0;  // 重複選択を防ぐ
    }

    std::vector<cv::Point> out;
    out.reserve(k);
    for (int s : selected) out.push_back(points[s]);
    return out;
}

std::vector<cv::Point> sample_points(const cv::Mat& candidate, int k) {
    const std::vector<cv::Point> points = argwhere_xy(candidate);
    return farthest_point_sampling(points, std::min<int>(k, static_cast<int>(points.size())));
}

std::vector<cv::Point> prompt_points(const cv::Mat& core, const std::vector<cv::Point>& peaks, int k) {
    std::vector<cv::Point> selected = peaks;
    const std::vector<cv::Point> points = argwhere_xy(core);
    if (!points.empty() && static_cast<int>(selected.size()) < k) {
        std::vector<double> min_d2(points.size(), std::numeric_limits<double>::infinity());
        for (size_t i = 0; i < points.size(); ++i)
            for (const cv::Point& s : selected) {
                const double dx = points[i].x - s.x, dy = points[i].y - s.y;
                min_d2[i] = std::min(min_d2[i], dx * dx + dy * dy);
            }
        while (static_cast<int>(selected.size()) < k) {
            size_t idx = 0;
            for (size_t i = 1; i < points.size(); ++i)
                if (min_d2[i] > min_d2[idx]) idx = i;
            if (!(min_d2[idx] > 0.0)) break;  // 残りの点がすべて選択済みの点と重なる
            selected.push_back(points[idx]);
            for (size_t i = 0; i < points.size(); ++i) {
                const double dx = points[i].x - points[idx].x, dy = points[i].y - points[idx].y;
                min_d2[i] = std::min(min_d2[i], dx * dx + dy * dy);
            }
        }
    }
    return selected;
}

std::vector<cv::Point> select_background_points(const cv::Mat& foreground_binary, int count, const cv::Mat& foreground_map) {
    const cv::Mat background = mask_not(foreground_binary);
    cv::Mat labeled;
    const int num_labels = cv::connectedComponents(background, labeled, 4, CV_32S);

    // 画像端につながる背景の成分
    std::vector<uint8_t> touches_border(num_labels, 0);
    for (int x = 0; x < labeled.cols; ++x) {
        touches_border[labeled.at<int>(0, x)] = 1;
        touches_border[labeled.at<int>(labeled.rows - 1, x)] = 1;
    }
    for (int y = 0; y < labeled.rows; ++y) {
        touches_border[labeled.at<int>(y, 0)] = 1;
        touches_border[labeled.at<int>(y, labeled.cols - 1)] = 1;
    }
    touches_border[0] = 0;
    cv::Mat outer(labeled.size(), CV_8U);
    for (int y = 0; y < labeled.rows; ++y)
        for (int x = 0; x < labeled.cols; ++x) outer.at<uint8_t>(y, x) = touches_border[labeled.at<int>(y, x)];
    if (count_nonzero(outer) == 0) outer = background;

    // 前景（outer でない画素）からの距離。画像の端は前景として扱わない
    cv::Mat dist;
    cv::distanceTransform(outer, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);

    cv::Mat candidate;
    for (int margin : {params::BACKGROUND_POINT_MARGIN, 1}) {
        candidate = cv::Mat::zeros(outer.size(), CV_8U);
        for (int y = 0; y < outer.rows; ++y)
            for (int x = 0; x < outer.cols; ++x)
                candidate.at<uint8_t>(y, x) = (outer.at<uint8_t>(y, x) && dist.at<float>(y, x) >= static_cast<float>(margin)) ? 1 : 0;
        if (count_nonzero(candidate) > 0) break;
    }

    if (!foreground_map.empty()) {
        // 確信度の低い（値の小さい）パッチから count 個以上そろうところまでを候補にする
        std::vector<float> values;
        for (int y = 0; y < candidate.rows; ++y)
            for (int x = 0; x < candidate.cols; ++x)
                if (candidate.at<uint8_t>(y, x)) values.push_back(foreground_map.at<float>(y, x));
        if (static_cast<int>(values.size()) >= count && count > 0) {
            std::nth_element(values.begin(), values.begin() + (count - 1), values.end());
            const float level = values[count - 1];
            for (int y = 0; y < candidate.rows; ++y)
                for (int x = 0; x < candidate.cols; ++x)
                    if (candidate.at<uint8_t>(y, x) && !(foreground_map.at<float>(y, x) <= level)) candidate.at<uint8_t>(y, x) = 0;
        }
    }
    return sample_points(candidate, count);
}

// =============================================================================
// マスクの解析・整形
// =============================================================================

std::vector<Component> connected_components(const cv::Mat& mask) {
    cv::Mat labels;
    const int n = cv::connectedComponents(to_binary01(mask), labels, 8, CV_32S);
    std::vector<Component> result;
    for (int i = 1; i < n; ++i) {
        cv::Mat region;
        cv::compare(labels, i, region, cv::CMP_EQ);
        region = to_binary01(region);

        // 最内点: 成分の外（画像の外も含む）までの距離が最大の位置（同じなら行優先で先頭）
        cv::Mat padded, edt;
        cv::copyMakeBorder(region, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::distanceTransform(padded, edt, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);
        cv::Point peak(0, 0);
        float best = -1.0f;
        for (int y = 0; y < region.rows; ++y)
            for (int x = 0; x < region.cols; ++x) {
                const float v = edt.at<float>(y + 1, x + 1);
                if (v > best) {
                    best = v;
                    peak = cv::Point(x, y);
                }
            }

        cv::Mat core = erode3x3(region);
        result.push_back({peak, count_nonzero(core) > 0 ? core : region, region});
    }
    return result;
}

bool bounding_box(const cv::Mat& mask, int margin, cv::Rect* box) {
    const cv::Rect r = cv::boundingRect(mask);
    if (r.width <= 0 || r.height <= 0) return false;
    const int x0 = std::max(r.x - margin, 0), y0 = std::max(r.y - margin, 0);
    const int x1 = std::min(r.x + r.width + margin, mask.cols), y1 = std::min(r.y + r.height + margin, mask.rows);
    *box = cv::Rect(x0, y0, x1 - x0, y1 - y0);
    return true;
}

namespace {

// 外接矩形で切り出した 0/1 のマスク（余白つき）の小さな成分を除去し、メディアンフィルタで平滑化する
cv::Mat clean_mask_crop(const cv::Mat& crop, int min_area) {
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(crop, labels, stats, centroids, 8, CV_32S);

    // 面積が足りる連結成分だけを残す（背景ラベル0は常に除外）
    std::vector<uint8_t> keep(n, 0);
    bool keep_all = true;
    for (int i = 1; i < n; ++i) {
        keep[i] = stats.at<int>(i, cv::CC_STAT_AREA) >= min_area ? 1 : 0;
        keep_all = keep_all && keep[i];
    }
    cv::Mat kept;
    if (keep_all) {
        kept = crop;  // 全成分を残すなら、そのまま
    } else {
        kept.create(crop.size(), CV_8U);
        for (int y = 0; y < crop.rows; ++y) {
            const int* l = labels.ptr<int>(y);
            uint8_t* k = kept.ptr<uint8_t>(y);
            for (int x = 0; x < crop.cols; ++x) k[x] = keep[l[x]];
        }
    }
    // 輪郭の細かなノイズを平滑化
    cv::Mat smoothed;
    cv::medianBlur(kept, smoothed, params::MEDIAN_BLUR_KSIZE);
    return smoothed;
}

}  // namespace

cv::Mat clean_mask(const cv::Mat& mask, int min_area, cv::Rect* box_out) {
    const cv::Mat binary = to_binary01(mask);
    cv::Mat cleaned = cv::Mat::zeros(binary.size(), CV_8U);
    cv::Rect box;
    if (!bounding_box(binary, 1, &box)) {
        if (box_out) *box_out = cv::Rect();
        return cleaned;
    }
    clean_mask_crop(binary(box).clone(), min_area).copyTo(cleaned(box));
    if (box_out) *box_out = box;
    return cleaned;
}

void remove_small_regions(cv::Mat& semantic_map, int min_area) {
    const int height = semantic_map.rows, width = semantic_map.cols;

    // 1. クラスごとに小さな連結成分を背景へ戻す（存在しないクラスは外接矩形が空なので飛ばされる）
    double max_class = 0.0;
    cv::minMaxLoc(semantic_map, nullptr, &max_class);
    for (int class_id = 1; class_id <= static_cast<int>(max_class); ++class_id) {
        cv::Mat region;
        cv::compare(semantic_map, class_id, region, cv::CMP_EQ);
        cv::Rect box;
        if (!bounding_box(region, 1, &box)) continue;  // 外接矩形の中だけで処理する（clean_mask の結果は外接矩形の外で常に 0）
        const cv::Mat crop = to_binary01(region(box).clone());
        const cv::Mat keep = clean_mask_crop(crop, min_area);
        cv::Mat target = semantic_map(box);
        for (int y = 0; y < box.height; ++y) {
            const uint8_t* c = crop.ptr<uint8_t>(y);
            const uint8_t* k = keep.ptr<uint8_t>(y);
            uint8_t* t = target.ptr<uint8_t>(y);
            for (int x = 0; x < box.width; ++x)
                if (c[x] && !k[x]) t[x] = 0;
        }
    }

    // 2. 背景に残った小さな穴を、周囲で最も多いクラスで埋める
    cv::Mat background, labels, stats, centroids;
    cv::compare(semantic_map, 0, background, cv::CMP_EQ);
    const int n = cv::connectedComponentsWithStats(background, labels, stats, centroids, 8, CV_32S);
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area) continue;
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int x0 = std::max(x - 1, 0), y0 = std::max(y - 1, 0);
        const int x1 = std::min(x + w + 1, width), y1 = std::min(y + h + 1, height);
        const cv::Rect window(x0, y0, x1 - x0, y1 - y0);

        cv::Mat hole;
        cv::compare(labels(window), i, hole, cv::CMP_EQ);
        hole = to_binary01(hole);
        const cv::Mat dilated = dilate3x3(hole);
        cv::Mat target = semantic_map(window);
        std::array<int, 256> neighbor_counts{};
        bool any = false;
        for (int yy = 0; yy < window.height; ++yy)
            for (int xx = 0; xx < window.width; ++xx) {
                if (dilated.at<uint8_t>(yy, xx) && !hole.at<uint8_t>(yy, xx)) {
                    const uint8_t v = target.at<uint8_t>(yy, xx);
                    if (v > 0) {
                        ++neighbor_counts[v];
                        any = true;
                    }
                }
            }
        if (!any) continue;
        const int majority = static_cast<int>(std::max_element(neighbor_counts.begin(), neighbor_counts.end()) - neighbor_counts.begin());
        for (int yy = 0; yy < window.height; ++yy)
            for (int xx = 0; xx < window.width; ++xx)
                if (hole.at<uint8_t>(yy, xx)) target.at<uint8_t>(yy, xx) = static_cast<uint8_t>(majority);
    }
}

cv::Mat nearest_feature_index(const cv::Mat& is_feature) {
    const int rows = is_feature.rows, cols = is_feature.cols;
    cv::Mat index(rows, cols, CV_32S, cv::Scalar(-1));
    if (count_nonzero(is_feature) == 0) return index;

    const double kInf = 1e20;
    const int longest = std::max(rows, cols);
    std::vector<double> f(longest), d(longest), z(longest + 1);
    std::vector<int> arg(longest), v(longest);
    cv::Mat dist(rows, cols, CV_64F);
    cv::Mat near_row(rows, cols, CV_32S);

    // 列方向: 各列の中で最も近い特徴画素までの二乗距離と、その行
    for (int x = 0; x < cols; ++x) {
        for (int y = 0; y < rows; ++y) f[y] = is_feature.at<uint8_t>(y, x) ? 0.0 : kInf;
        distance_transform_1d(f.data(), rows, d.data(), arg.data(), v.data(), z.data());
        for (int y = 0; y < rows; ++y) {
            dist.at<double>(y, x) = d[y];
            near_row.at<int>(y, x) = arg[y];
        }
    }
    // 行方向: 各行で、列方向の結果を合成する
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) f[x] = dist.at<double>(y, x);
        distance_transform_1d(f.data(), cols, d.data(), arg.data(), v.data(), z.data());
        for (int x = 0; x < cols; ++x) {
            if (d[x] >= kInf / 2) continue;
            const int nx = arg[x];
            index.at<int>(y, x) = near_row.at<int>(y, nx) * cols + nx;
        }
    }
    return index;
}

void fill_narrow_gaps(cv::Mat& semantic_map, const cv::Mat& foreground) {
    const int n = foreground.rows;
    const cv::Mat core = erode3x3(to_binary01(foreground));
    if (count_nonzero(core) == 0) return;

    // 1 パッチ = GAP_FILL_PIXELS_PER_PATCH 画素に縮小したマスクで、閉処理で埋まる画素（隙間）を求める
    const int pixels = params::GAP_FILL_PIXELS_PER_PATCH;
    const int side = n * pixels;
    const int height = semantic_map.rows, width = semantic_map.cols;

    cv::Mat foreground255, small, united;
    cv::compare(semantic_map, 0, foreground255, cv::CMP_GT);
    cv::resize(foreground255, small, cv::Size(side, side), 0, 0, cv::INTER_AREA);
    cv::compare(small, 128, united, cv::CMP_GE);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(pixels + 1, pixels + 1));
    cv::Mat closed;
    cv::morphologyEx(united, closed, cv::MORPH_CLOSE, kernel);

    cv::Mat core255, core_small, not_united, gap_small;
    core.convertTo(core255, CV_8U, 255.0);
    cv::resize(core255, core_small, cv::Size(side, side), 0, 0, cv::INTER_NEAREST);
    cv::bitwise_not(united, not_united);
    cv::bitwise_and(closed, not_united, gap_small);
    gap_small = dilate3x3(gap_small);  // 1 画素広げて、縮小による取りこぼしを防ぐ
    cv::bitwise_and(gap_small, core_small, gap_small);
    if (count_nonzero(gap_small) == 0) return;

    cv::Mat gap_full, background;
    cv::resize(gap_small, gap_full, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::compare(semantic_map, 0, background, cv::CMP_EQ);
    cv::Mat gap;
    cv::bitwise_and(gap_full, background, gap);
    if (count_nonzero(gap) == 0) return;

    // 隙間ごとに、外接矩形（最寄りのクラスが入るよう 1 パッチ分広げる）の中で、最寄りの前景画素のクラスを与える
    const int margin = static_cast<int>(std::ceil(static_cast<double>(std::max(height, width)) / n));
    const cv::Mat ring_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(margin | 1, margin | 1));  // 半径 約 半パッチ
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(gap, labels, stats, centroids, 8, CV_32S);
    for (int i = 1; i < count; ++i) {
        const int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        const int y0 = std::max(y - margin, 0), y1 = std::min(y + h + margin, height);
        const int x0 = std::max(x - margin, 0), x1 = std::min(x + w + margin, width);
        const cv::Rect window(x0, y0, x1 - x0, y1 - y0);
        cv::Mat crop = semantic_map(window);  // semantic_map と同じメモリ

        cv::Mat fill;
        cv::compare(labels(window), i, fill, cv::CMP_EQ);
        fill = to_binary01(fill);
        // 隙間の両側（半パッチ以内）が 2 クラス以上に接しているときだけ埋める（物体の凹みや、1 クラスの物体の中の穴は埋めない）
        cv::Mat ring;
        cv::dilate(fill, ring, ring_kernel);
        std::array<bool, 256> seen{};
        int distinct = 0;
        for (int yy = 0; yy < crop.rows; ++yy)
            for (int xx = 0; xx < crop.cols; ++xx) {
                const uint8_t c = crop.at<uint8_t>(yy, xx);
                if (ring.at<uint8_t>(yy, xx) && !fill.at<uint8_t>(yy, xx) && c > 0 && !seen[c]) {
                    seen[c] = true;
                    ++distinct;
                }
            }
        if (distinct < 2) continue;

        cv::Mat is_foreground;
        cv::compare(crop, 0, is_foreground, cv::CMP_GT);
        const cv::Mat nearest = nearest_feature_index(is_foreground);
        cv::Mat before = crop.clone();  // 埋める前の値から引く（埋めた画素は参照しない）
        for (int yy = 0; yy < crop.rows; ++yy)
            for (int xx = 0; xx < crop.cols; ++xx) {
                if (!fill.at<uint8_t>(yy, xx)) continue;
                const int idx = nearest.at<int>(yy, xx);
                if (idx >= 0) crop.at<uint8_t>(yy, xx) = before.at<uint8_t>(idx / crop.cols, idx % crop.cols);
            }
    }
}

// =============================================================================
// パッチ格子 <-> 作業解像度の変換
// =============================================================================

cv::Mat patch_mean(const cv::Mat& patch_map) {
    cv::Mat as_float, out;
    patch_map.convertTo(as_float, CV_32F);
    cv::blur(as_float, out, cv::Size(params::PATCH_NEIGHBORHOOD, params::PATCH_NEIGHBORHOOD), cv::Point(-1, -1), cv::BORDER_REFLECT);
    return out;
}

cv::Mat guided_filter(const cv::Mat& guide, const cv::Mat& src, int radius, float eps) {
    const cv::Size ksize(2 * radius + 1, 2 * radius + 1);
    auto box = [&](const cv::Mat& x) {
        cv::Mat out;
        cv::boxFilter(x, out, -1, ksize, cv::Point(-1, -1), true, cv::BORDER_REFLECT);
        return out;
    };
    const cv::Mat mean_i = box(guide), mean_p = box(src);
    const cv::Mat a = (box(guide.mul(src)) - mean_i.mul(mean_p)) / (box(guide.mul(guide)) - mean_i.mul(mean_i) + eps);
    const cv::Mat b = mean_p - a.mul(mean_i);
    return box(a).mul(guide) + box(b);
}

cv::Mat resize_map(const cv::Mat& patch_map, cv::Size size, int interpolation) {
    cv::Mat as_float, out;
    patch_map.convertTo(as_float, CV_32F);
    cv::resize(as_float, out, size, 0, 0, interpolation);
    return out;
}

// =============================================================================
// kNN グラフ
// =============================================================================

KnnGraph knn_graph(const float* features, int n, int dim, int k) {
    std::vector<float> similarity(static_cast<size_t>(n) * n);
    simd::gram_symmetric(features, n, dim, similarity.data());
    std::vector<int> neighbors(static_cast<size_t>(n) * k);
    simd::topk_rows(similarity.data(), n, k, true, neighbors.data());

    // 有向グラフ directed の i -> j と j -> i のどちらかがあれば辺とする（directed.maximum(directed.T)）
    std::vector<std::vector<int>> adjacency(n);
    for (int i = 0; i < n; ++i)
        for (int t = 0; t < k; ++t) {
            const int j = neighbors[static_cast<size_t>(i) * k + t];
            if (j < 0) continue;
            adjacency[i].push_back(j);
            adjacency[j].push_back(i);
        }

    KnnGraph graph;
    graph.n = n;
    graph.indptr.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        std::sort(adjacency[i].begin(), adjacency[i].end());
        adjacency[i].erase(std::unique(adjacency[i].begin(), adjacency[i].end()), adjacency[i].end());
        graph.indptr[i + 1] = graph.indptr[i] + static_cast<int>(adjacency[i].size());
    }
    graph.indices.reserve(graph.indptr[n]);
    graph.weights.reserve(graph.indptr[n]);
    for (int i = 0; i < n; ++i) {
        // 行を正規化する。Python 版（scipy）は float32 のまま 1 / 次数 を求める
        const float w = 1.0f / static_cast<float>(adjacency[i].size());
        for (int j : adjacency[i]) {
            graph.indices.push_back(j);
            graph.weights.push_back(w);
        }
    }
    return graph;
}

std::vector<float> propagate(const std::vector<float>& values, const KnnGraph& graph, double alpha, int iterations) {
    // Python 版は float32 のまま計算する（値・グラフ・係数がすべて float32）
    const float one_minus_alpha = static_cast<float>(1.0 - alpha), alpha_f = static_cast<float>(alpha);
    std::vector<float> propagated = values, next(values.size());
    for (int it = 0; it < iterations; ++it) {
        for (int i = 0; i < graph.n; ++i) {
            float acc = 0.0f;
            for (int jj = graph.indptr[i]; jj < graph.indptr[i + 1]; ++jj) {
                const float product = graph.weights[jj] * propagated[graph.indices[jj]];
                acc += product;
            }
            const float a = one_minus_alpha * values[i];
            const float b = alpha_f * acc;
            next[i] = a + b;
        }
        propagated.swap(next);
    }
    return propagated;
}

}  // namespace pes
