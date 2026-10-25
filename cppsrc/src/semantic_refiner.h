// 粗いセマンティックセグメンテーションを EdgeTAM で精緻化し、1枚のセマンティックマップに統合する（core/common/semantic_refiner.py）
#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "edgetam_runner.h"
#include "mask_utils.h"
#include "pe_segmentation.h"

namespace pes {

// SemanticRefiner::refine の結果
struct RefineOutput {
    cv::Mat semantic_map;                                // (height, width) CV_8U。0=背景、1~=前景クラス
    std::vector<int> prompt_indices;                     // クラス1~に対応する、前景プロンプトのインデックス
    std::vector<std::array<float, 2>> point_coords;      // EdgeTAM に与えたプロンプト点（画像座標 (x, y)、全クラス分）
    std::vector<int> point_labels;                       // 上の点のラベル（1=正例、0=負例）
};

class SemanticRefiner {
public:
    SemanticRefiner(EdgeTamRunner& edgetam, int feature_size);

    void set_verbose(bool verbose) { verbose_ = verbose; }

    // 粗いセマンティックセグメンテーションを EdgeTAM で精緻化し、1枚のセマンティックマップに統合する。
    // クラスごとに EdgeTAM で前景マスクを推定し、重なり部分は「EdgeTAM の確信度 × 粗い前景らしさ(foreground_map)」が高いクラスを優先する。
    // 背景は全クラス共通でクラス0、前景は1から振る。どのクラスのマスクにも入らなかった粗い連結成分は、その成分だけで再推定する。
    //
    // resized: リサイズ済み（正規化前、RGB、image_size 四方）の画像。初期マスクの縁を整えるガイドになる。
    // foreground_map: (F, F) CV_32F、foreground_mask: (F, F) CV_8U、semantic_labels: (F, F) CV_32S。labels_list: 連番 -> プロンプト番号
    // feats: EdgeTAM エンコーダの出力
    RefineOutput refine(const cv::Mat& resized, const cv::Mat& foreground_map, const cv::Mat& foreground_mask, const cv::Mat& semantic_labels,
                        const std::vector<int>& labels_list, int width, int height, const EdgeTamRunner::Features& feats,
                        const PesRunParams& run);

private:
    // 1枚の画像を精緻化する間、各段階で共有する入力と、更新していく結果
    struct Context {
        const EdgeTamRunner::Features* feats;
        std::vector<std::array<double, 2>> background_coords;  // 背景の負例点（画像座標）
        cv::Mat foreground_map;                                // 粗い前景らしさ（パッチ格子）
        int width, height;                                     // 元画像サイズ
        cv::Size size;                                         // 作業解像度 (幅, 高さ)
        int min_area;                                          // ノイズとして除去する最小面積（作業解像度の画素数）
        cv::Mat guide;                                         // 初期マスクの縁を整えるガイド（入力画像の輝度。高解像度特徴量と同じ大きさ、CV_32F、0〜1）
        const PesRunParams* run;

        cv::Mat semantic_map;  // 作業解像度。0=背景、1~=前景クラス
        cv::Mat best_score;    // 各画素で勝ったクラスのスコア
        std::vector<std::array<float, 2>> point_coords;
        std::vector<int> point_labels;
    };
    struct RegionResult {
        cv::Mat binary;  // 作業解像度の 2 値マスク (CV_8U, 0/1)
        cv::Mat score;   // 作業解像度の重なり解決用スコア (CV_32F)
        cv::Mat core;
    };

    // クラスごとに EdgeTAM でマスクを推定し、スコアの高いクラスを優先して semantic_map に書き込む。クラス番号 -> 粗い連結成分のリストを返す
    std::vector<std::vector<Component>> refine_classes(Context& ctx, const std::vector<int>& semantic_ids, const cv::Mat& semantic_labels,
                                                              const cv::Mat& foreground_binary);
    // 取りこぼし防止: 最内点がどのクラスにも覆われていない粗い連結成分は、単独で再推定する
    void rescue_uncovered(Context& ctx, const std::vector<std::vector<Component>>& class_components);
    // 粗いマスクに対する EdgeTAM の 2 値マスク・重なり解決用スコア・core を返す
    RegionResult refine_region(Context& ctx, const cv::Mat& coarse_mask, const std::vector<Component>& comps, const std::string& tag);
    // EdgeTAM に初期マスクとして与える、前景確率のマップ（高解像度特徴量と同じ大きさ、0〜1）
    cv::Mat build_prior_map(const Context& ctx, const cv::Mat& coarse_mask, const cv::Mat& coarse_map, const cv::Mat& core) const;
    // 正例点と共通の負例点で EdgeTAM を反復実行し、低解像度ロジット (mask_size²) を返す。実行しなければ空
    std::vector<float> estimate_class_logit(Context& ctx, const cv::Mat& prior_map, const std::vector<cv::Point>& foreground_points,
                                            const std::string& tag);
    // パッチ格子の座標 (x, y) を、各パッチの中心に当たる画像座標へ変換する
    std::vector<std::array<double, 2>> transform_coords(const std::vector<cv::Point>& pts, int width, int height) const;
    // 初期マスクの縁を整えるガイド。入力画像の輝度を、EdgeTAM の高解像度特徴量の大きさに縮小したもの
    cv::Mat make_guide(const cv::Mat& resized) const;

    EdgeTamRunner& edgetam_;
    int feature_size_;
    int high_res_feature_size_;  // EdgeTAM の高解像度特徴量の一辺の画素数
    int guide_radius_;           // ガイデッドフィルタの半径 = 粗いマスクの信頼限界（1パッチ）が、高解像度特徴量の何画素か
    bool verbose_ = false;
};

}  // namespace pes
