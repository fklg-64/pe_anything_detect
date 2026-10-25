// CLIP 特徴量と KMeans による、パッチ単位の粗い前景/セマンティック推定（core/common/coarse_segmentation.py）
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "clip_extractor.h"
#include "pe_segmentation.h"

namespace pes {

// KMeans のクラスタ分けと、クラスタごとの分類結果（パッチ格子上）
struct ClusterView {
    std::vector<int> labels;      // パッチごとのクラスタ番号（平坦）
    std::vector<int> index_list;  // クラスタ番号ごとの対応プロンプト番号（前景プロンプト -> 背景プロンプトの通し番号）
};

// generate_foreground_map の結果（パッチ格子上）
struct CoarseForeground {
    cv::Mat foreground_map;   // (F, F) CV_32F 前景確信度
    cv::Mat foreground_mask;  // (F, F) CV_8U 確信度が閾値を超えるパッチを 1 とした 2 値マスク
    ClusterView clusters;     // EdgeTAM 併用時は PE のクラスタ × 伝播後のプロンプトの組
    bool has_pe_clusters = false;
    ClusterView pe_clusters;  // 伝播前の PE だけのクラスタ（EdgeTAM を併用しないときは空）
    double kmeans_ms = 0.0, cluster_tokens_ms = 0.0, propagate_ms = 0.0;
};

// クラスタの代表トークンを、前景プロンプト（先頭 num_fg 個）・背景プロンプト（続く num_bg 個）のテキスト特徴量 text_all (M, D) と照合し、
// (前景確信度, 最も近いプロンプト番号) を返す。プロンプト番号は 前景プロンプト -> 背景プロンプトの通し番号
std::pair<float, int> classify_cluster_feature(const float* cluster_feature, int dim, const float* text_all, int num_fg, int num_bg, double threshold);

class CoarseSegmenter {
public:
    explicit CoarseSegmenter(ClipExtractor& extractor) : extractor_(extractor), feature_size_(extractor.feature_size()), text_dim_(extractor.text_feature_dim()) {}

    // 各段階の処理時間を標準出力に出すか
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // --- プロンプトとテキスト特徴量（前景プロンプト -> 背景プロンプトの順に並べて使う） ---
    // prompts_dir の foreground.yaml / background.yaml を読み、プロンプトを置き換えて特徴量を抽出する
    void load_prompts(const std::string& prompts_dir);
    // 追加（既にある文は飛ばす）。追加した分の特徴量を抽出して返り値は追加した数
    int add_prompts(PesPromptKind kind, const std::vector<std::string>& texts);
    void remove_prompt_at(PesPromptKind kind, int index);
    int find_prompt(PesPromptKind kind, const std::string& text) const;  // 無ければ -1
    void clear_prompts(PesPromptKind kind);
    const std::vector<std::string>& prompts(PesPromptKind kind) const { return kind == PES_PROMPT_FOREGROUND ? fg_prompts_ : bg_prompts_; }
    // 前景・背景の両方にプロンプトがあるか（無いと前景/背景を判定できない）
    bool ready() const { return !fg_prompts_.empty() && !bg_prompts_.empty(); }

    // --- 粗い推定 ---
    // edge_embed（EdgeTAM エンコーダの image_embed (C, h, w) の先頭。h·w はパッチ数と同じ）を渡すと、PE の判定で得た前景確信度を、
    // EdgeTAM の特徴の類似度が高いパッチどうしで伝播して揃える
    CoarseForeground generate_foreground_map(const ClipExtractor::ImageFeatures& features, const float* edge_embed, int edge_channels,
                                             const PesRunParams& run);

    // パッチごとの対応プロンプト番号を連番に直す。labels_list: 連番 -> プロンプト番号、semantic_labels: (F, F) CV_32S の連番ラベル
    void assign_semantic_labels(const std::vector<int>& cluster_labels, const std::vector<int>& cluster_index_list,
                                std::vector<int>* labels_list, cv::Mat* semantic_labels) const {
        assign_labels(feature_size_, cluster_labels, cluster_index_list, labels_list, semantic_labels);
    }
    static void assign_labels(int feature_size, const std::vector<int>& cluster_labels, const std::vector<int>& cluster_index_list,
                              std::vector<int>* labels_list, cv::Mat* semantic_labels);

    // パッチ単位の粗い結果を整える（穴埋め・細いクラス領域の統合）。前景マスク (CV_8U) とセマンティックラベル (CV_32S) を返す
    static void refine_coarse_segmentation(const cv::Mat& semantic_labels, const cv::Mat& foreground_mask, cv::Mat* out_mask, cv::Mat* out_labels);

    // --- テスト用に公開している部品 ---
    // 画像端に接しない背景の連結成分（穴）を前景にし、labels（in-place）には最寄りの前景ラベルを与える
    static cv::Mat fill_holes(const cv::Mat& fg, cv::Mat& labels, const cv::Mat& original_fg);
    // 細いクラス領域を、最も長く接する他クラスへ統合する（labels を in-place で更新）
    static void merge_thin_regions(const cv::Mat& fg, cv::Mat& labels);
    // 前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える。confidence / prompt_index はパッチ（平坦）ごと
    static void propagate_with_edgetam(std::vector<float>& confidence, std::vector<int>& prompt_index, const float* edge_embed, int channels,
                                       int num_patches, int num_foreground_prompts, double foreground_threshold);
    // クラスタ番号 0〜ごとの対応プロンプト番号（クラスタの最初のパッチから取る）
    static std::vector<int> prompt_per_cluster(const std::vector<int>& cluster_labels, const std::vector<int>& prompt_index);

private:
    ClipExtractor& extractor_;
    int feature_size_;
    std::vector<std::string> fg_prompts_, bg_prompts_;
    std::vector<float> fg_features_, bg_features_;  // (プロンプト数, D)
    // 前景 -> 背景の順に並べた全プロンプトの特徴量 (M, D)。Python 版の np.vstack([前景, 背景]) と同じ並びで 1 回の行列積にする
    // （行列積の丸め方が列の位置で変わるため、前景と背景を別々に計算してはいけない）
    std::vector<float> text_all_;
    bool text_all_dirty_ = true;
    int text_dim_ = 0;
    bool verbose_ = false;
};

}  // namespace pes
