// CLIP（PE）の OpenVINO backend。画像特徴量・テキスト特徴量・クラスタごとのトークンを求める（core/openvino/feature_extractor.py）
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "ov_common.h"
#include "tokenizer.h"

namespace pes {

class ClipExtractor {
public:
    ClipExtractor(ov::Core& core, const PesModelParams& params, std::unique_ptr<ClipTokenizer> tokenizer);

    int target_size() const { return target_size_; }
    int patch_size() const { return patch_size_; }
    int feature_size() const { return feature_size_; }
    static constexpr float kMean[3] = {0.5f, 0.5f, 0.5f};  // 画像の正規化（CLIP）
    static constexpr float kStd[3] = {0.5f, 0.5f, 0.5f};

    // extract_image_features の結果。次に extract_image_features を呼ぶまで有効（モデルの出力バッファをそのまま指す）
    struct ImageFeatures {
        ov::Tensor raw_patch;    // (1, N+1, C) 先頭は CLS トークン。クラスタごとのトークン抽出の入力
        const float* fused;      // (1, N+1, Cf) 層を連結して正規化した特徴。先頭は CLS
        int num_tokens;          // N + 1
        int fused_dim;           // Cf
    };

    // (1,3,S,S) の float 画像テンソル -> 特徴量
    ImageFeatures extract_image_features(const float* image_tensor);

    // プロンプト -> (N, D) の L2 正規化済みテキスト特徴量（行優先）
    std::vector<float> extract_text_features(const std::vector<std::string>& prompts);
    int text_feature_dim() const { return text_dim_; }

    // パッチ特徴とクラスタ番号（長さ N）-> クラスタごとの代表トークン (K, D)、K = 最大ラベル + 1。次の呼び出しまで有効
    // 戻り値のポインタは行優先で num_clusters * D 個
    const float* extract_cluster_tokens(const ImageFeatures& features, const float* cluster_labels, int num_labels, int* num_clusters, int* dim);

private:
    void warm_up();

    int target_size_;
    int patch_size_ = 0;
    int feature_size_ = 0;
    int text_dim_ = 0;
    std::unique_ptr<ClipTokenizer> tokenizer_;

    ov::CompiledModel visual_, text_, patch_logits_;
    ov::InferRequest visual_req_, text_req_, patch_logits_req_;
};

}  // namespace pes
