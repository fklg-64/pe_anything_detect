// EdgeTAM の OpenVINO backend。推論呼び出し（エンコーダ・デコーダ）と、座標の前処理を持つ
// （core/openvino/edgetam_runner.py + core/common/edgetam_helper.py）
#pragma once

#include <memory>
#include <vector>

#include "common.h"
#include "ov_common.h"

namespace pes {

class EdgeTamRunner {
public:
    EdgeTamRunner(ov::Core& core, const PesModelParams& params);

    int target_size() const { return target_size_; }
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};  // 画像の正規化（EdgeTAM。ImageNet の値）
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

    // エンコーダの出力。モデルの出力バッファを指すので、次に encode を呼ぶまで有効
    struct Features {
        ov::Tensor high_res_feats0;  // (1, 32, 128, 128)
        ov::Tensor high_res_feats1;  // (1, 64, 64, 64)
        ov::Tensor image_embed;      // (1, 256, 32, 32)
    };

    // (1,3,S,S) の float 画像テンソル -> 特徴量
    Features encode(const float* image_tensor);

    // 点プロンプトの座標を、元画像の座標 (x, y) から EdgeTAM の入力座標（0〜target_size）へ変換する
    // （EdgeTamHelper.preprocess の transform_coords。float32 で: x / 画像の幅 * target_size）
    void to_model_coords(const std::vector<std::array<double, 2>>& image_coords, int width, int height, float* out) const;

    // デコード。点プロンプト（モデル座標）と初期マスクで推定し、スコア最大のマスクのロジットを best_logit に書く
    // （後処理: ロジットを ±32 に clip。best_logit は mask_size × mask_size の float）。スコア最大のマスクの番号を返す
    int decode_best(const Features& features, const float* point_coords /* N×2 */, const int32_t* point_labels, int num_points,
                    const float* mask_input /* mask_size × mask_size */, float* best_logit);

    int mask_size() const { return mask_size_; }

private:
    void warm_up();

    int target_size_;
    int mask_size_;  // 高解像度特徴量・ロジットの一辺（target_size / edgetam_high_res_feature_downscale_factor）

    ov::CompiledModel encoder_, decoder_;
    ov::InferRequest encoder_req_, decoder_req_;
};

}  // namespace pes
