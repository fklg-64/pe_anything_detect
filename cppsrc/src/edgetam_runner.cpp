#include "edgetam_runner.h"

#include <array>
#include <cstring>

#include "params.h"
#include "simd.h"

namespace pes {

EdgeTamRunner::EdgeTamRunner(ov::Core& core, const PesModelParams& params)
    : target_size_(params.image_size), mask_size_(params.image_size / params.edgetam_high_res_feature_downscale_factor) {
    if (params.edgetam_high_res_feature_downscale_factor <= 0)
        throw Error(PES_ERROR_INVALID_ARGUMENT, "edgetam_high_res_feature_downscale_factor は 1 以上にしてください");
    const std::string dir = params.model_dir;
    const std::string suffix = model_file_suffix(params.et_use_int8 != 0);
    const std::string device = upper_device_name(params.device);
    const ov::AnyMap config = make_compile_config(params.ov_num_thread);
    try {
        auto encoder_model = core.read_model(model_path(dir, "edgetam_encoder" + suffix));
        auto decoder_model = core.read_model(model_path(dir, "edgetam_decoder" + suffix));
        encoder_ = core.compile_model(encoder_model, device, config);
        decoder_ = core.compile_model(decoder_model, device, config);
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        throw Error(PES_ERROR_LOAD, std::string("EdgeTAM モデルの読み込みに失敗しました: ") + e.what());
    }
    encoder_req_ = encoder_.create_infer_request();
    decoder_req_ = decoder_.create_infer_request();
    warm_up();
}

// ダミー入力でエンコーダとデコーダを 1 回ずつ実行する（初回実行の遅延を起動時に済ませる）
void EdgeTamRunner::warm_up() {
    const size_t pixels = static_cast<size_t>(3) * target_size_ * target_size_;
    AlignedArray<float> zeros = make_aligned<float>(pixels);
    std::memset(zeros.get(), 0, pixels * sizeof(float));
    const Features features = encode(zeros.get());

    const float coords[2] = {0.0f, 0.0f};
    const int32_t labels[1] = {1};
    std::vector<float> mask(static_cast<size_t>(mask_size_) * mask_size_, 0.0f), out(mask.size());
    decode_best(features, coords, labels, 1, mask.data(), out.data());
}

EdgeTamRunner::Features EdgeTamRunner::encode(const float* image_tensor) {
    const ov::Shape shape{1, 3, static_cast<size_t>(target_size_), static_cast<size_t>(target_size_)};
    encoder_req_.set_input_tensor(ov::Tensor(ov::element::f32, shape, const_cast<float*>(image_tensor)));
    encoder_req_.infer();
    Features f;
    f.high_res_feats0 = encoder_req_.get_output_tensor(0);
    f.high_res_feats1 = encoder_req_.get_output_tensor(1);
    f.image_embed = encoder_req_.get_output_tensor(2);
    const ov::Shape s0 = f.high_res_feats0.get_shape();
    if (static_cast<int>(s0[2]) != mask_size_ || static_cast<int>(s0[3]) != mask_size_)
        throw Error(PES_ERROR_RUNTIME, "EdgeTAM の高解像度特徴量の大きさ " + shape_to_string(s0) + " が edgetam_high_res_feature_downscale_factor と合いません（期待: " +
                                           std::to_string(mask_size_) + "×" + std::to_string(mask_size_) + "）");
    return f;
}

void EdgeTamRunner::to_model_coords(const std::vector<std::array<double, 2>>& image_coords, int width, int height, float* out) const {
    // Python 版は float32 に直してから x / 幅 を計算し、target_size を掛ける
    const float w = static_cast<float>(width), h = static_cast<float>(height), size = static_cast<float>(target_size_);
    for (size_t i = 0; i < image_coords.size(); ++i) {
        const float x = static_cast<float>(image_coords[i][0]);
        const float y = static_cast<float>(image_coords[i][1]);
        out[i * 2] = (x / w) * size;
        out[i * 2 + 1] = (y / h) * size;
    }
}

int EdgeTamRunner::decode_best(const Features& features, const float* point_coords, const int32_t* point_labels, int num_points,
                               const float* mask_input, float* best_logit) {
    const size_t n = static_cast<size_t>(num_points);
    const size_t m = static_cast<size_t>(mask_size_);
    float has_mask = 1.0f;

    decoder_req_.set_input_tensor(0, features.high_res_feats0);
    decoder_req_.set_input_tensor(1, features.high_res_feats1);
    decoder_req_.set_input_tensor(2, features.image_embed);
    decoder_req_.set_input_tensor(3, ov::Tensor(ov::element::f32, ov::Shape{1, n, 2}, const_cast<float*>(point_coords)));
    decoder_req_.set_input_tensor(4, ov::Tensor(ov::element::i32, ov::Shape{1, n}, const_cast<int32_t*>(point_labels)));
    decoder_req_.set_input_tensor(5, ov::Tensor(ov::element::f32, ov::Shape{1, 1, m, m}, const_cast<float*>(mask_input)));
    decoder_req_.set_input_tensor(6, ov::Tensor(ov::element::f32, ov::Shape{1}, &has_mask));
    decoder_req_.infer();

    const ov::Tensor masks = decoder_req_.get_output_tensor(0);   // (1, K, h, w)
    const ov::Tensor scores = decoder_req_.get_output_tensor(1);  // (1, K)
    const ov::Shape mshape = masks.get_shape();
    const size_t num_masks = mshape[1];
    if (num_masks == 0) return -1;
    if (mshape[2] != m || mshape[3] != m)
        throw Error(PES_ERROR_RUNTIME, "EdgeTAM デコーダの出力マスクの大きさ " + shape_to_string(mshape) + " が想定と違います");

    // スコア最大のマスク（np.argmax と同じく、同値なら先頭）
    const float* s = scores.data<float>();
    size_t best = 0;
    for (size_t i = 1; i < num_masks; ++i)
        if (s[i] > s[best]) best = i;

    // 後処理: ロジットを ±32 に clip する
    simd::clamp(masks.data<float>() + best * m * m, best_logit, m * m, -params::MASK_LOGIT_CLIP, params::MASK_LOGIT_CLIP);
    return static_cast<int>(best);
}

}  // namespace pes
