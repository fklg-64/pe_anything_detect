#include "clip_extractor.h"

#include <cstring>

#include "params.h"
#include "simd.h"

namespace pes {

ClipExtractor::ClipExtractor(ov::Core& core, const PesModelParams& params, std::unique_ptr<ClipTokenizer> tokenizer)
    : target_size_(params.image_size), tokenizer_(std::move(tokenizer)) {
    const std::string dir = params.model_dir;
    const std::string suffix = model_file_suffix(params.pe_use_int8 != 0);
    const std::string device = upper_device_name(params.device);
    const ov::AnyMap config = make_compile_config(params.ov_num_thread);

    try {
        std::shared_ptr<ov::Model> visual_model = core.read_model(model_path(dir, "pe_visual_model" + suffix));
        // 未使用の出力は計算・転送が無駄なので、コンパイル前にグラフから外す（pe_visual_model の出力のうち、使うのは
        // raw_patch_features と fused_layer_features だけ。結果は変わらない）
        const ov::ResultVector results = visual_model->get_results();
        for (const auto& result : results) {
            const auto& names = result->output(0).get_names();
            if (names.count("cls2patch_attention") || names.count("raw_image_features")) visual_model->remove_result(result);
        }
        std::shared_ptr<ov::Model> text_model = core.read_model(model_path(dir, "pe_text_model" + suffix));
        std::shared_ptr<ov::Model> logits_model = core.read_model(model_path(dir, "pe_patch_logits_model" + suffix));

        // メタデータ
        patch_size_ = std::stoi(visual_model->get_rt_info<std::string>("patch_size"));
        feature_size_ = target_size_ / patch_size_;
        if (patch_size_ <= 0 || target_size_ % patch_size_ != 0)
            throw Error(PES_ERROR_INVALID_ARGUMENT, "image_size (" + std::to_string(target_size_) + ") はパッチサイズ (" +
                                                        std::to_string(patch_size_) + ") の倍数にしてください");

        visual_ = core.compile_model(visual_model, device, config);
        text_ = core.compile_model(text_model, device, config);
        patch_logits_ = core.compile_model(logits_model, device, config);
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        throw Error(PES_ERROR_LOAD, std::string("PE モデルの読み込みに失敗しました: ") + e.what());
    }
    visual_req_ = visual_.create_infer_request();
    text_req_ = text_.create_infer_request();
    patch_logits_req_ = patch_logits_.create_infer_request();

    warm_up();
}

// 初期実行時の遅延を回避する（モデルごとに 1 回ずつ実行しておく）
void ClipExtractor::warm_up() {
    const size_t pixels = static_cast<size_t>(3) * target_size_ * target_size_;
    AlignedArray<float> zeros = make_aligned<float>(pixels);
    std::memset(zeros.get(), 0, pixels * sizeof(float));
    ImageFeatures features = extract_image_features(zeros.get());

    const std::vector<float> text = extract_text_features({"warm up"});
    (void)text;

    std::vector<float> labels(static_cast<size_t>(feature_size_) * feature_size_, 0.0f);
    int k = 0, d = 0;
    extract_cluster_tokens(features, labels.data(), static_cast<int>(labels.size()), &k, &d);
}

ClipExtractor::ImageFeatures ClipExtractor::extract_image_features(const float* image_tensor) {
    const ov::Shape shape{1, 3, static_cast<size_t>(target_size_), static_cast<size_t>(target_size_)};
    visual_req_.set_input_tensor(ov::Tensor(ov::element::f32, shape, const_cast<float*>(image_tensor)));
    visual_req_.infer();
    ImageFeatures out;
    out.raw_patch = visual_req_.get_tensor("raw_patch_features");
    const ov::Tensor fused = visual_req_.get_tensor("fused_layer_features");
    out.fused = fused.data<float>();
    out.num_tokens = static_cast<int>(fused.get_shape()[1]);
    out.fused_dim = static_cast<int>(fused.get_shape()[2]);
    return out;
}

std::vector<float> ClipExtractor::extract_text_features(const std::vector<std::string>& prompts) {
    const size_t n = prompts.size();
    if (n == 0) return {};
    constexpr int kLen = params::TEXT_CONTEXT_LENGTH;
    std::vector<int64_t> ids(n * kLen);
    for (size_t i = 0; i < n; ++i) tokenizer_->encode(prompts[i], kLen, ids.data() + i * kLen);

    text_req_.set_input_tensor(ov::Tensor(ov::element::i64, ov::Shape{n, static_cast<size_t>(kLen)}, ids.data()));
    text_req_.infer();
    const ov::Tensor out = text_req_.get_output_tensor(0);
    const ov::Shape shape = out.get_shape();
    text_dim_ = static_cast<int>(shape[1]);
    std::vector<float> features(out.data<float>(), out.data<float>() + n * text_dim_);
    simd::normalize_rows(features.data(), static_cast<int>(n), text_dim_, 0.0f);  // np.linalg.norm で割る（eps なし）
    return features;
}

const float* ClipExtractor::extract_cluster_tokens(const ImageFeatures& features, const float* cluster_labels, int num_labels,
                                                   int* num_clusters, int* dim) {
    patch_logits_req_.set_input_tensor(0, features.raw_patch);
    patch_logits_req_.set_input_tensor(
        1, ov::Tensor(ov::element::f32, ov::Shape{static_cast<size_t>(num_labels)}, const_cast<float*>(cluster_labels)));
    patch_logits_req_.infer();
    const ov::Tensor out = patch_logits_req_.get_output_tensor(0);
    *num_clusters = static_cast<int>(out.get_shape()[0]);
    *dim = static_cast<int>(out.get_shape()[1]);
    return out.data<float>();
}

}  // namespace pes
