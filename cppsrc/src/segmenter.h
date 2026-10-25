// CLIP による粗い推定 -> EdgeTAM による精緻化 の全体の流れ（画像1枚分）をつなぐ（core/common/segmentation_runner.py）。
// モデル・プロンプト・テキスト特徴量を持つ。C API の PesContext の実体。
#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "clip_extractor.h"
#include "coarse_segmentation.h"
#include "edgetam_runner.h"
#include "pe_segmentation.h"
#include "semantic_refiner.h"

namespace pes {

class Segmenter {
public:
    // モデルを読み込む（PE と EdgeTAM。ウォームアップまで行う）
    explicit Segmenter(const PesModelParams& params);

    void set_verbose(bool verbose);
    bool verbose() const { return verbose_; }

    CoarseSegmenter& prompts() { return coarse_; }
    const CoarseSegmenter& prompts() const { return coarse_; }

    // 1枚の画像をセグメンテーションする。*result は未初期化でよい（確保した配列は pes_result_free で解放する）
    void segment(const PesImage& image, const PesRunParams& run, PesResult* result);

private:
    PesModelParams params_;
    bool verbose_ = false;

    ov::Core core_;
    ClipExtractor clip_;
    EdgeTamRunner edgetam_;
    CoarseSegmenter coarse_;
    SemanticRefiner refiner_;

    AlignedArray<float> clip_tensor_, edgetam_tensor_;  // 正規化済みの入力 (1, 3, S, S)
};

}  // namespace pes
