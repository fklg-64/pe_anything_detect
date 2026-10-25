// 設定（yaml）とプロンプトの読み込み
#pragma once

#include <string>
#include <vector>

#include "pe_segmentation.h"

namespace pes {

void set_default_model_params(PesModelParams* p);
void set_default_prompt_params(PesPromptParams* p);
void set_default_run_params(PesRunParams* p);
void set_default_output_params(PesOutputParams* p);

// configs/ov_config.yaml 形式を読み、yaml にあるキーだけ上書きする（NULL の引数は無視）
void load_config_yaml(const std::string& path, PesModelParams* model, PesPromptParams* prompt, PesRunParams* run,
                      PesOutputParams* output);

// 前景プロンプト: color × shape の全組み合わせから "a photo of a <color> <shape> object." を作り、custom を足す
std::vector<std::string> load_color_shape_prompts(const std::string& path);
// 背景プロンプト: base と custom をそのまま連結する
std::vector<std::string> load_plain_prompts(const std::string& path);

// 固定長 char 配列への安全なコピー（収まらなければ Error）
void copy_to_buffer(char* dst, size_t capacity, const std::string& src, const char* what);

}  // namespace pes
