// OpenVINO backend（PE / EdgeTAM）に共通の準備
#pragma once

#include <openvino/openvino.hpp>
#include <string>

#include "pe_segmentation.h"

namespace pes {

// モデルファイル名の末尾。INT8 量子化モデルは "_int8.xml"、そうでなければ ".xml"
inline const char* model_file_suffix(bool use_int8) { return use_int8 ? "_int8.xml" : ".xml"; }

// デバイス名は大文字にする（"cpu" -> "CPU"）
std::string upper_device_name(const std::string& device);

// compile_model に渡す実行時の設定。低遅延（1枚ずつ処理）向け
ov::AnyMap make_compile_config(int num_threads);

// ディレクトリとファイル名を連結し、存在を確認する（無ければ PES_ERROR_LOAD）
std::string model_path(const std::string& dir, const std::string& name);

// ov::Shape の表示用
std::string shape_to_string(const ov::Shape& shape);

}  // namespace pes
