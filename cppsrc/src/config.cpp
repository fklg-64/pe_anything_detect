#include "config.h"

#include <cctype>
#include <cstring>

#include "common.h"
#include "yaml_lite.h"

namespace pes {

void copy_to_buffer(char* dst, size_t capacity, const std::string& src, const char* what) {
    if (src.size() + 1 > capacity) throw Error(PES_ERROR_INVALID_ARGUMENT, std::string(what) + " が長すぎます（" + std::to_string(capacity - 1) + " バイトまで）");
    std::memcpy(dst, src.c_str(), src.size() + 1);
}

// 既定値は configs/ov_config.yaml と同じ
void set_default_model_params(PesModelParams* p) {
    std::memset(p, 0, sizeof(*p));
    p->image_size = 512;
    copy_to_buffer(p->model_dir, sizeof(p->model_dir), "./weights/openvino_models_l6", "model_dir");
    p->ov_num_thread = 4;
    copy_to_buffer(p->device, sizeof(p->device), "cpu", "device");
    p->et_use_int8 = 0;
    p->pe_use_int8 = 0;
    p->edgetam_high_res_feature_downscale_factor = 4;
    p->tokenizer_merges_path[0] = '\0';
}

void set_default_prompt_params(PesPromptParams* p) {
    std::memset(p, 0, sizeof(*p));
    copy_to_buffer(p->prompts_dir, sizeof(p->prompts_dir), "./prompts/candidates/color_free_bg", "prompts_dir");
}

void set_default_run_params(PesRunParams* p) {
    std::memset(p, 0, sizeof(*p));
    p->k_clusters = 8;
    p->n_iter = 10;
    p->foreground_threshold = 0.5;
    p->edgetam_foreground_points_per_class = 4;
    p->edgetam_background_point_count = 4;
    p->use_edgetam_clustering = 1;
    p->edgetam_refine_iterations = 2;
    p->semantic_max_side = 1024;
}

void set_default_output_params(PesOutputParams* p) {
    std::memset(p, 0, sizeof(*p));
    copy_to_buffer(p->output_dir, sizeof(p->output_dir), "./outputs/experiment_results", "output_dir");
    p->overlay_alpha = 0.5f;
    p->show_result = 0;
    static const int palette[] = {
        0,   0,   0,   255, 0,   0,   0,   255, 0,   0,   0,   255, 255, 255, 0,   0,   255, 255, 255, 0,   255, 192, 192, 192, 128, 0,   0,
        128, 128, 0,   0,   128, 0,   128, 0,   128, 0,   0,   128, 255, 165, 0,   255, 105, 180, 255, 69,  0,   255, 20,  147, 34,  139, 34,
        0,   255, 127, 255, 182, 193, 255, 215, 0,   139, 69,  19,  244, 164, 96,  72,  61,  139, 0,   191, 255, 70,  130, 180, 147, 112, 219,
        153, 50,  204, 255, 228, 225, 255, 0,   255, 255, 228, 181, 255, 99,  71,  0,   250, 154, 221, 160, 221, 238, 130, 238, 186, 85,  211,
        255, 240, 245, 219, 112, 147, 255, 69,  0,   240, 128, 128, 255, 105, 180};
    p->color_palette_size = static_cast<int>(sizeof(palette) / sizeof(palette[0]));
    std::memcpy(p->color_palette, palette, sizeof(palette));
}

namespace {

// 子ノードがあれば読む。型が合わなければエラー
void read_int(const yaml::Node& root, const char* key, int& out) {
    const yaml::Node* n = root.get(key);
    if (!n || n->is_null()) return;
    bool b;
    if (n->to_int(out)) return;
    if (n->to_bool(b)) { out = b ? 1 : 0; return; }  // true/false を 0/1 として受ける（use_int8 など）
    throw Error(PES_ERROR_LOAD, std::string("設定 ") + key + " は整数で指定してください");
}

void read_float(const yaml::Node& root, const char* key, float& out) {
    const yaml::Node* n = root.get(key);
    if (!n || n->is_null()) return;
    double v;
    if (!n->to_double(v)) throw Error(PES_ERROR_LOAD, std::string("設定 ") + key + " は数値で指定してください");
    out = static_cast<float>(v);
}

void read_double(const yaml::Node& root, const char* key, double& out) {
    const yaml::Node* n = root.get(key);
    if (!n || n->is_null()) return;
    if (!n->to_double(out)) throw Error(PES_ERROR_LOAD, std::string("設定 ") + key + " は数値で指定してください");
}

void read_bool(const yaml::Node& root, const char* key, int& out) {
    const yaml::Node* n = root.get(key);
    if (!n || n->is_null()) return;
    bool b;
    int i;
    if (n->to_bool(b)) out = b ? 1 : 0;
    else if (n->to_int(i)) out = i != 0 ? 1 : 0;
    else throw Error(PES_ERROR_LOAD, std::string("設定 ") + key + " は true / false で指定してください");
}

void read_string(const yaml::Node& root, const char* key, char* dst, size_t capacity) {
    const yaml::Node* n = root.get(key);
    if (!n || n->is_null()) return;
    if (n->type != yaml::Node::Type::Scalar) throw Error(PES_ERROR_LOAD, std::string("設定 ") + key + " は文字列で指定してください");
    copy_to_buffer(dst, capacity, n->scalar, key);
}

std::vector<std::string> string_list(const yaml::Node& root, const char* key, const std::string& path) {
    std::vector<std::string> out;
    const yaml::Node* n = root.get(key);
    if (!n || n->is_null()) return out;
    if (n->type != yaml::Node::Type::List) throw Error(PES_ERROR_LOAD, path + ": " + key + " は配列で指定してください");
    for (const yaml::Node& item : n->list) {
        if (item.type != yaml::Node::Type::Scalar) throw Error(PES_ERROR_LOAD, path + ": " + key + " の要素は文字列で指定してください");
        out.push_back(item.scalar);
    }
    return out;
}

}  // namespace

void load_config_yaml(const std::string& path, PesModelParams* model, PesPromptParams* prompt, PesRunParams* run,
                      PesOutputParams* output) {
    const yaml::Node root = yaml::parse_file(path);
    if (root.type != yaml::Node::Type::Map) throw Error(PES_ERROR_LOAD, path + ": 設定のトップレベルは「キー: 値」の形式にしてください");

    if (model) {
        read_int(root, "image_size", model->image_size);
        read_string(root, "model_dir", model->model_dir, sizeof(model->model_dir));
        read_int(root, "ov_num_thread", model->ov_num_thread);
        read_string(root, "device", model->device, sizeof(model->device));
        read_bool(root, "et_use_int8", model->et_use_int8);
        read_bool(root, "pe_use_int8", model->pe_use_int8);
        read_int(root, "edgetam_high_res_feature_downscale_factor", model->edgetam_high_res_feature_downscale_factor);
    }
    if (prompt) read_string(root, "prompts_dir", prompt->prompts_dir, sizeof(prompt->prompts_dir));
    if (run) {
        read_int(root, "k_clusters", run->k_clusters);
        read_int(root, "n_iter", run->n_iter);
        read_double(root, "foreground_threshold", run->foreground_threshold);
        read_int(root, "edgetam_foreground_points_per_class", run->edgetam_foreground_points_per_class);
        read_int(root, "edgetam_background_point_count", run->edgetam_background_point_count);
        read_bool(root, "use_edgetam_clustering", run->use_edgetam_clustering);
        read_int(root, "edgetam_refine_iterations", run->edgetam_refine_iterations);
        read_int(root, "semantic_max_side", run->semantic_max_side);
    }
    if (output) {
        read_string(root, "output_dir", output->output_dir, sizeof(output->output_dir));
        read_float(root, "overlay_alpha", output->overlay_alpha);
        read_bool(root, "show_result", output->show_result);
        if (const yaml::Node* pal = root.get("color_palette")) {
            if (pal->type != yaml::Node::Type::List) throw Error(PES_ERROR_LOAD, "設定 color_palette は配列で指定してください");
            if (pal->list.size() > PES_PALETTE_MAX) throw Error(PES_ERROR_LOAD, "color_palette が長すぎます");
            int n = 0;
            for (const yaml::Node& item : pal->list) {
                int v;
                if (!item.to_int(v)) throw Error(PES_ERROR_LOAD, "color_palette の要素は整数で指定してください");
                output->color_palette[n++] = v;
            }
            output->color_palette_size = n;
        }
    }
}

std::vector<std::string> load_color_shape_prompts(const std::string& path) {
    const yaml::Node root = yaml::parse_file(path);
    if (root.type != yaml::Node::Type::Map) throw Error(PES_ERROR_LOAD, path + ": プロンプトの yaml は { \"custom\": [...], ... } の形式にしてください");
    std::vector<std::string> prompts;
    const std::vector<std::string> colors = string_list(root, "color", path);
    const std::vector<std::string> shapes = string_list(root, "shape", path);
    for (const std::string& color : colors) {
        for (const std::string& shape : shapes) {
            const char first = color.empty() ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(color[0])));
            const bool vowel = first == 'a' || first == 'e' || first == 'i' || first == 'o' || first == 'u';
            prompts.push_back(std::string("a photo of ") + (vowel ? "an " : "a ") + color + " " + shape + " object.");
        }
    }
    for (std::string& s : string_list(root, "custom", path)) prompts.push_back(std::move(s));
    return prompts;
}

std::vector<std::string> load_plain_prompts(const std::string& path) {
    const yaml::Node root = yaml::parse_file(path);
    if (root.type != yaml::Node::Type::Map) throw Error(PES_ERROR_LOAD, path + ": プロンプトの yaml は { \"base\": [...], ... } の形式にしてください");
    std::vector<std::string> prompts = string_list(root, "base", path);
    for (std::string& s : string_list(root, "custom", path)) prompts.push_back(std::move(s));
    return prompts;
}

}  // namespace pes
