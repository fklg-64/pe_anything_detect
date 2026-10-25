// C API（extern "C"）の実装。例外は境界で捕まえて PesStatus に変換する
#include <cstdlib>
#include <cstring>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "common.h"
#include "config.h"
#include "segmenter.h"
#include "simd.h"

struct PesContext {
    std::unique_ptr<pes::Segmenter> segmenter;
    std::string last_error;
};

namespace {

thread_local std::string g_last_error;  // pes_create など、コンテキストが無いときのエラー

// ctx が NULL でも使える。例外を戻り値とエラーメッセージに変換して実行する
template <typename F>
int guarded(PesContext* ctx, F&& f) {
    try {
        f();
        if (ctx) ctx->last_error.clear();
        return PES_OK;
    } catch (const pes::Error& e) {
        (ctx ? ctx->last_error : g_last_error) = e.what();
        return e.status();
    } catch (const std::bad_alloc&) {
        (ctx ? ctx->last_error : g_last_error) = "メモリを確保できません";
        return PES_ERROR_RUNTIME;
    } catch (const std::exception& e) {
        (ctx ? ctx->last_error : g_last_error) = e.what();
        return PES_ERROR_RUNTIME;
    } catch (...) {
        (ctx ? ctx->last_error : g_last_error) = "不明なエラー";
        return PES_ERROR_RUNTIME;
    }
}

void require(const void* ptr, const char* name) {
    if (!ptr) throw pes::Error(PES_ERROR_INVALID_ARGUMENT, std::string(name) + " が NULL です");
}

void check_kind(PesPromptKind kind) {
    if (kind != PES_PROMPT_FOREGROUND && kind != PES_PROMPT_BACKGROUND) throw pes::Error(PES_ERROR_INVALID_ARGUMENT, "プロンプトの種類が不正です");
}

}  // namespace

extern "C" {

const char* pes_version(void) { return "1.0.0"; }

int pes_cpu_has_avx2(void) { return pes::simd::cpu_has_avx2() ? 1 : 0; }

void pes_default_model_params(PesModelParams* p) { if (p) pes::set_default_model_params(p); }
void pes_default_prompt_params(PesPromptParams* p) { if (p) pes::set_default_prompt_params(p); }
void pes_default_run_params(PesRunParams* p) { if (p) pes::set_default_run_params(p); }
void pes_default_output_params(PesOutputParams* p) { if (p) pes::set_default_output_params(p); }

int pes_load_config_yaml(const char* path, PesModelParams* model, PesPromptParams* prompt, PesRunParams* run, PesOutputParams* output) {
    return guarded(nullptr, [&] {
        require(path, "path");
        pes::load_config_yaml(path, model, prompt, run, output);
    });
}

int pes_create(const PesModelParams* params, PesContext** out_ctx) {
    if (out_ctx) *out_ctx = nullptr;
    return guarded(nullptr, [&] {
        require(params, "params");
        require(out_ctx, "out_ctx");
        auto ctx = std::make_unique<PesContext>();
        ctx->segmenter = std::make_unique<pes::Segmenter>(*params);
        *out_ctx = ctx.release();
    });
}

void pes_destroy(PesContext* ctx) { delete ctx; }

const char* pes_last_error(const PesContext* ctx) { return ctx ? ctx->last_error.c_str() : g_last_error.c_str(); }

void pes_set_verbose(PesContext* ctx, int verbose) {
    if (ctx && ctx->segmenter) ctx->segmenter->set_verbose(verbose != 0);
}

int pes_extract_text_features(PesContext* ctx, const PesPromptParams* params) {
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        PesPromptParams defaults;
        if (!params) {
            pes::set_default_prompt_params(&defaults);
            params = &defaults;
        }
        ctx->segmenter->prompts().load_prompts(params->prompts_dir);
    });
}

int pes_add_prompts(PesContext* ctx, PesPromptKind kind, const char* const* texts, int num_texts, int* out_num_added) {
    if (out_num_added) *out_num_added = 0;
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        check_kind(kind);
        if (num_texts < 0 || (num_texts > 0 && !texts)) throw pes::Error(PES_ERROR_INVALID_ARGUMENT, "texts が不正です");
        std::vector<std::string> list;
        for (int i = 0; i < num_texts; ++i) {
            require(texts[i], "texts[i]");
            list.emplace_back(texts[i]);
        }
        const int added = ctx->segmenter->prompts().add_prompts(kind, list);
        if (out_num_added) *out_num_added = added;
    });
}

int pes_add_prompt(PesContext* ctx, PesPromptKind kind, const char* text) {
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        require(text, "text");
        check_kind(kind);
        if (ctx->segmenter->prompts().find_prompt(kind, text) >= 0) throw pes::Error(PES_ERROR_ALREADY_EXISTS, std::string("同じプロンプトが既にあります: ") + text);
        ctx->segmenter->prompts().add_prompts(kind, {text});
    });
}

int pes_remove_prompt(PesContext* ctx, PesPromptKind kind, const char* text) {
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        require(text, "text");
        check_kind(kind);
        const int index = ctx->segmenter->prompts().find_prompt(kind, text);
        if (index < 0) throw pes::Error(PES_ERROR_NOT_FOUND, std::string("プロンプトがありません: ") + text);
        ctx->segmenter->prompts().remove_prompt_at(kind, index);
    });
}

int pes_remove_prompt_at(PesContext* ctx, PesPromptKind kind, int index) {
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        check_kind(kind);
        ctx->segmenter->prompts().remove_prompt_at(kind, index);
    });
}

int pes_clear_prompts(PesContext* ctx, PesPromptKind kind) {
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        check_kind(kind);
        ctx->segmenter->prompts().clear_prompts(kind);
    });
}

int pes_prompt_count(const PesContext* ctx, PesPromptKind kind) {
    if (!ctx || (kind != PES_PROMPT_FOREGROUND && kind != PES_PROMPT_BACKGROUND)) return -1;
    return static_cast<int>(ctx->segmenter->prompts().prompts(kind).size());
}

const char* pes_prompt_text(const PesContext* ctx, PesPromptKind kind, int index) {
    if (!ctx || (kind != PES_PROMPT_FOREGROUND && kind != PES_PROMPT_BACKGROUND)) return nullptr;
    const auto& list = ctx->segmenter->prompts().prompts(kind);
    if (index < 0 || index >= static_cast<int>(list.size())) return nullptr;
    return list[index].c_str();
}

int pes_run(PesContext* ctx, const PesRunParams* params, const PesImage* image, PesResult* result) {
    if (result) std::memset(result, 0, sizeof(*result));
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        require(image, "image");
        require(result, "result");
        PesRunParams defaults;
        if (!params) {
            pes::set_default_run_params(&defaults);
            params = &defaults;
        }
        ctx->segmenter->segment(*image, *params, result);
    });
}

int pes_run_file(PesContext* ctx, const PesRunParams* params, const char* image_path, PesResult* result) {
    if (result) std::memset(result, 0, sizeof(*result));
    return guarded(ctx, [&] {
        require(ctx, "ctx");
        require(image_path, "image_path");
        require(result, "result");
        const cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);  // 3 チャンネル BGR（グレースケール・アルファ付きも 3 チャンネルになる）
        if (bgr.empty()) throw pes::Error(PES_ERROR_LOAD, std::string("画像を読み込めません: ") + image_path);
        PesImage image{bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step), PES_PIXEL_BGR};
        PesRunParams defaults;
        if (!params) {
            pes::set_default_run_params(&defaults);
            params = &defaults;
        }
        ctx->segmenter->segment(image, *params, result);
    });
}

void pes_result_free(PesResult* r) {
    if (!r) return;
    if (r->class_prompts) {
        for (int i = 0; i < r->num_classes; ++i) std::free(r->class_prompts[i]);
    }
    std::free(r->semantic_map);
    std::free(r->prompt_indices);
    std::free(r->class_prompts);
    std::free(r->point_coords);
    std::free(r->point_labels);
    std::free(r->foreground_map);
    std::free(r->foreground_mask);
    std::free(r->cluster_labels);
    std::free(r->cluster_prompt_index);
    std::free(r->raw_semantic_labels);
    std::free(r->semantic_labels);
    std::memset(r, 0, sizeof(*r));
}

}  // extern "C"
