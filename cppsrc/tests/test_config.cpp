// 設定 yaml・プロンプト yaml の読み込みの確認（リポジトリの configs/ov_config.yaml と prompts/ を実際に読む）
#include <cstdio>
#include <cstring>
#include <string>

#include "common.h"
#include "config.h"

#ifndef PES_SOURCE_DIR
#define PES_SOURCE_DIR "."
#endif

static int g_failed = 0;
#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            ++g_failed;                                           \
            std::printf("  FAIL %s:%d  ", __FILE__, __LINE__);    \
            std::printf(__VA_ARGS__);                             \
            std::printf("\n");                                    \
        }                                                         \
    } while (0)

int main() {
    const std::string root = std::string(PES_SOURCE_DIR) + "/..";
    try {
        // 設定: ov_config.yaml を読んだ結果が、組み込みの既定値と一致する（既定値は ov_config.yaml と同じにしてある）
        PesModelParams m, m0;
        PesPromptParams p, p0;
        PesRunParams r, r0;
        PesOutputParams o, o0;
        pes::set_default_model_params(&m0);
        pes::set_default_prompt_params(&p0);
        pes::set_default_run_params(&r0);
        pes::set_default_output_params(&o0);
        // 既定値と違う値で埋めてから読む（上書きされることの確認）
        pes::set_default_model_params(&m);
        pes::set_default_prompt_params(&p);
        pes::set_default_run_params(&r);
        pes::set_default_output_params(&o);
        m.image_size = 1;
        m.ov_num_thread = 99;
        r.k_clusters = 99;
        r.foreground_threshold = 0.1;
        o.color_palette_size = 0;
        std::snprintf(m.model_dir, sizeof(m.model_dir), "x");
        pes::load_config_yaml(root + "/configs/ov_config.yaml", &m, &p, &r, &o);

        CHECK(m.image_size == 512, "image_size=%d", m.image_size);
        CHECK(std::string(m.model_dir) == m0.model_dir, "model_dir=%s", m.model_dir);
        CHECK(m.ov_num_thread == 4, "ov_num_thread=%d", m.ov_num_thread);
        CHECK(std::string(m.device) == "cpu", "device=%s", m.device);
        CHECK(m.et_use_int8 == 0 && m.pe_use_int8 == 0, "int8");
        CHECK(m.edgetam_high_res_feature_downscale_factor == 4, "downscale");
        CHECK(std::string(p.prompts_dir) == p0.prompts_dir, "prompts_dir=%s", p.prompts_dir);
        CHECK(r.k_clusters == r0.k_clusters && r.n_iter == r0.n_iter && r.foreground_threshold == r0.foreground_threshold, "run params");
        CHECK(r.edgetam_foreground_points_per_class == 4 && r.edgetam_background_point_count == 4, "points");
        CHECK(r.use_edgetam_clustering == 1 && r.edgetam_refine_iterations == 2 && r.semantic_max_side == 1024, "run params 2");
        CHECK(std::string(o.output_dir) == o0.output_dir && o.overlay_alpha == 0.5f && o.show_result == 0, "output params");
        CHECK(o.color_palette_size == o0.color_palette_size, "palette size %d vs %d", o.color_palette_size, o0.color_palette_size);
        bool same = true;
        for (int i = 0; i < o.color_palette_size; ++i) same = same && o.color_palette[i] == o0.color_palette[i];
        CHECK(same, "palette 内容");

        // プロンプト: コメント・末尾のカンマつきの flow 形式
        const std::string dir = root + "/prompts/candidates/color_free_bg";
        const auto fg = pes::load_color_shape_prompts(dir + "/foreground.yaml");
        const auto bg = pes::load_plain_prompts(dir + "/background.yaml");
        CHECK(fg.size() == 29, "前景 %zu 個", fg.size());
        CHECK(bg.size() == 23, "背景 %zu 個", bg.size());
        CHECK(!fg.empty() && fg.front() == "a photo of a screw." && fg.back() == "a photo of a animal.", "前景の先頭・末尾");
        CHECK(!bg.empty() && bg.front() == "a photo of an empty industrial workbench.", "背景の先頭");
    } catch (const std::exception& e) {
        std::printf("エラー: %s\n", e.what());
        return 2;
    }
    std::printf(g_failed ? "test_config: %d 件失敗\n" : "test_config: すべて成功\n", g_failed);
    return g_failed ? 1 : 0;
}
