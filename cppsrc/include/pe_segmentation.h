/*
 * pe_segmentation.h
 *
 * PE-Spatial CLIP + EdgeTAM によるセマンティックセグメンテーション（OpenVINO / C++ 版）の公開 API。
 * Python 版 core/common + core/openvino の C++ への書き起こしで、共有ライブラリ（libpe_segmentation.so）または
 * 静的ライブラリ（libpe_segmentation.a）として、別のプログラムから extern "C" の関数呼び出しで使う。
 *
 * 基本の使い方（呼び出しは 3 つ）:
 *
 *   PesModelParams mp;  pes_default_model_params(&mp);
 *   PesContext* ctx = NULL;
 *   pes_create(&mp, &ctx);                       // 1. モデル読み込み
 *
 *   PesPromptParams pp;  pes_default_prompt_params(&pp);
 *   pes_extract_text_features(ctx, &pp);         // 2. テキスト特徴量抽出（prompts_dir の yaml を読む）
 *   pes_add_prompt(ctx, PES_PROMPT_FOREGROUND, "a photo of a spoon.");   // 追加・削除のたびに特徴量を更新
 *
 *   PesRunParams rp;  pes_default_run_params(&rp);
 *   PesResult result;
 *   pes_run_file(ctx, &rp, "image.png", &result); // 3. 処理実行
 *   ... result.semantic_map ...
 *   pes_result_free(&result);
 *   pes_destroy(ctx);
 *
 * パラメタの名前と既定値は configs/ov_config.yaml と同じ。
 * スレッド安全性: 1 つの PesContext を複数スレッドから同時に使ってはいけない（別々の PesContext なら可）。
 */
#ifndef PE_SEGMENTATION_H
#define PE_SEGMENTATION_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  ifdef PES_BUILDING_SHARED
#    define PES_API __declspec(dllexport)
#  else
#    define PES_API
#  endif
#else
#  define PES_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PES_PATH_MAX 1024
#define PES_DEVICE_MAX 32
#define PES_PALETTE_MAX 256  /* color_palette の要素数（R,G,B を並べた数）の上限 */

/* ------------------------------------------------------------------------- */
/* 戻り値                                                                      */
/* ------------------------------------------------------------------------- */
typedef enum PesStatus {
    PES_OK = 0,
    PES_ERROR_INVALID_ARGUMENT = -1,  /* 引数が不正（NULL・範囲外など） */
    PES_ERROR_LOAD = -2,              /* モデル・設定・画像ファイルの読み込みに失敗 */
    PES_ERROR_NO_PROMPTS = -3,        /* 前景または背景のプロンプトが無い（テキスト特徴量が未抽出） */
    PES_ERROR_ALREADY_EXISTS = -4,    /* 同じプロンプトが既にある */
    PES_ERROR_NOT_FOUND = -5,         /* 指定したプロンプトが無い */
    PES_ERROR_RUNTIME = -6            /* 推論などの実行中のエラー（pes_last_error に詳細） */
} PesStatus;

typedef enum PesPromptKind {
    PES_PROMPT_FOREGROUND = 0,
    PES_PROMPT_BACKGROUND = 1
} PesPromptKind;

typedef enum PesPixelFormat {
    PES_PIXEL_RGB = 0,
    PES_PIXEL_BGR = 1  /* OpenCV の cv::Mat (CV_8UC3) の並び */
} PesPixelFormat;

/* ------------------------------------------------------------------------- */
/* パラメタ                                                                     */
/* ------------------------------------------------------------------------- */

/* モデル読み込み（pes_create）のパラメタ */
typedef struct PesModelParams {
    int  image_size;                 /* 処理画像サイズ（CLIP / EdgeTAM に入れる正方形の一辺）。既定 512 */
    char model_dir[PES_PATH_MAX];    /* OpenVINO モデルのディレクトリ。既定 "./weights/openvino_models_l6" */
    int  ov_num_thread;              /* OpenVINO の推論スレッド数。既定 4 */
    char device[PES_DEVICE_MAX];     /* 推論デバイス。既定 "cpu" */
    int  et_use_int8;                /* EdgeTAM に INT8 モデルを使うか。既定 0 */
    int  pe_use_int8;                /* PE (CLIP) に INT8 モデルを使うか。既定 0 */
    int  edgetam_high_res_feature_downscale_factor;  /* EdgeTAM の高解像度特徴量の縮小倍率（縦横それぞれ 1/N）。既定 4 */
    /* 空なら本ライブラリに埋め込み済みの CLIP BPE（merges）を使う。別の merges.txt を使うときだけ指定する */
    char tokenizer_merges_path[PES_PATH_MAX];
} PesModelParams;

/* テキスト特徴量抽出（pes_extract_text_features）のパラメタ */
typedef struct PesPromptParams {
    char prompts_dir[PES_PATH_MAX];  /* foreground.yaml / background.yaml のあるディレクトリ。既定 "./prompts/candidates/color_free_bg" */
} PesPromptParams;

/* 処理実行（pes_run / pes_run_file）のパラメタ */
typedef struct PesRunParams {
    int   k_clusters;                          /* KMeans のクラスタ数。既定 8 */
    int   n_iter;                              /* KMeans のループ回数。既定 10 */
    double foreground_threshold;               /* 前景と判断する確信度。既定 0.5 */
    int   edgetam_foreground_points_per_class; /* EdgeTAM に入れる前景点数（クラスごと）。既定 4 */
    int   edgetam_background_point_count;      /* EdgeTAM に入れる背景点数。既定 4 */
    int   use_edgetam_clustering;              /* EdgeTAM の特徴で前景確信度を伝播して揃えるか。既定 1 */
    int   edgetam_refine_iterations;           /* EdgeTAM でマスクを反復精緻化する回数。既定 2 */
    int   semantic_max_side;                   /* セマンティックマップのマスク演算を行う作業解像度の長辺上限。既定 1024 */
} PesRunParams;

/* 出力・可視化用のパラメタ（ライブラリの処理には使わない。サンプルや呼び出し側の保存処理向け） */
typedef struct PesOutputParams {
    char  output_dir[PES_PATH_MAX];            /* 結果の出力先。既定 "./outputs/experiment_results" */
    float overlay_alpha;                       /* 元画像に重畳するセマンティックカラーの不透明度。既定 0.5 */
    int   show_result;                         /* 結果をウィンドウで表示するか。既定 0 */
    int   color_palette[PES_PALETTE_MAX];      /* カラーパレット（R,G,B の並び） */
    int   color_palette_size;                  /* color_palette の有効な要素数 */
} PesOutputParams;

/* 既定値（configs/ov_config.yaml と同じ）で埋める */
PES_API void pes_default_model_params(PesModelParams* params);
PES_API void pes_default_prompt_params(PesPromptParams* params);
PES_API void pes_default_run_params(PesRunParams* params);
PES_API void pes_default_output_params(PesOutputParams* params);

/*
 * configs/ov_config.yaml 形式の yaml を読み、該当するパラメタを上書きする（yaml に無いキーは変更しない。NULL の引数は無視）。
 * 先に pes_default_*_params で既定値を入れておくこと。
 */
PES_API int pes_load_config_yaml(const char* path, PesModelParams* model, PesPromptParams* prompt,
                                 PesRunParams* run, PesOutputParams* output);

/* ------------------------------------------------------------------------- */
/* コンテキスト（モデル + プロンプト + テキスト特徴量）                                   */
/* ------------------------------------------------------------------------- */
typedef struct PesContext PesContext;

/* ライブラリのバージョン文字列 */
PES_API const char* pes_version(void);

/* 実行中の CPU が AVX2 + FMA に対応しているか（1=対応。非対応のときは自動でスカラー実装に切り替わる） */
PES_API int pes_cpu_has_avx2(void);

/*
 * 1. モデル読み込み。PE（画像・テキスト・クラスタトークン）と EdgeTAM（エンコーダ・デコーダ）を読み込み、ウォームアップする。
 * 成功すると *out_ctx に新しいコンテキストを返す（pes_destroy で解放）。
 */
PES_API int pes_create(const PesModelParams* params, PesContext** out_ctx);
PES_API void pes_destroy(PesContext* ctx);

/* 直近のエラーメッセージ（UTF-8）。ctx が NULL のときは pes_create 等の、コンテキストを返す前のエラー（スレッドごと） */
PES_API const char* pes_last_error(const PesContext* ctx);

/* 1 の標準出力に各段階の処理時間を出す（Python 版の Timer 相当）。既定は 0 */
PES_API void pes_set_verbose(PesContext* ctx, int verbose);

/* ------------------------------------------------------------------------- */
/* 2. テキスト特徴量とプロンプト                                                       */
/* ------------------------------------------------------------------------- */

/*
 * prompts_dir の foreground.yaml / background.yaml を読み、前景・背景のプロンプトを置き換えて、テキスト特徴量を抽出する。
 * （前景は color × shape の組み合わせ + custom、背景は base + custom。Python 版 CoarseSegmenter と同じ。）
 */
PES_API int pes_extract_text_features(PesContext* ctx, const PesPromptParams* params);

/*
 * プロンプトの追加。追加したプロンプトのテキスト特徴量をその場で抽出する（既存の特徴量はそのまま）。
 * 同じ文のプロンプトが既にあれば PES_ERROR_ALREADY_EXISTS。
 */
PES_API int pes_add_prompt(PesContext* ctx, PesPromptKind kind, const char* text);
/* 複数まとめて追加（特徴量の抽出は 1 回の推論）。既にあるものは飛ばす。*out_num_added（NULL 可）に追加した数を返す */
PES_API int pes_add_prompts(PesContext* ctx, PesPromptKind kind, const char* const* texts, int num_texts,
                            int* out_num_added);

/* プロンプトの削除（特徴量も同時に取り除く）。文で指定するか、番号（0 始まり）で指定する。無ければ PES_ERROR_NOT_FOUND */
PES_API int pes_remove_prompt(PesContext* ctx, PesPromptKind kind, const char* text);
PES_API int pes_remove_prompt_at(PesContext* ctx, PesPromptKind kind, int index);
PES_API int pes_clear_prompts(PesContext* ctx, PesPromptKind kind);

/* 現在のプロンプト。pes_prompt_text が返す文字列は、次にプロンプトを変更するまで有効 */
PES_API int pes_prompt_count(const PesContext* ctx, PesPromptKind kind);
PES_API const char* pes_prompt_text(const PesContext* ctx, PesPromptKind kind, int index);

/* ------------------------------------------------------------------------- */
/* 3. 処理実行                                                                  */
/* ------------------------------------------------------------------------- */

/* 入力画像（8bit 3 チャンネルのインターリーブ） */
typedef struct PesImage {
    const uint8_t* data;
    int width;
    int height;
    int stride;       /* 1 行のバイト数。0 なら width * 3 */
    int pixel_format; /* PesPixelFormat */
} PesImage;

/* 各段階の処理時間 [ms]（Python 版のログと同じ区切り） */
typedef struct PesTimings {
    double preprocess_ms;           /* 画像前処理（リサイズ・正規化） */
    double edgetam_encode_ms;       /* EdgeTAM エンコーダ（use_edgetam_clustering=1 のとき粗い判定の前に実行） */
    double clip_features_ms;        /* PE の画像特徴量 */
    double coarse_foreground_ms;    /* 粗い前景マスク取得（下の 3 つを含む） */
    double kmeans_ms;               /*   KMeans */
    double cluster_tokens_ms;       /*   クラスタごとの CLS トークン抽出（推論）と照合 */
    double propagate_ms;            /*   EdgeTAM 特徴での確信度の伝播 */
    double semantic_labels_ms;      /* セマンティックラベル付与 */
    double coarse_refine_ms;        /* 粗いセグメンテーションの整形 */
    double refine_ms;               /* EdgeTAM による詳細セグメンテーション */
    double total_ms;                /* 合計 */
} PesTimings;

/*
 * 処理結果。pes_run が確保し、pes_result_free で解放する。
 * パッチ格子の配列は feature_size × feature_size（行優先）。
 */
typedef struct PesResult {
    /* 最終結果 */
    int width;                      /* 入力画像と同じ大きさ */
    int height;
    uint8_t* semantic_map;          /* height × width。0=背景、1〜num_classes=前景クラス */
    int num_classes;                /* 前景クラスの数（背景を除く） */
    int* prompt_indices;            /* [num_classes] クラス 1〜に対応する前景プロンプトの番号（pes_prompt_text で文を引ける） */
    char** class_prompts;           /* [num_classes] 同じ前景プロンプトの文（結果が保持する複製。プロンプトを変更しても変わらない） */
    int num_points;                 /* EdgeTAM に与えたプロンプト点の数（全クラス分） */
    float* point_coords;            /* [num_points × 2] 画像座標 (x, y) */
    int* point_labels;              /* [num_points] 1=正例、0=負例 */

    /* 粗い推定の中間結果 */
    int feature_size;
    float* foreground_map;          /* 前景らしさ */
    uint8_t* foreground_mask;       /* 整形後の粗い前景マスク（0/1） */
    int* cluster_labels;            /* クラスタ番号。use_edgetam_clustering=1 のときは PE のクラスタ × 伝播後のプロンプトの組 */
    int num_clusters;
    int* cluster_prompt_index;      /* [num_clusters] クラスタごとの対応プロンプト番号（前景プロンプト → 背景プロンプトの通し番号） */
    int* raw_semantic_labels;       /* 整形前の粗いセマンティックラベル */
    int* semantic_labels;           /* 整形後の粗いセマンティックラベル */

    PesTimings timings;
} PesResult;

/* 画像（メモリ上）を処理する。*result は未初期化でよい（成功時に確保され、失敗時は空になる） */
PES_API int pes_run(PesContext* ctx, const PesRunParams* params, const PesImage* image, PesResult* result);
/* 画像ファイル（OpenCV で読めるもの）を処理する */
PES_API int pes_run_file(PesContext* ctx, const PesRunParams* params, const char* image_path, PesResult* result);
PES_API void pes_result_free(PesResult* result);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PE_SEGMENTATION_H */
