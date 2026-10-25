# cppsrc — PE-Spatial CLIP + EdgeTAM セマンティックセグメンテーション（OpenVINO / C++ 版）

Python 版（`core/common` + `core/openvino`、`develop.py` が使う処理）を C++ に書き起こしたもの。
OpenVINO だけで動き（PyTorch は不要）、**共有ライブラリ `libpe_segmentation.so`（と静的ライブラリ `.a`）** にして、
別のプログラムから `extern "C"` の関数呼び出しで使える。

呼び出しは 3 つ:

| # | 関数 | 内容 | 渡すパラメタ |
|---|------|------|-------------|
| 1 | `pes_create` | モデル読み込み（PE・EdgeTAM。ウォームアップまで） | `PesModelParams` |
| 2 | `pes_extract_text_features` | テキスト特徴量抽出（`prompts_dir` の yaml）。`pes_add_prompt` / `pes_remove_prompt` で追加・削除 | `PesPromptParams` |
| 3 | `pes_run` / `pes_run_file` | 処理実行（画像 → セマンティックマップ） | `PesRunParams` |

パラメタの名前と既定値は `configs/ov_config.yaml` と同じ。

## ディレクトリ構成

```
cppsrc/
├── CMakeLists.txt
├── build.sh                   ビルドスクリプト
├── include/pe_segmentation.h  公開 API（extern "C"）。別のプログラムが include するのはこれだけ
├── src/                       実装（下の対応表）
├── examples/main.cpp          サンプル（pe_segmentation_sample）。画像のパスを渡して実行
├── data/clip_bpe_merges.txt   CLIP の BPE の merges（OpenAI CLIP、MIT ライセンス。timm/PE-Core-T-16-384 の merges.txt と同一）。ビルド時にライブラリへ埋め込む
├── tests/                     単体テスト（CTest）
└── scripts/                   Python 版との照合用スクリプト（正解データの作成・結果の比較）
```

## 必要なもの

- g++ 9 以上（C++17）、CMake 3.16 以上
- OpenVINO の C++ SDK（2026.1 で確認。既定の探索先は `/opt/intel/openvino`。別の場所なら `OPENVINO_DIR`）
- OpenCV 4.x の開発パッケージ（core / imgproc / imgcodecs。Ubuntu なら `libopencv-dev`）
- 変換済みの OpenVINO モデル（`python convert_openvino.py` の出力。既定は `weights/openvino_models_l6`）
- AVX2 + FMA 対応の CPU を推奨（非対応なら自動でスカラー実装に切り替わる。動くが遅い）

## ビルド

```bash
cd cppsrc
./build.sh                 # Release。build/ に libpe_segmentation.so / .a と build/bin/ にサンプル・テスト
./build.sh --test          # ビルド後に単体テストを実行
./build.sh --install DIR   # DIR/lib・DIR/include・DIR/bin に install
./build.sh --help          # オプション一覧（--debug --clean --jobs N --shared-only --static-only --no-examples --no-tests）
```

OpenVINO を別の場所に入れている場合は `OPENVINO_DIR=/path/to/openvino/runtime/cmake ./build.sh`。

## サンプルの実行

**リポジトリのルートで**実行する（モデル・プロンプトの既定パスが `./weights/...`、`./prompts/...` の相対パスのため。
ルート以外で実行しても、現在のディレクトリに無ければ、実行ファイルから見たリポジトリのルートを探す）。
画像のパス以外は Python 版（`configs/ov_config.yaml`）と同じ値が既定。

```bash
./cppsrc/build/bin/pe_segmentation_sample /path/to/image.png            # 複数の画像も可
./cppsrc/build/bin/pe_segmentation_sample -v image1.png image2.png       # 各段階の処理時間も表示
./cppsrc/build/bin/pe_segmentation_sample --add-fg "a photo of a spoon." --remove-bg "a photo of a shadow." image.png
./cppsrc/build/bin/pe_segmentation_sample --config configs/ov_config.yaml --output-dir out image.png
```

結果は `outputs/experiment_results/<画像名>/`（`--output-dir` で変更）に保存される: `semantic_map.png`（画素値 = クラス番号）、
`semantic_color.png`、`semantic_overlay.png`、`semantic_result.png`（重畳 + 凡例）、`foreground_mask_n<クラス>.png`、
粗い推定の中間結果（`foreground_map.png` など）。Python 版の `ResultSaver` と同じ内容を OpenCV だけで書き出している。

## 別のプログラムからライブラリを使う

```c
#include "pe_segmentation.h"

PesModelParams mp;   pes_default_model_params(&mp);     // 既定値は ov_config.yaml と同じ
PesPromptParams pp;  pes_default_prompt_params(&pp);
PesRunParams rp;     pes_default_run_params(&rp);

PesContext* ctx = NULL;
if (pes_create(&mp, &ctx) != PES_OK) { fprintf(stderr, "%s\n", pes_last_error(NULL)); return 1; }   // 1. モデル読み込み

pes_extract_text_features(ctx, &pp);                                       // 2. テキスト特徴量抽出
pes_add_prompt(ctx, PES_PROMPT_FOREGROUND, "a photo of a spoon.");         //    追加（その場で特徴量を抽出）
pes_remove_prompt(ctx, PES_PROMPT_BACKGROUND, "a photo of a shadow.");     //    削除

PesResult result;
if (pes_run_file(ctx, &rp, "image.png", &result) == PES_OK) {              // 3. 処理実行
    // result.semantic_map : height × width の uint8（0=背景、1〜num_classes=前景クラス）
    // result.class_prompts[i - 1] : クラス i に対応する前景プロンプトの文
    pes_result_free(&result);
}
pes_destroy(ctx);
```

メモリ上の画像は `pes_run(ctx, &rp, &image, &result)`（`PesImage` に RGB または BGR の 8bit 3 チャンネルのポインタ・幅・高さ・
1 行のバイト数を入れる。OpenCV の `cv::Mat` なら `{mat.data, mat.cols, mat.rows, (int)mat.step, PES_PIXEL_BGR}`）。

### リンクの仕方

共有ライブラリ（`libpe_segmentation.so`）は OpenVINO・OpenCV を中に取り込んでいないので、実行時にそれらが見つかる必要がある
（`build.sh` が作る `build/` では RPATH に入っているので `LD_LIBRARY_PATH` は不要。install 後は `INSTALL_RPATH_USE_LINK_PATH` で同じ）。

```bash
g++ -std=c++17 app.cpp -I cppsrc/include -L cppsrc/build -lpe_segmentation -Wl,-rpath,$PWD/cppsrc/build -o app     # 共有
g++ -std=c++17 app.cpp -I cppsrc/include cppsrc/build/libpe_segmentation.a \
    -L/opt/intel/openvino/runtime/lib/intel64 -lopenvino $(pkg-config --libs opencv4) -o app                         # 静的
```

静的ライブラリを C のプログラムからリンクするときは `-lstdc++` も要る。CMake なら `add_subdirectory(cppsrc)` して `target_link_libraries(app PRIVATE pe_segmentation)`（静的は `pe_segmentation_static`。
OpenVINO・OpenCV が推移的に付く）。

### パラメタ（`configs/ov_config.yaml` のキーと同名）

| 構造体 | フィールド | 既定 | 意味 |
|--------|-----------|------|------|
| `PesModelParams`（`pes_create`） | `image_size` | 512 | 処理画像サイズ（CLIP / EdgeTAM に入れる正方形の一辺。PE のパッチサイズ 16 の倍数） |
| | `model_dir` | `./weights/openvino_models_l6` | OpenVINO モデルのディレクトリ |
| | `ov_num_thread` | 4 | OpenVINO の推論スレッド数 |
| | `device` | `cpu` | デバイス |
| | `et_use_int8` / `pe_use_int8` | 0 | INT8 モデル（`*_int8.xml`）を使うか |
| | `edgetam_high_res_feature_downscale_factor` | 4 | EdgeTAM の高解像度特徴量の縮小倍率 |
| | `tokenizer_merges_path` | 空 | 空なら埋め込み済みの BPE を使う（別の `merges.txt` を使うときだけ） |
| `PesPromptParams`（`pes_extract_text_features`） | `prompts_dir` | `./prompts/candidates/color_free_bg` | `foreground.yaml` / `background.yaml` のディレクトリ |
| `PesRunParams`（`pes_run`） | `k_clusters` | 8 | KMeans のクラスタ数 |
| | `n_iter` | 10 | KMeans のループ回数 |
| | `foreground_threshold` | 0.5 | 前景と判断する確信度 |
| | `edgetam_foreground_points_per_class` | 4 | EdgeTAM に入れる前景点数（クラスごと） |
| | `edgetam_background_point_count` | 4 | EdgeTAM に入れる背景点数 |
| | `use_edgetam_clustering` | 1 | EdgeTAM の特徴で前景確信度を伝播して揃えるか |
| | `edgetam_refine_iterations` | 2 | EdgeTAM でマスクを反復精緻化する回数 |
| | `semantic_max_side` | 1024 | マスク演算を行う作業解像度の長辺上限 |
| `PesOutputParams`（保存側。ライブラリは使わない） | `output_dir` `overlay_alpha` `show_result` `color_palette` | | サンプルの保存処理が使う |

yaml から読むなら `pes_load_config_yaml(path, &mp, &pp, &rp, &op)`（yaml に無いキーは変更しない。NULL の引数は無視）。
コードに固定している値（`core/common/params.py`）は `src/params.h`。値を変えるときはここだけ。

### プロンプトの追加・削除

| 関数 | 動作 |
|------|------|
| `pes_extract_text_features(ctx, &pp)` | yaml を読み、前景（color × shape + custom）・背景（base + custom）を**置き換えて**特徴量を抽出 |
| `pes_add_prompt(ctx, kind, text)` / `pes_add_prompts(ctx, kind, texts, n, &added)` | 末尾に追加。**追加した分の特徴量をその場で抽出**（既存の特徴量はそのまま）。同じ文は `PES_ERROR_ALREADY_EXISTS`（まとめて追加では飛ばす） |
| `pes_remove_prompt(ctx, kind, text)` / `pes_remove_prompt_at(ctx, kind, index)` | 削除（特徴量も同時に取り除く）。無ければ `PES_ERROR_NOT_FOUND` |
| `pes_clear_prompts(ctx, kind)` | 全削除 |
| `pes_prompt_count` / `pes_prompt_text` | 現在のプロンプト |

`kind` は `PES_PROMPT_FOREGROUND` / `PES_PROMPT_BACKGROUND`。前景・背景の両方が 1 つ以上ないと `pes_run` は `PES_ERROR_NO_PROMPTS` を返す。
プロンプトを変えると、結果の `prompt_indices`（前景 → 背景の通し番号）の指す先も変わるので、文が必要なら結果の `class_prompts`（結果が持つ複製）を使う。

### 結果（`PesResult`）

最終結果（`semantic_map`・`prompt_indices`・`class_prompts`・EdgeTAM に与えた点 `point_coords` / `point_labels`）と、
粗い推定の中間結果（パッチ格子 `feature_size`² の `foreground_map`・`foreground_mask`・`cluster_labels`・`semantic_labels` など）、
各段階の処理時間 `timings` が入る。`pes_result_free` で解放する。`pes_set_verbose(ctx, 1)` で Python 版と同じ形式のログも出せる。

### エラーとスレッド安全性

- すべての関数は `PesStatus` を返す（`PES_OK` = 0、エラーは負）。C++ の例外は境界で捕まえて戻り値に変換するので、呼び出し側に例外は出ない。
  詳細な文は `pes_last_error(ctx)`（`pes_create` など `ctx` が無いときは `pes_last_error(NULL)`、スレッドごと）。
- 1 つの `PesContext` を複数スレッドから同時に使ってはいけない（別々の `PesContext` なら可。ただしモデルを複数読み込むのでメモリと CPU を食う）。
- `pes_create` は `cv::setNumThreads(2)` を呼ぶ（Python 版の `OPENCV_NUM_THREADS` と同じ。OpenCV のグローバル設定なので、呼び出し側の OpenCV の使い方に影響する）。

## Python 版との対応

| C++ (`src/`) | Python 版 | メモ |
|--------------|-----------|------|
| `segmenter.cpp` | `core/common/segmentation_runner.py` | 画像 1 枚分の全体の流れ |
| `clip_extractor.cpp` | `core/openvino/feature_extractor.py` | PE の画像・テキスト・クラスタトークン。未使用の出力（`cls2patch_attention` と `raw_image_features`）をグラフから外す |
| `edgetam_runner.cpp` | `core/openvino/edgetam_runner.py` + `core/common/edgetam_helper.py` | エンコーダ・デコーダと座標変換。エンコーダの出力はデコーダの入力へコピー無しで渡す |
| `coarse_segmentation.cpp` | `core/common/coarse_segmentation.py` | KMeans・クラスタ分類・EdgeTAM 特徴での伝播・粗いラベルの整形。プロンプトの管理もここ |
| `semantic_refiner.cpp` | `core/common/semantic_refiner.py` | クラスごとの EdgeTAM 精緻化・重なり解決・取りこぼしの再推定 |
| `mask_utils.cpp` | `core/common/mask_utils.py` | 点サンプリング・連結成分・ノイズ除去・ガイデッドフィルタ・kNN グラフ |
| `kmeans.cpp` | `faiss.Kmeans` | faiss と同じ手順の自前実装 |
| `tokenizer.cpp` | `openvino_tokenizers` の `pe_tokenizer.xml` | CLIP の BPE の自前実装（拡張ライブラリ不要） |
| `yaml_lite.cpp` `config.cpp` | `yaml.safe_load` / `core/utils.py` | 設定・プロンプト yaml の最小パーサー |
| `params.h` | `core/common/params.py` | 固定ハイパーパラメータ |
| `simd.cpp` `simd_avx2.cpp` | NumPy / BLAS | AVX2 + FMA の数値カーネル（下記） |

### ライブラリの置き換え

| Python 版 | C++ 版 |
|-----------|--------|
| `scipy.ndimage` の `binary_erosion` / `binary_dilation` / `label` | `cv::erode` / `cv::dilate` / `cv::connectedComponents`（境界の扱いも scipy と同じ設定） |
| `distance_transform_edt`（距離のみ） | `cv::distanceTransform(DIST_L2, DIST_MASK_PRECISE)` |
| `distance_transform_edt(return_indices=True)` | OpenCV に「最寄りの特徴画素の位置」を返す関数が無いので自前（Felzenszwalb の線形時間アルゴリズム）。距離が同じ画素が複数あるときの選び方だけ scipy と異なりうる |
| `scipy.sparse`（kNN グラフ・伝播） | CSR を自前（float32 のまま計算する点も Python 版と同じ） |
| NumPy / BLAS の行列積 | 自前の AVX2 + FMA カーネル |
| `faiss.Kmeans` | 自前（`mt19937` による初期化・空クラスタの分割・BLAS と同じ丸め方まで faiss に合わせてある） |
| PIL（画像読み込み）、`cv2` | OpenCV (`cv::imread`, `cv::resize` など)。デコードとリサイズの結果は Python 版とビット単位で同じことを確認 |
| matplotlib の凡例つき図 | `cv::putText` で描いた凡例（サンプル側） |

## AVX2 による高速化

`simd_avx2.cpp` だけ `-mavx2 -mfma` でビルドし、実行時に CPU を調べて切り替える（非対応 CPU でも動く。`simd::set_force_scalar` で強制もできる）。
他のファイルは汎用の命令のまま。浮動小数点は NumPy と同じ演算順序にそろえるため `-ffp-contract=off`（FMA は AVX2 カーネルの中で明示的に使う）。

| カーネル | 用途 | 備考 |
|----------|------|------|
| `gram_symmetric` / `topk_rows` | EdgeTAM 特徴の kNN グラフ（1024×256 の類似度行列。対称性で上三角だけ・8×8 タイルの FMA、各行の上位 8 個の選択） | gram 約 3.5ms（スカラー 100ms 超）、topk 約 0.6ms |
| `gemv_t_blas` / `normalize_rows` / `sum_squares_numpy` | クラスタ分類（トークンの正規化 → テキスト特徴量との類似度）、テキスト特徴量の正規化 | NumPy / OpenBLAS と同じ丸め方（下記） |
| `gemm_nt_chain` / `norm_sq` | KMeans の割り当て（直列 FMA の連鎖を 8 点同時に進めて遅延を隠す） | faiss（OpenBLAS）と同じ丸め方 |
| `normalize_to_chw` | 画像の前処理（u8 → CLIP と EdgeTAM の 2 種類の正規化済み CHW を 1 回の走査で） | 約 4 倍（512×512 で 0.3ms） |
| `transpose` | (C, N) → (N, C)（8×8 ブロック） | 約 5 倍 |
| `sigmoid` / `score_from_logit` / `clamp` / `greater_than` / `update_best` | 精緻化（ロジット → 確信度・重なり解決・二値化） | `exp` は Cephes の近似（相対誤差 ~1e-7） |

「NumPy / OpenBLAS と同じ丸め方」は、`np.linalg.norm`（pairwise summation）、`(1×512) @ (512×M)`（OpenBLAS の sgemv: 4 列ずつ 8 レーン FMA、余りの 2 列は
SSE 4 レーンの mul+add、余りの 1 列は AVX 8 レーンの mul+add）、faiss の KMeans（sgemm: 直列な FMA の連鎖）を実測で突き合わせて再現したもの。
これでテキスト特徴量・クラスタトークンの正規化と類似度（logits）、KMeans のクラスタ番号が Python 版とビット単位で一致する
（`tests/test_reference` と `tests/test_kmeans`。KMeans は実画像 141 枚の特徴量でも faiss と全点一致）。確信度は softmax の `exp` が NumPy の SIMD 実装と最後の 1 ulp 違いうる。

`tests/test_simd` が AVX2 とスカラーの結果の一致（ビット単位。`sigmoid` / `score_from_logit` だけ誤差の範囲）を確認する。

## テスト

```bash
./build.sh --test                 # simd / tokenizer / config / api（モデルがあれば）
```

Python 版と突き合わせる正解データが要るテストは、先に Python でデータを作り、`PES_TEST_DATA_DIR` を指定して CMake を再実行する:

```bash
python cppsrc/scripts/gen_tokenizer_cases.py                       # tests/data/tokenizer_cases.json（pe_tokenizer.xml の出力）
python cppsrc/scripts/gen_kmeans_cases.py   /tmp/pes_test/kmeans    # faiss.Kmeans の出力
python cppsrc/scripts/gen_reference_cases.py /tmp/pes_test/reference   # mask_utils / coarse_segmentation の純粋な関数の出力
PES_TEST_DATA_DIR=/tmp/pes_test ./cppsrc/build.sh --test
```

| テスト | 内容 |
|--------|------|
| `test_simd` | AVX2 とスカラーの一致、速度の目安 |
| `test_tokenizer` | `pe_tokenizer.xml`（OpenVINO）と同じ ID（プロンプト全部 + ランダムな文 740 件） |
| `test_config` | `configs/ov_config.yaml` と `prompts/` の yaml を読めること、既定値が yaml と一致 |
| `test_kmeans` | `faiss.rand_perm` と `faiss.Kmeans(seed=42)` のクラスタ番号が一致 |
| `test_reference` | 点サンプリング・連結成分・`clean_mask`・`remove_small_regions`・`fill_narrow_gaps`・ガイデッドフィルタ・伝播・粗いラベルの整形が Python 版と一致 |
| `test_api` | C API の通し（読み込み → 抽出 → 実行、プロンプトの追加・削除、エラーの戻り値、stride・RGB/BGR・極小/縦長の画像） |

## Python 版との結果の比較

```bash
python cppsrc/scripts/compare_with_python.py --images-file list.txt          # 画像のパスを 1 行に 1 つ
python cppsrc/scripts/compare_with_python.py --system-openvino img1.png ...  # Python 版も C++ 版と同じ OpenVINO で動かす
```

同じ画像・同じ設定で Python 版（`SegmentationRunner` + OpenVINO backend）と C++ 版を動かし、最終のセマンティックマップと
粗い推定の中間結果（前景確信度・クラスタ・ラベル・EdgeTAM に与えた点）を比べる。

### 結果（141 枚。MVTec 15 カテゴリ × 5 枚 + 分類サンプル 38 枚 + 物体検出 12 枚 + COCO 16 枚。OK/NG 混在）

| 項目 | 結果 |
|------|------|
| PE・EdgeTAM の出力（`fused_layer_features` `raw_patch_features` `image_embed`、テキスト特徴量） | 同じ OpenVINO 同士でビット単位で一致（FP32・PE の INT8 とも） |
| KMeans のクラスタ番号・粗いセマンティックラベル・粗い前景マスク | **141 / 141 枚で完全一致** |
| 最終のセマンティックマップ | **127 / 141 枚で全画素一致**。平均の一致画素 99.963%、最小 98.2%。前景 IoU は平均 0.9993（最小 0.959） |
| 前景確信度（`foreground_map`）の最大差 | 中央値 1e-6 |
| 設定を変えたとき（`image_size` 384・640、`k_clusters` 6、`foreground_threshold` 0.4、点数 3、反復 3、`semantic_max_side` 640） | 同じ水準（クラスタ・粗いラベルは全部一致、最終マップの一致 98〜100%） |

全画素が一致しない 14 枚は、すべて**背景の負例点の選択**が Python 版と違うことが原因（最終マスクが数十〜数千画素変わる）。
背景の負例点は「前景確信度が最も低いパッチ」から選ぶため、確信度がほぼ同じパッチが並ぶ画像では、~1e-6 の差で別のパッチが選ばれる。
差の出どころは、(1) Python 版の kNN グラフの隣接リストの順序（`argpartition` と `scipy.sparse.maximum` の出力順）が実装依存で、
伝播の足し込みの順序が C++ 版と違うこと、(2) 行列積 `X @ X.T` の OpenBLAS 内部の丸め（大きい行列で 1〜2 ulp 違う）による kNN の 8 位と 9 位の入れ替わり、
(3) `np.exp`（float32 の SIMD 実装）と `std::exp` の最後の 1 ulp。Python 版どうしでも BLAS やバージョンが変われば同じ程度の差が出る。
実際、同じ Python 関数に C++ 版の確信度を入れると C++ 版と同じ点が選ばれる（`select_background_points` は単体テストで全件一致）。

INT8: PE の INT8（`pe_use_int8`）は Python 版と同じ結果。EdgeTAM の INT8（`et_use_int8`）はもともと壊れている（エンコーダの出力が 1e15 のオーダーになり、前景が 0%。CLAUDE.md の記載どおり）ので、C++ 版でも使えない。

### 処理時間（capsule/000.png、1000×1000、OpenVINO 4 スレッド）

同じマシンで、他のプロセスが動いている状態（負荷が変動するので、30〜40 回のうちの最小と中央値）で交互に測った。

| | 最小 | 中央値 |
|---|------|--------|
| Python 版（`SegmentationRunner`） | 236〜262 ms | 267〜313 ms |
| **C++ 版** | **196〜200 ms** | **226〜236 ms** |

C++ 版の内訳（最小だった回）:

| 段階 | ms | Python 版のログ（同じ画像） |
|------|----|----------------------------|
| 画像前処理 | 0.8 | 2.6 |
| EdgeTAM エンコーダ | 27〜31 | 32 |
| PE 画像特徴量 | 79〜82 | 85 |
| 粗い前景マスク取得（KMeans 0.6 + クラスタトークン 1.9 + 伝播 6） | 9 | 15 |
| 粗いラベルの整形 | 0.3 | 0.9 |
| EdgeTAM による詳細化（デコーダ 8 回 約 55ms を含む） | 69〜81 | 91 |

モデルの推論（PE・EdgeTAM・デコーダ。約 165ms）は OpenVINO のまま変えていないので、C++ 化で減らせるのは前後の処理（Python 版で約 70ms → C++ 版で約 30ms）。
`pe_segmentation_sample --repeat N` で、各回の合計と、最小だった回の内訳が出る。

## 注意・既知の差

- **数値のゆれ**: 上の「結果」のとおり、PE・EdgeTAM の出力と KMeans・粗いラベルは Python 版と一致するが、前景確信度は ~1e-6 ずれ、
  確信度がほぼ同じパッチが並ぶ画像では背景の負例点が別のパッチになって、最終マスクが少し変わることがある。完全に同じにはできない（Python 側の順序が実装依存のため）。
- **OpenVINO のバージョン**: Python 版は pip 版 openvino（`convert_openvino.py` と同じ理由でシステム版を `sys.path` から外している）、
  C++ 版はシステム版（`/opt/intel/openvino`）を使う。IR（xml/bin）は共通で、推論結果はビット単位で一致することを確認している。
- **トークナイザ**: `pe_tokenizer.xml` は `openvino_tokenizers` の拡張が要る（pip 版 openvino にしか読み込めない）ため、
  CLIP の BPE を C++ で実装した。通常の英語のプロンプトは完全に一致する。互換文字の正規化（NFKC。全角英数字・合字など）は未対応。
- **画像サイズ**: モデルは動的サイズだが、`image_size` は PE のパッチサイズ（16）の倍数にすること。
- 入力画像は 8bit 3 チャンネル。`pes_run_file` は OpenCV で読めるもの（グレースケール・アルファ付きは 3 チャンネルに変換される）。
- Python 版の手法を変えたとき（`core/common`）は、対応する C++ の関数も更新して `scripts/compare_with_python.py` で確認すること。
