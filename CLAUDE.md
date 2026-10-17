# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

[facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)（PE-Spatial CLIP）と EdgeTAM（SAM2 系の軽量プロンプトセグメンテーション）を組み合わせ、CPU 上で高速に前景/背景・セマンティックセグメンテーションを行うリポジトリ。最終目標は OpenVINO で ~200ms。コメント・出力・README は日本語。

ビルド・lint・テストの仕組みは無い。実行はスクリプトを直接動かす。

## コマンド

```bash
python convert_openvino.py   # PyTorch (weights/*.pt) -> OpenVINO (weights/openvino_models)。実行時は先に必須
python develop.py            # OpenVINO 版の実行（本命・最新）
python test_pytorch.py       # PyTorch 版の実行（動作確認用。最新の処理と同期しているとは限らない）
python prompts/create_prompts.py  # prompts/semantic.json を生成
python tools/eval_prompts.py [--prompts-dir DIR]  # MVTec(fg_mask)でプロンプトを評価（前景IoU・クラスの割れやすさ）
python tools/make_synthetic_dataset.py --parts DIR --backgrounds DIR --out DIR  # 部品の切り抜きを背景に合成し、マスク・個数の正解付きデータを作る
python tools/eval_synthetic.py --data DIR  # 合成データでIoU・有無・個数を条件別に評価
```

- 実行対象画像は `develop.py` / `test_pytorch.py` 内の `image_paths` リストを直接編集する（CLI 引数なし）。
- 設定は `configs/ov_config.yaml`（OpenVINO）/ `configs/torch_config.yaml`（PyTorch）。`core.utils.load_config()` の既定は ov_config。
- `convert_openvino.py` は冒頭で `/opt/intel/openvino` を `sys.path`/`PYTHONPATH` から除去する（pip 版 openvino との衝突回避）。この処理を `import openvino` より前に保つこと。
- `weights/`, `outputs/`, `*.png`, `*.jpg` は gitignore 対象。

## アーキテクチャ

`core/torch/` と `core/openvino/` が並行する同名構造（`feature_extractor`, `edgetam_runner`, `segmentation_runner`）。**開発の主軸は `core/openvino/`**。`core/torch/` は変換元モデルの定義も兼ねる（`convert_openvino.py` が `DecodeModel`, `RepVitModel`, `TorchCLIPFeatureExtractor` 等を import して変換）。`core/torch/sam2/` は SAM2/EdgeTAM のベンダリングコード。

パイプライン（`core/openvino/segmentation_runner.py` の `SegmentationRunner.run`）:

1. `preprocess_image` → `OVCLIPFeatureExtractor`（CLIP 画像特徴: 画像全体・パッチ・fused パッチ）
2. 画像分類（`_image_classification`）
3. `_generate_foreground_map`: テキストプロンプト（`prompts/foreground.yaml` / `background.yaml`）との類似度 + faiss KMeans のクラスタで粗い前景マスクとクラスタ分割を得る
4. `semantic_segmentation`: クラスタ代表トークンを `prompts/semantic.json` のカテゴリ語彙と照合してセマンティックラベルを付与
5. `select_prompt_points`: 前景（クラスごと `edgetam_foreground_points_per_class` 点）/ 背景（`edgetam_background_point_count` 点）を `farthest_point_sampling` 等で抽出
6. `_run_edgetam_refinement`: ポイントを EdgeTAM の encoder/decoder（`OVEdgeTamRunner`）に渡し精緻化
7. `_save_results` が `output_dir/<画像名>/` に可視化を保存

注意: 現在 `run()` 内の EdgeTAM 精緻化は `and False` で無効化されている（セマンティックセグメンテーション実装中の WIP）。変更時は意図的なものか確認すること。

`core/utils.py` に共通ヘルパー（`Timer` コンテキストマネージャー、`load_image`, `preprocess_image`, `load_config`）。各処理は `with Timer(...)` で ms 計測する流儀。

## 補足

- 座標系: 入力画像 → CLIP パッチ特徴マップ（`image_size`/patch）→ EdgeTAM の座標の変換は `SegmentationRunner.transform_coords` / `OVEdgeTamHelper.transform_coords` を使う。
- INT8 モデルは `et_use_int8` / `pe_use_int8` で切替（EdgeTAM / PE それぞれ）。
- データセットのパス（`/home/llm/datasets/...`）は環境固有でリポジトリには含まれない。
