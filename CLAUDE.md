# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

[facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)（PE-Spatial CLIP）と EdgeTAM（SAM2 系の軽量プロンプトセグメンテーション）を組み合わせ、CPU 上で高速に前景/背景・セマンティックセグメンテーションを行うリポジトリ。最終目標は OpenVINO で ~200ms。コメント・出力・README は日本語。

ビルド・lint・テストの仕組みは無い。実行はスクリプトを直接動かす。

## コマンド

```bash
python convert_openvino.py   # PyTorch (weights/*.pt) -> OpenVINO (weights/openvino_models)。実行時は先に必須
python develop.py            # OpenVINO 版の実行（本命・最新）
python test_pytorch.py       # PyTorch 版の実行（core/common を共有するので OpenVINO 版と同じ処理。手法の実験用）
python test_pytorch_sam2.py  # PyTorch 版を SAM2.1（公式 Hiera-T）で精緻化して実行（configs/torch_config_sam2.yaml。EdgeTAM との差し替えの検証用）
python prompts/create_prompts.py  # prompts/semantic.json を生成
python tools/evaluation/eval_prompts.py [--prompts-dir DIR]  # MVTec(fg_mask)でプロンプトを評価（前景IoU・クラスの割れやすさ）
python tools/evaluation/eval_prompts.py --gt-dir DIR  # 任意の画像+正解マスク(DIR/images, DIR/masks)で前景IoU・物体の検出率(大きさ別)・背景の誤検出を評価
python tools/evaluation/diag_clusters.py  # 粗いセグメンテーションの診断（クラスタ単位の前景/背景判定の誤り率）
python tools/evaluation/diag_refined_judgement.py [--gt-dir DIR]  # リファイン後のマスクを前景/背景に再判定したときの効果を診断（採用していない案の検証用）
python tools/experiments/tiling/eval_tiling.py [--mode baseline|A|B] [--viz-dir DIR] [--diag]  # Overlapping Tiling（4タイル+全体縮小）で前景マップを作る実験。本体には組み込んでいない
python tools/experiments/decontam/exp_decontam.py --phase p0,p1,p2,p3,p4,p5,p6 --num 6  # 前景特徴への背景混入を軽減する手法の検証（PyTorch版、結果は outputs/decontam/results.json）
python tools/experiments/decontam/report_decontam.py  # 上の結果を outputs/decontam/index.html に可視化
python tools/experiments/decontam/exp_decontam_cases.py  # 粗いマスクと最終結果（EdgeTAMまで）を画像で並べる
python tools/experiments/decontam_followup/exp_followup_data.py  # 小物体・複数物体の合成データ（MVTecの部品×背景）を outputs/decontam_followup/data/ に作る
python tools/experiments/decontam_followup/exp_followup.py --task {synthetic,mvtec,refine,develop,ov,cases,timing} --variants ...  # 追検証（変種は exp_decontam.py の registry 名。--verify で独立した画像）
python tools/experiments/decontam_followup/report_followup.py  # 上の結果を outputs/decontam_followup/index.html に可視化
python tools/experiments/dual_cluster/exp_dualcluster.py --task {m1,analysis,mvtec,synthetic,cases,viz,cost,develop} --variants ...  # PE と EdgeTAM の特徴を別々にKMeansして粗い結果を統合する実験（PyTorch版、結果は outputs/dual_cluster/。本体には組み込んでいない）
python tools/experiments/dual_cluster/report_dualcluster.py  # 上の結果を outputs/dual_cluster/index.html に可視化
python tools/experiments/dual_cluster_recheck/exp_recheck.py --task {oracle,control,mvtec,synthetic,cases,develop} --variants ... [--setting default|ring] [--verify]  # dual_cluster の再検証（zipper を除く・精緻化で粗いマスクの外周1パッチを「不明」にする案。結果は outputs/dual_cluster_recheck/。本体には組み込んでいない）
python tools/experiments/dual_cluster_recheck/exp_fgarea.py  # fg_mask の面積比（zipper を評価から除く根拠）
python tools/experiments/dual_cluster_recheck/report_recheck.py  # 上の結果を outputs/dual_cluster_recheck/index.html に可視化
python tools/setup/vendor_sam2.py [--src /home/llm/sam2]  # 公式 SAM2.1 を core/torch/sam2_official/ にコピー（パッケージ名を sam2_official に変えるだけ。core/torch/sam2 は EdgeTAM の fork で別物）
python tools/experiments/sam2/exp_sam2.py --task {mvtec,synthetic,timing,cases,develop} --backend {edgetam,sam2} --points 前景:背景,... [--setting default|ring] [--verify]  # EdgeTAM → SAM2.1 の切り替えと、プロンプト点数の測定（PyTorch版、結果は outputs/sam2/。本体には組み込んでいない）
python convert_openvino.py --cluster-layers 6 --out-dir ./weights/openvino_models_l6 --pe-only  # クラスタリングに使う層を変えた PE のモデルだけを別フォルダに書き出す（既存のモデルは上書きしない）
python tools/yoloe/app.py  # YOLOE（テキストプロンプトのゼロショット検出）を試すWeb GUI。CLIは tools/yoloe/yoloe.py。詳細は tools/yoloe/README.md
python tools/dataset/make_synthetic_dataset.py --parts DIR --backgrounds DIR --out DIR  # 部品の切り抜きを背景に合成し、マスク・個数の正解付きデータを作る
python tools/evaluation/eval_synthetic.py --data DIR  # 合成データでIoU・有無・個数を条件別に評価
```

- `tools/` は用途別のフォルダ: `evaluation/`（評価・診断）, `dataset/`, `gui/`, `experiments/<名前>/`（手法の検証実験。本体には未組み込みで、`outputs/<名前>/` と対）。実験どうしは `from tools.experiments.decontam import exp_decontam as E` の形で import し合う。構成は `tools/README.md`。評価カテゴリは `tools/evaluation/eval_prompts.py` の `OBJECT_CATEGORIES`（10 種。過去の数値はこれ）と `EVAL_CATEGORIES`（zipper を除く 9 種。zipper は fg_mask が画面の 78% で物体として扱う意味が無い）。
- 実行対象画像は `develop.py` / `test_pytorch.py` 内の `image_paths` リストを直接編集する（CLI 引数なし）。
- 設定は `configs/ov_config.yaml`（OpenVINO）/ `configs/torch_config.yaml`（PyTorch）。`core.utils.load_config()` の既定は ov_config。
- `convert_openvino.py` は冒頭で `/opt/intel/openvino` を `sys.path`/`PYTHONPATH` から除去する（pip 版 openvino との衝突回避）。この処理を `import openvino` より前に保つこと。
- `weights/`, `outputs/`, `*.png`, `*.jpg` は gitignore 対象。

## アーキテクチャ

処理の本体は backend 非依存の `core/common/` にあり、モデル推論の「口」だけを `core/openvino/`（OpenVINO）と `core/torch/`（PyTorch）が持つ。**手法の開発は `core/common/` で行えば、両 backend に反映される。** `develop.py`（OpenVINO）と `test_pytorch.py`（PyTorch）は backend の生成部だけが違う。`core/torch/` は変換元モデルの定義も兼ねる（`convert_openvino.py` が `DecodeModel`, `RepVitModel`, `TorchCLIPFeatureExtractor` 等を import して変換）。`core/torch/sam2/` は SAM2/EdgeTAM のベンダリングコード。

`core/common/` の構成（torch / openvino を import しない）:

- `backend.py`: backend の契約（`CLIPBackend` / `EdgeTamBackend` の Protocol）。入出力はすべて `np.ndarray`。
- `edgetam_helper.py`: `EdgeTamHelper`。EdgeTAM の座標変換・プロンプト前処理・後処理（NumPy）。backend はこれを継承して `encode` / `decode` だけ実装する。
- `segmentation_runner.py`: `SegmentationRunner`。`segment(image)`（保存なしのパイプライン、`SegmentationResult` を返す）と `run(path)`（読み込み・`segment`・保存）。
- `coarse_segmentation.py`: `CoarseSegmenter`。CLIP 特徴量・テキストプロンプトとの照合・faiss KMeans による粗い前景/セマンティック推定と、パッチ単位の整形。
- `semantic_refiner.py`: `SemanticRefiner`。クラスごとに EdgeTAM backend でマスクを推定し、スコアで重なりを解決して 1 枚のセマンティックマップにする。
- `mask_utils.py`: 点サンプリング（`farthest_point_sampling` 等）・連結成分・ノイズ除去などの純粋関数。
- `params.py`: yaml に出さずコードに固定したハイパーパラメータ（値を変えるときはここだけ編集）。
- `visualization.py` / `result.py`: 結果の保存・可視化 / `SegmentationResult`。

backend:

- OpenVINO: `core/openvino/feature_extractor.py`（`OVCLIPFeatureExtractor`）, `edgetam_runner.py`（`OVEdgeTamRunner`）
- PyTorch（実験用）: `core/torch/feature_extractor.py` の `TorchCLIPProbe`。全層の hidden・attention 出力を取り出し、後半ブロックを attention マスク付きで再実行できる。`tools/experiments/decontam/exp_decontam*.py` が使う。
- PyTorch（SAM2.1）: `core/torch/sam2_runner.py` の `TorchSam2Backend`。EdgeTAM のクラス（`RepVitModel` / `DecodeModel` / `TorchEdgeTamBackend`）を継承し、モデルの取得だけを公式 SAM2.1 に差し替える（デコーダーの入出力と特徴量の形状が同じなので、`SegmentationRunner(extractor, TorchSam2Backend(config), config)` でそのままつながる）。`core/torch/sam2_official/` は公式 SAM2.1 のコピー（パッケージ名 `sam2_official`、`UPSTREAM.md` 参照）。`core/torch/sam2/` は EdgeTAM の fork で公式とは別物（同名のパッケージは同時に置けない）。
- PyTorch: `core/torch/feature_extractor.py`（`TorchCLIPBackend`。変換済みモデルと同じグラフ `to_openvino=True` で動かし、出力を ndarray 化）, `edgetam_runner.py`（`TorchEdgeTamBackend`）
- `core/torch/legacy/segmentation_runner.py`: 共通化前の旧パイプライン（単一マスク前景）。参照用で、現在の config では動かない。

パイプライン（`SegmentationRunner.segment`）:

1. `resize_image` + `normalize_image`（リサイズは CLIP と EdgeTAM で共通、正規化はモデルごと）
2. `CoarseSegmenter.extract_features`: CLIP 画像特徴（パッチ・fused パッチ）
3. `generate_foreground_map`: テキストプロンプト（`prompts_dir` の `foreground.yaml` / `background.yaml`）との類似度 + faiss KMeans のクラスタで粗い前景マスクとクラスタ分割を得る
4. `assign_semantic_labels`: クラスタに対応するプロンプトで粗いセマンティックラベルを付与
5. `refine_coarse_segmentation`: 穴埋め・細い領域の統合
6. `SemanticRefiner.refine`: 前景（クラスごと `edgetam_foreground_points_per_class` 点）/ 背景（`edgetam_background_point_count` 点）のプロンプト点で EdgeTAM の encoder/decoder を反復し、クラスを統合
7. `run()` では `ResultSaver` が `output_dir/<画像名>/` に可視化を保存

`core/utils.py` に共通ヘルパー（`Timer` コンテキストマネージャー、`load_image`, `resize_image`, `normalize_image`, `preprocess_image`, `load_config`）。各処理は `with Timer(...)` で ms 計測する流儀。

## 補足

- 座標系: 入力画像 → CLIP パッチ特徴マップ（`image_size`/patch）→ EdgeTAM の座標の変換は `SemanticRefiner.transform_coords` / `EdgeTamHelper.transform_coords` を使う。
- クラスタリングに使う PE の層は config の `cluster_layers`（既定 `[8, 9, 10]`）。OpenVINO 版は変換時に固定される（`convert_openvino.py --cluster-layers`）。第 6 層のみのモデルは `weights/openvino_models_l6` に作成済み（切り替えは `configs/ov_config.yaml` の `model_dir`）。
- INT8 モデルは `et_use_int8` / `pe_use_int8` で切替（EdgeTAM / PE それぞれ）。
- データセットのパス（`/home/llm/datasets/...`）は環境固有でリポジトリには含まれない。
