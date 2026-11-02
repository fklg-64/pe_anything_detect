# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

[facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)（PE-Spatial CLIP）と EdgeTAM（SAM2 系の軽量プロンプトセグメンテーション）を組み合わせ、CPU 上で高速に前景/背景・セマンティックセグメンテーションを行うリポジトリ。最終目標は OpenVINO で ~200ms。コメント・出力・README は日本語。

ビルド・lint・テストの仕組みは無い。実行はスクリプトを直接動かす。

## コマンド

```bash
python convert_openvino.py   # PyTorch (weights/*.pt) -> OpenVINO (weights/openvino_models)。実行時は先に必須。`--edgetam-only` で EdgeTAM だけ（約 30 秒）。`--region-only` で矩形用の代表トークンのモデル（`pe_region_tokens_model`。`tools/visualizer` が使う。既存のモデルは触らない）だけ。デコーダーは画像サイズ可変（変換後に別サイズで PyTorch と照合して確認）
python develop.py            # OpenVINO 版の実行（本命・最新）
python test_pytorch.py       # PyTorch 版の実行（core/common を共有するので OpenVINO 版と同じ処理。手法の実験用）
python test_pytorch_sam2.py  # PyTorch 版を SAM2.1（公式 Hiera-T）で精緻化して実行（configs/torch_config_sam2.yaml。EdgeTAM との差し替えの検証用）
python tools/evaluation/eval_prompts.py [--prompts-dir DIR]  # MVTec(fg_mask)でプロンプトを評価（前景IoU・クラスの割れやすさ）
python tools/evaluation/eval_prompts.py --gt-dir DIR  # 任意の画像+正解マスク(DIR/images, DIR/masks)で前景IoU・物体の検出率(大きさ別)・背景の誤検出を評価
python tools/evaluation/eval_pipeline.py --backend {torch,openvino} [--cluster-layers 6] [--set key=value ...] [--tasks mvtec,verify,synthetic,agree,timing]  # 本体のパイプライン（SegmentationRunner + PyTorch / OpenVINO backend）を MVTec と合成データで評価（設定を上書きして比べられる。結果は outputs/implementation/）
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
python tools/experiments/diagnosis/exp_diagnosis.py --task {floor,oracle,decompose,classes,cases} [--data-set synthetic|mvtec]  # 性能向上の方針のための診断（粗い結果を正解にしたときの上限・粗い誤りと最終マスクの誤りの分解・クラス単位の評価。結果は outputs/diagnosis/。本体には組み込んでいない）
python tools/experiments/diagnosis/exp_timing.py --task {stages,backbones}  # OpenVINO 版の段階別の時間 / 候補バックボーン（PE-Spatial・DINOv3 など）の OpenVINO 遅延（重みなし）
python tools/experiments/diagnosis/report_diagnosis.py  # 上の結果を outputs/diagnosis/index.html に可視化
python tools/experiments/prior_upsample/exp_prior.py --task {oracle,mvtec,synthetic} --modes default,ring,ring_gg4 [--coarse dual_and_embed_L6] [--points 4:4] [--verify]  # 精緻化の事前マスクの縁の実験（外周「不明」+ ガイデッドフィルタ = 手法 D。結果は outputs/prior_upsample/。本体には組み込んでいない）
python tools/experiments/prior_upsample/viz_prior.py --task {develop,mvtec,synthetic}  # 事前マスクの作り方ごとの事例画像
python tools/experiments/crop_cls/exp_crop.py --task {components,mvtec,synthetic,classes} --variants crop224_masked_and,...  # 領域の分類を切り抜きの CLIP 埋め込みで行う実験（C1。効果なし。結果は outputs/crop_cls/）
python tools/experiments/dense_features/exp_dense.py --task {mvtec,synthetic} --variants feat_spatial,feat_pe6+edge,...  # クラスタリングの特徴源の差し替え・連結（C2。PE-Spatial-T など。効果なし。結果は outputs/dense_features/）
python tools/experiments/phase2/report_phase2.py  # 上の 3 実験の結果を outputs/phase2/index.html に可視化
python convert_openvino.py --cluster-layers 6 --out-dir ./weights/openvino_models_l6 --pe-only  # クラスタリングに使う層を変えた PE のモデルだけを別フォルダに書き出す（既存のモデルは上書きしない）
python tools/experiments/bg_points/exp_bg_{mechanism,quantile,diverse}.py OUT.json  # 背景の負例点が密集する理由と、ばらしたときの精度の測定（効果なし。結果は outputs/bg_points/）
python tools/experiments/fg_points/exp_fg_points.py OUT.json [--variants base,w_depth,rim_seed,deep_core,deepest]  # 前景のプロンプト点を粗いマスクの縁から離す案（効果なし・離すほど悪化。結果は outputs/fg_points/。診断は diag_fg_points.py、表は analyze.py、報告は report_fg_points.py）
python tools/experiments/fg_points/exp_snap.py OUT.json --variants base,oracle_p1,shift_core,snap_all_p1_best,hf_region_w,...  # 前景の点を、端を選んだまま物体の上に合わせる案（パッチ内で動かす snap_*、EdgeTAM の高解像度特徴量の画素から選ぶ hf_*。hf_region_w が MVTec +0.016・独立 +0.023・合成 +0.010。効きは toothbrush に偏る。**hf_region_w は本体に組み込み済み**（`core/common/object_points.py`）で、変種 `base` = 従来のパッチ中心の点（`mask_utils.prompt_points`。本体の切替 `use_object_points` は 2026-10-31 に削除したので、`RegionEstimator.foreground_points` を差し替えて再現する）、`main` = 本体。名前の規則は exp_snap.py の docstring。副作用は check_fg_points.py、事例は viz_snap.py、結果は outputs/fg_points/index.html）
python tools/experiments/hires_prior/exp_hires_prior.py --coarse {oracle,pe} --out OUT.json [--variants ...]  # EdgeTAM の 128×128 特徴で事前マスクを高分解能にする案 H1〜H4 の測定（効果なし。結果は outputs/hires_prior/。exp_oracle_gt_prior.py は正解の 128×128 を事前マスクにした上限）
python tools/experiments/crack_fix/report_crack_fix.py  # クラス境界の割れの対処（前景内部の細い隙間を埋める）と BLAS のスレッド制限のレポート（outputs/crack_fix/）
python tools/experiments/small_objects/eval_lattice.py [--data DIR]  # 格子状に並べた物体の合成データ（make_lattice_data.py）で、変更前 / プロンプトのみ / 本体 を比べる
python tools/experiments/small_objects/make_lattice_data.py [--kinds plain --missing 0.05 --jitter 0.01 --out DIR]  # 部品を格子に並べた合成データ（正解付き）
python tools/experiments/small_objects/regress.py  # 変更前（元のプロンプト・格子分離なし）と現在の最終結果を、実画像・COCO128・D2S で比べる（一致 IoU・格子分離の作動）
python tools/experiments/small_objects/timing.py  # 変更前と現在の処理時間を、同じ画像で交互に測る（静かな状態で）
python tools/experiments/small_objects/cache_coarse.py && python tools/experiments/small_objects/prompt_candidates.py  # 前景プロンプトの候補の効果と副作用（粗い判定のキャッシュ screen.py で EdgeTAM なしに速く比べ、最終結果で確かめる）
python tools/experiments/small_objects/make_report_images.py  # outputs/small_objects/index.html 用の画像（006 の入力・正解・段階ごとの結果など）
python tools/experiments/semantic_merge/runall.py --name {current,orig,core,tie1.0,tie1_obj08,allone_post08,...} --sets single,typed,free [--suffix S]  # 1 つの物体が複数クラスに割れる問題（tube / transparent tubes など）の変種比較（PyTorch/OV 共通の本体を直接動かす。結果は outputs/semantic_merge/rows_<名前>.json）。`python tools/experiments/semantic_merge/compare.py 名前...` で表に。変種は variants.py / coarse_variants.py。`current` は本体の統合を切った基準、`core` が本体の既定
python tools/experiments/semantic_merge/regress_classes.py  # 実画像・COCO128・D2S でクラスの数・名前の変化（orig / 前回 / 現在）
python tools/experiments/semantic_merge/montage_classes.py --paths P... --out OUT.jpg --variants current,core  # クラス分けの比較画像
python tools/experiments/semantic_merge/timing.py  # 統合の有無の処理時間（OpenVINO）
python tools/experiments/semantic_merge/report_semantic_merge.py  # 上の結果を outputs/semantic_merge/index.html に可視化
python tools/visualizer/app.py [--open] [--port 8765]  # 画像の上の回転できる矩形ごとに、PE の特徴が前景・背景のどのプロンプトに近いかを見る Web GUI（物体のインスタンス検出の検討用。詳細は tools/visualizer/README.md。初回は `python convert_openvino.py --region-only`）
python tools/yoloe/app.py  # YOLOE（テキストプロンプトのゼロショット検出）を試すWeb GUI。CLIは tools/yoloe/yoloe.py。詳細は tools/yoloe/README.md
python tools/dataset/make_synthetic_dataset.py --parts DIR --backgrounds DIR --out DIR  # 部品の切り抜きを背景に合成し、マスク・個数の正解付きデータを作る
python tools/evaluation/eval_synthetic.py --data DIR  # 合成データでIoU・有無・個数を条件別に評価
python tools/experiments/speed/bench.py --tag NAME --images {small,cable,all} --save [--set key=value ...]  # 処理時間（捨て 2 回 + 計測の中央値）と出力（最終マップ・点・中間画像）の保存（結果は outputs/speed/NAME/）。変更前のコードのコピー（`git archive HEAD`）の中で実行すれば、基準も同じスクリプトで測れる（`--set model_dir=<絶対パス>`）。交互に実行して、ノート PC のばらつきを抑える
python tools/experiments/speed/compare.py BASE CAND [--verbose]  # 上の 2 つの結果を、`np.array_equal`（全項目）・最終マップの一致率・処理時間の比で比べる（出力が同じはずの高速化の確認用）
```

- 説明資料は `docs/`（**図を中心に、短く**書く。各節は「フロー図（mermaid）→ 実画像 → 数行 → 表」）: 全体像 `overview.md`（フロー図・処理のタイムライン図・処理時間・モジュール構成）・粗い段階 `coarse_segmentation.md`・詳細段階 `detailed_segmentation.md`・格子の検出方法 `lattice_detection.md`（シリンジの巣の画像つき。画像は `python tools/docs/make_lattice_doc_images.py` で `docs/images/latdet_*.jpg` に再生成。本体と一致するか確かめる）・手法の変遷と比較 `method_history.md`（採用 / 不採用の表）。画像は JPEG（幅 1200px 以下・合計 5MB 以下。git に載せるため）で、`python tools/docs/make_doc_images.py --image PATH --prefix NAME [--model-dir weights/openvino_models_l6]` で `docs/images/` に再生成し、`python tools/docs/make_timeline_image.py` でタイムライン図を作り、`tools/docs/lint_docs.py` で検査、`tools/docs/shrink_images.py` で縮める。処理を変えたら資料と画像も更新する。更新は `.claude/agents/docs-writer.md`（docs-writer エージェント）に任せられる。
- **高速化（2026-10-31。ケーブル画像 cable/000 が Ryzen 5 5500U で 約 783 → 472ms、出力は同じ）**: ① 静的形状（`ov_static_shape: true`）のデコーダー・エンコーダーのグラフをコンパイル前に書き換えて Reorder を除く（`core/openvino/graph_edits.py`。常に行う。デコード 1 回 約 16 → 9ms。`high_res_feats0` は内部で NHWC を持ち、外から見える形は NCHW のビュー（`OVModel` の `StaticPlan.input_perms` / `output_perms`）。動的形状のモデル（格子分離）は書き換えない。グラフが想定と違うモデルでは `RuntimeError`。`ov_static_shape: false` なら書き換えずに動く）。② 領域（クラス・取りこぼしの成分）ごとの推定を、`worker_threads`（既定 2 の Python のスレッド。`core/common/parallel.py` の `Workers` / `JobGroup`）で次のデコードと重ねる（`RegionEstimator.estimate` が `RefineInput` を読むだけで `RegionResult` を返し、`SemanticRefiner` が従来と同じ順に反映する。デコーダーは `EdgeTamSession` の lock で同時に 1 つ。取りこぼしの再推定は、クラス反映後に覆われていない成分を先に投機的に推定し、反映のときに覆われていたら捨てる）。EdgeTAM の kNN グラフ（`CoarseSegmenter.edge_graph_of`）も PE と重ねる。③ NumPy 段（`clean_mask`・`normalize_image`・`mean_feature` など。bit 一致）。OpenVINO のスレッド数は 4 のまま。再推定の反復を 1 回にする案は最終マップの一致率が 99.9% を割ったので採っていない。`Timer` の表示は `segment` の中では貯めて、終わってからまとめて出す（`Timer.deferred`）。詳細と効かなかった案は `docs/method_history.md` と `docs/overview.md`。**C++ 版（cppsrc/）は未追従**。
- 処理時間の内訳は `docs/overview.md`、高速化（BLAS のスレッド制限・NumPy 段の外接矩形処理）と効かなかった案は `docs/method_history.md`。
- 性能向上が確認できた手法（外周「不明」・外周 + ガイデッドフィルタ・PE と EdgeTAM の統合・点数 4:4）の詳細は `docs/method_history.md`。**外周 + ガイデッドフィルタ（精緻化の初期マスク）・点数 4:4・EdgeTAM の特徴での前景確信度の伝播（常に行う。以前の AND を置き換えた。設定キー `use_edgetam_clustering` は削除）・反復 2 回は本体に組み込み済み**（共通コードなので PyTorch・OpenVINO の両方）。実験用のコード（`tools/experiments/`）は、従来の運用（8:8・外周なし）を基準にしている。
- **前景の点を物体の上に選ぶ（2026-10-30）**: 前景のプロンプト点を、パッチの中心ではなく EdgeTAM の高解像度特徴量（128×128）の画素から、物体らしさ（core の特徴との類似度 − 背景の特徴との類似度）で重みづけした最遠点サンプリングで選ぶ（`core/common/object_points.py`。粗いマスクの縁のパッチも候補。パラメータ増なし）。MVTec +0.016・独立 +0.023・合成 +0.010 で、効きは toothbrush（細長く領域が分かれる物体）に偏る。背景の特徴が作れないときは従来のパッチ中心の点。従来の点の選び方（`mask_utils.prompt_points`）への切替は削除した（`tools/experiments/` の古い実験は、この点の選び方が基準になっている）。報告は `outputs/fg_points/index.html`。**C++ 版（cppsrc/）は未追従**。
- **006_syringe（透明な筒が格子に並ぶ巣）への対応（2026-10-27〜28）**: 前景プロンプト `a photo of transparent tubes.`（複数形が必須）と、格子状に並んだ小物体を 1 個ずつに分ける `core/common/lattice.py`・`lattice_separator.py`（`SemanticRefiner.refine` の最後。画像の勾配の自己相関で格子を検出したときだけ、セルごとの箱プロンプトで EdgeTAM を窓で切り出した特徴で一括デコード）を本体に組み込み済み（PyTorch・OpenVINO 共通）。006 の IoU 0.0 → 0.91。MVTec・独立 48 枚・合成は悪化なし。作動は巣のような画像に限る。報告は `outputs/small_objects/index.html`、説明は `docs/detailed_segmentation.md`「格子状に並んだ小物体の分離」。**C++ 版（cppsrc/）は未追従**。
- `tools/` は用途別のフォルダ: `evaluation/`（評価・診断）, `dataset/`, `gui/`, `experiments/<名前>/`（手法の検証実験。本体には未組み込みで、`outputs/<名前>/` と対）。実験どうしは `from tools.experiments.decontam import exp_decontam as E` の形で import し合う。構成は `tools/README.md`。評価カテゴリは `tools/evaluation/eval_prompts.py` の `OBJECT_CATEGORIES`（10 種。過去の数値はこれ）と `EVAL_CATEGORIES`（zipper を除く 9 種。zipper は fg_mask が画面の 78% で物体として扱う意味が無い）。
- 実行対象画像は `configs/dataset.yaml` の `image_paths`（`configs/dataset.sample.yaml` をコピーして作る。gitignore 対象）。`develop.py` / `test_pytorch.py` に CLI 引数は無い。
- 設定は `configs/ov_config.yaml`（OpenVINO）/ `configs/torch_config.yaml`（PyTorch）/ `configs/torch_config_sam2.yaml`（SAM2.1）。`core.utils.load_config()` の既定は ov_config。**コード側に設定の既定値は無い**（キーは yaml に必ず書く。`config[...]`）。`model_dir` は OpenVINO モデルのフォルダで、`convert_openvino.py` の保存先でもある。
- `convert_openvino.py` は冒頭で `SYSTEM_OPENVINO_DIR`（`params.py`。`/opt/intel/openvino`）を `sys.path`/`PYTHONPATH` から除去する（pip 版 openvino との衝突回避）。この処理を `import openvino` より前に保つこと。
- `weights/`, `outputs/`, `*.png`, `*.jpg` は gitignore 対象（`docs/images/*.jpg` だけ例外）。`tools/` も `tools/docs/` 以外は gitignore 対象（実験・評価のスクリプトは、git に載せず手元に置く運用）。

## アーキテクチャ

処理の本体は backend 非依存の `core/common/` にあり、モデル推論の「口」だけを `core/openvino/`（OpenVINO）と `core/torch/`（PyTorch）が持つ。**手法の開発は `core/common/` で行えば、両 backend に反映される。** `develop.py`（OpenVINO）と `test_pytorch.py`（PyTorch）は backend の生成部だけが違う。`core/torch/` は変換元モデルの定義も兼ねる（`convert_openvino.py` が `DecodeModel`, `RepVitModel`, `TorchCLIPFeatureExtractor` 等を import して変換）。`core/torch/sam2/` は SAM2/EdgeTAM のベンダリングコード。

`core/common/` の構成（torch / openvino を import しない）:

- `backend.py`: backend の契約（`CLIPBackend` / `EdgeTamBackend` の Protocol）。入出力はすべて `np.ndarray`。
- `edgetam_helper.py`: `EdgeTamHelper`。EdgeTAM の座標変換・プロンプト前処理・後処理（NumPy）。backend はこれを継承して `encode` / `decode` だけ実装する。
- `segmentation_runner.py`: `SegmentationRunner`。`segment(image)`（保存なしのパイプライン、`SegmentationResult` を返す）と `run(path)`（読み込み・`segment`・保存）。
- `coarse_segmentation.py`: `CoarseSegmenter`。CLIP 特徴量・テキストプロンプトとの照合・faiss KMeans による粗い前景/セマンティック推定（EdgeTAM の特徴での確信度の伝播・似たクラスのまとめを含む）。パッチ単位のラベル付けまで（整形は 2026-10-31 に削除。呼び出しが無効になっていた）。
- `semantic_refiner.py`: `SemanticRefiner`。精緻化の進行役: 準備 → クラスごとの推定 → 取りこぼしの再推定 → 後処理（小領域の除去・隙間埋め・格子分離）→ 元画像の大きさへ。領域の推定結果を、決まった順に `RefineState` へ反映する（スコアで重なりを解決して 1 枚のセマンティックマップにする）。
- `region_estimator.py`: `RegionEstimator`。1 領域（クラス・取りこぼしの成分）を EdgeTAM で推定する（前景点 → 初期マスク（外周「不明」+ ガイデッドフィルタ）→ デコードの反復 → 2 値マスクとスコア）。`RefineInput`（全領域に共通の入力。読むだけ）を受け取り何も書き換えないので、別スレッドで重ねて実行できる。
- `edgetam_session.py`: `EdgeTamSession`。EdgeTAM backend の呼び出し口（`encode` / `decode_prompts` / `decode_batch`）。デコーダーは lock で同時に 1 つ（格子分離のセルごとのデコードも通る）。
- `parallel.py`: `Workers`（スレッドプール。`worker_threads` が 0 なら、仕事は `result()` でその場で実行）と `JobGroup`（抜けるとき、始まっていない仕事を取り消す）。
- リサイズ: 画像は `image_size` 四方（アスペクト比を無視）にリサイズする（`core/utils.resize_image`）。`image_size` はパッチサイズ（16）と EdgeTAM の 32 の倍数であること（`SegmentationRunner` が検査）。プロンプト点・マスクは常に元画像の座標・大きさ。アスペクト比を保つモード（fit / pad）は 2026-10-31 に削除した。C++ 版（cppsrc/）も square のみ。格子分離・クラスのまとめ（`semantic_merge.py`）は C++ 版は未追従。
- `lattice.py` / `lattice_separator.py`: 規則的に並んだ同じ物体（巣・トレイ・ブリスター）の格子の検出（画像の勾配の自己相関。`detect`）と、セルごとの箱プロンプトによる物体の分離（`LatticeSeparator.separate`）。クラス単位のデコードは、隙間が 1 パッチ未満の物体の集まりを 1 つの塊にする（EdgeTAM は点に従い、事前マスクはほとんど効かない）ため。デコーダーが画像サイズ可変なので、特徴を箱の周りの窓（`LATTICE_WINDOW_TOKENS`）で切り出して全セルを一括実行する（80 個で約 95ms）。予測 IoU の下位 4 分位点が 0.5 未満なら何も変えない。定数は `params.py` の `LATTICE_*`。
- `semantic_merge.py`: 1 つの物体が似たプロンプトの別クラスに割れるのを防ぐ。`merge_near_ties`（PE のクラスタのロジットが僅差 `PROMPT_TIE_TOLERANCE` 以内のプロンプトを、より大きいクラスへ統合）、`merge_synonyms_in_objects`（1 つの連結成分の中の、テキスト埋め込みの距離 `PROMPT_SYNONYM_DISTANCE` 以内のクラスをまとめる。組は `synonym_representatives` で前景プロンプトから作る）。`CoarseSegmenter.generate_foreground_map` の伝播の後で呼ぶ（`tie_tolerance` / `synonym_representative` はインスタンスの属性で、実験で切れる）。
- `mask_utils.py`: 点サンプリング（`farthest_point_sampling` 等）・連結成分（`Component` = 最内点 `peak`・`core`・`region` の NamedTuple）・ノイズ除去・パッチ近傍の演算（`patch_kernel()`・`erode`・`dilate`・`squared_distance_to_background`。cv2。距離は二乗して整数に丸めるので、scipy の EDT と最内点が一致する）・kNN グラフの伝播などの純粋関数。節ごとに並べてある（型と数値 / 点のサンプリング / 連結成分とマスクの整形 / パッチ格子 <-> 作業解像度 / 伝播）。`prompt_points`（パッチ中心の前景点）は、`object_points` が使えないときの代わり。
- `object_points.py`: 前景のプロンプト点を、EdgeTAM の高解像度特徴量の画素から物体の上に選ぶ純粋関数（`normalize_features` → `background_features`（画像に 1 回）→ `objectness_map`（クラスごと）→ `object_prompt_points`）。`RegionEstimator.foreground_points` が呼ぶ。
- `params.py`: yaml に出さずコードに固定したハイパーパラメータ・定数（値を変えるときはここだけ編集。NumPy などを import しない定数だけのファイル。EdgeTAM / PE の正規化の平均と標準偏差・特徴量の縮小倍率 `EDGETAM_FEATURE_STRIDES`・ロジットの上限 `EDGETAM_LOGIT_CLIP`・OpenVINO 実行環境の定数・結果の保存と表示（色・一覧HTML）の定数もここ）。
- scipy を使うのは、FFT（`lattice.py`）・疎行列（kNN グラフ）・階層クラスタリング（`semantic_merge.py`）・隙間のラベル付き距離変換（`fill_narrow_gaps`）だけ。それ以外のマスク演算は cv2（出力は scipy 版と bit 一致）。
- `visualization.py` / `gallery.py` / `result.py`: 結果の画像の保存（`ResultSaver`。matplotlib の図だけメインスレッドで描き、他の PNG は並行して書く）/ 一覧HTML（`gallery.py`。ひな形は `gallery_template.html`）/ `SegmentationResult`。

backend:

- OpenVINO: `core/openvino/feature_extractor.py`（`OVCLIPFeatureExtractor`）, `edgetam_runner.py`（`OVEdgeTamRunner`）
  - **領域の代表トークンは 2 通り**（`custom_attentionpoollatent_forward(pool=)`）。`extract_cluster_tokens`（`pool="cls_attn"`。本体の粗い判定のクラスタごとのトークン）は CLS を必ず注意の対象に含め、CLS が注意の大半を受ける（領域のパッチ数 1〜4 で 99.6%・25〜99 で 90%・100 以上でも約 70%）。数十パッチ未満の小さな領域（矩形 1 つなど）は、どこでも画像全体の CLS とほぼ同じトークンになり、前景が背景になる。小さな領域は `extract_region_tokens`（`pool="mean"`。領域のパッチの value の平均。モデルは `pe_region_tokens_model`。他のモデルと同じく起動時に読み込む。無いと起動時に FileNotFoundError）を使う。本体の粗い判定は変えていない（12 枚で最終前景マスクが完全一致）。測定は `tools/visualizer/README.md`。**粗い判定の `extract_cluster_tokens` を `extract_region_tokens` に替えてはいけない**: 背景クラスタが前景と判定される割合が 2% → 27%（パッチ数で重み）に増え、粗い前景率が capsule 25→73%・transistor 27→87%・cable 44→68% になる（12 枚で前景→背景は 3 パッチだけ。一方向にずれる）。背景に素材・場面のプロンプト 10 個を足すと 27% → 12% に減るが直りきらず、画像ごとの最良のしきい値が −1.8〜+3.8 とばらつくので 1 つのオフセットでも直らない（2026-11-03、クラスタ単位 96 個で測定）
  - `core/openvino/runtime_utils.py`: 2 つの backend 共通の準備。**import するだけで `/opt/intel/openvino` を `sys.path` から外す**ので、各 backend ファイルは `import openvino` より前にこれを import する（順序を保つこと）。コンパイル設定 `make_compile_config` と INT8 の接尾辞 `model_file_suffix` もここ。各 backend の初期化は `limit_cpu_threads()`（OpenCV と BLAS のスレッド数の制限）を呼ぶ。**NumPy の BLAS のスレッド数は 1 に制限する（`limit_blas_threads`）。BLAS の既定（20 スレッド）は、行列積の後もスピン待ちして OpenVINO のスレッドとコアを取り合い、直後の推論を 2〜4 倍遅くする（処理時間が約 1/1.5 に）。`docs/method_history.md`**
  - `core/openvino/compiled_model.py`: `OVModel`（モデル 1 つの読み込みと実行。出力はモデルの出力の順のリスト）。config の `ov_static_shape: true` で、PE・EdgeTAM エンコーダー・デコーダーを `image_size` 四方に固定（読み込み時に reshape。**`convert_openvino.py` で変換し直す必要は無い**。`StaticPlan` で指定）してコンパイルする。約 23ms（約 12%）速くなるが、出力が動的形状と bit 単位では一致しない（MVTec・COCO・合成の 53 枚で、最終マップは 52 枚が完全一致・残り 1 枚も不一致画素 0.01%、クラスの変化なし）。形が合わない入力（格子分離のセルごとのデコード）は、動的形状のモデルで推論する（初めて必要になったときにコンパイルする）。
- PyTorch（実験用）: `core/torch/feature_extractor.py` の `TorchCLIPProbe`。全層の hidden・attention 出力を取り出し、後半ブロックを attention マスク付きで再実行できる。`tools/experiments/decontam/exp_decontam*.py` が使う。
- PyTorch（SAM2.1）: `core/torch/sam2_runner.py` の `TorchSam2Backend`。EdgeTAM のクラス（`RepVitModel` / `DecodeModel` / `TorchEdgeTamBackend`）を継承し、モデルの取得だけを公式 SAM2.1 に差し替える（デコーダーの入出力と特徴量の形状が同じなので、`SegmentationRunner(extractor, TorchSam2Backend(config), config)` でそのままつながる）。`core/torch/sam2_official/` は公式 SAM2.1 のコピー（パッケージ名 `sam2_official`、`UPSTREAM.md` 参照）。`core/torch/sam2/` は EdgeTAM の fork で公式とは別物（同名のパッケージは同時に置けない）。
- PyTorch: `core/torch/feature_extractor.py`（`TorchCLIPBackend`。変換済みモデルと同じグラフ `to_openvino=True` で動かし、出力を ndarray 化）, `edgetam_runner.py`（`TorchEdgeTamBackend`）

パイプライン（`SegmentationRunner.segment`）:

1. `resize_image` + `normalize_image`（リサイズは CLIP と EdgeTAM で共通、正規化はモデルごと）
2. EdgeTAM エンコーダーを先に 1 回だけ実行（結果は粗い判定と精緻化の両方で使う）。`CoarseSegmenter.extract_features`: CLIP 画像特徴（パッチ・fused パッチ）
3. `generate_foreground_map`: テキストプロンプト（`prompts_dir` の `foreground.yaml` / `background.yaml`）との類似度 + faiss KMeans のクラスタで粗い前景マスクとクラスタ分割を得る。EdgeTAM の `image_embed` の類似度（kNN）で前景確信度を伝播して揃える（誤検出を減らす。`docs/coarse_segmentation.md` §4）
   その後、`merge_near_ties` / `merge_synonyms_in_objects` で、1 つの物体の似たプロンプト（tube と transparent tubes など）のクラスをまとめる（別の部品を指す noodle / package などは保つ。`docs/coarse_segmentation.md` §5）。**前景プロンプトを足したら、前景マスクだけでなくクラスの数・名前も確認する**（`transparent tubes` の追加で cap の画像が 2 クラスに割れた）
4. `assign_semantic_labels`: クラスタに対応するプロンプトで粗いセマンティックラベルを付与
5. `SemanticRefiner.refine`: 前景（クラスごと `edgetam_foreground_points_per_class` 点、既定 4。EdgeTAM の高解像度特徴量の画素から物体らしさで選ぶ。`object_points.py`）/ 背景（`edgetam_background_point_count` 点、既定 4）のプロンプト点と、初期マスク（粗いマスクの外側 1 パッチを「不明」にし、128×128 に拡大して入力画像の輝度をガイドにしたガイデッドフィルタで縁を整えたもの）で EdgeTAM の decoder を反復し（`RegionEstimator.estimate`）、クラスを統合し、取りこぼしの成分を再推定し、前景内部の幅 1 パッチ未満の隙間（クラス境界の割れ）を埋める（`mask_utils.fill_narrow_gaps`。encoder は 2 の結果を使う）。最後に、物体が格子状に並んでいれば、セルごとに 1 個ずつに分ける（`LatticeSeparator`。作動は稀で、約 +150ms。判定は毎回 約 6ms）
6. `run()` では `ResultSaver` が `output_dir/<画像名>/` に可視化を保存（`Inference Time` には含まない。matplotlib の図だけメインスレッド、他の PNG はスレッドで並行して書く）

`core/utils.py` に共通ヘルパー（`Timer` コンテキストマネージャー、`load_image`, `resize_image`, `normalize_image`, `load_config`）。各処理は `with Timer(...)` で ms 計測する流儀。

## 補足

- 座標系: 入力画像 → CLIP パッチ特徴マップ（`image_size`/patch）→ EdgeTAM の座標の変換は `RegionEstimator.transform_coords` / `EdgeTamHelper.transform_coords` を使う。
- クラスタリングに使う PE の層は config の `cluster_layers`（既定 `[8, 9, 10]`）。OpenVINO 版は変換時に固定される（`convert_openvino.py --cluster-layers`）。第 6 層のみのモデルは `weights/openvino_models_l6` に作成済み（切り替えは `configs/ov_config.yaml` の `model_dir`）。
- INT8 モデルは `et_use_int8` / `pe_use_int8` で切替（EdgeTAM / PE それぞれ）。
- データセットのパス（`/home/llm/datasets/...`）は環境固有でリポジトリには含まれない。
