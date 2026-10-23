# セグメンテーションの手法（現状のまとめ）

テキストプロンプトだけで（学習なしで）、画像から「何がどこにあるか」を求める。
**①粗く当たりを付け → ②細かく切り出す**の2段階。経緯・実験結果は [coarse_segmentation.md](coarse_segmentation.md) / [semantic_refiner.md](semantic_refiner.md) / [improvement_methods.md](improvement_methods.md)。

## 全体像

```mermaid
flowchart TD
    IMG[画像] --> PRE[リサイズ 512 + 正規化]
    PRE --> C

    subgraph C["① 粗いセグメンテーション（PE）"]
        direction TB
        C1[パッチ特徴] --> C2[KMeans k=8]
        C2 --> C3[クラスタごとにテキストと照合]
        C3 --> C4[整形: 穴埋め・細い領域の統合]
    end

    C -->|"前景マスク / 前景らしさ / 粗いラベル"| R

    subgraph R["② 詳細セグメンテーション（EdgeTAM）"]
        direction TB
        R1[クラスごとに点 + 初期マスクで切り出す] --> R2[重なりをスコアで解決]
        R2 --> R3[取りこぼしを再推定]
        R3 --> R4[ノイズ除去]
    end

    R --> OUT["semantic_map<br/>0=背景 / 1〜=前景クラス"]
```

コード: `core/common/segmentation_runner.py` の `segment`。各処理は `with Timer(...)` で計測。

## ① 粗いセグメンテーション（`coarse_segmentation.py`）

```mermaid
flowchart TD
    A[PE のパッチ特徴] --> B["KMeans で k=8 クラスタに分ける"]
    B --> C["クラスタの代表ベクトルを作る"]
    C --> D["前景/背景プロンプトとの類似度<br/>各グループの最大値を比較（softmax）"]
    D --> E{"前景確信度 > 0.5 ?"}
    E -->|はい| F["前景。最も近い前景プロンプトがラベル"]
    E -->|いいえ| G["背景。最も近い背景プロンプトがラベル"]
    F --> H["前景らしさマップ / 前景マスク / 粗いラベル"]
    G --> H
    H --> I["整形<br/>・前景に囲まれた背景は前景にする<br/>・幅2パッチ以下のクラス領域は接するクラスに統合"]
```

### オプション: EdgeTAM の特徴でも判定して AND（`use_edgetam_clustering`）

```mermaid
flowchart LR
    P["PE の特徴 → KMeans → 判定"] --> AND{"両方が前景のパッチだけ前景<br/>（確信度の小さい方を採る）"}
    E["EdgeTAM の特徴 → KMeans → 判定<br/>（PE 特徴をクラスタごとにまとめて照合）"] --> AND
```

| | |
|---|---|
| 効果 | 背景の斑点・織り目の誤検出が半減（合成データ IoU +0.05） |
| 代償 | 小物体の検出率が少し低下。大きな単体物体（cable など）では下がることがある |
| 注意 | EdgeTAM のエンコードは1回だけ実行し、②にも使い回す |

## ② 詳細セグメンテーション（`semantic_refiner.py`）

### 全体

```mermaid
flowchart TD
    S0["準備（全クラス共通）<br/>・作業解像度 長辺 ≤1024<br/>・負例点4点: 端につながる背景のうち前景から離れ、前景確信度が最も低い所"]
    S0 --> L

    subgraph L["クラスごと（粗いマスクの連結成分ごとに点を置く）"]
        direction TB
        L1["正例点4点 + 負例点 + 初期マスクを作る"] --> L2["EdgeTAM で3回反復<br/>（前回のロジットを次の初期マスクに）"]
        L2 --> L3["ロジット > 0 を前景に → 小さな成分を除去"]
        L3 --> L4["スコアを計算"]
    end

    L --> M["重なる画素はスコアが高いクラスに"]
    M --> N{"最内点がどのクラスにも<br/>覆われていない成分がある?"}
    N -->|はい| O["その成分だけで再推定<br/>（極小成分は1パッチ広げた範囲まで）"]
    N -->|いいえ| P
    O --> P["小領域ノイズ除去 → 元画像サイズへ"]
```

### 入力の作り方

```mermaid
flowchart LR
    subgraph POS["正例点（クラスごと4点）"]
        direction TB
        P1["各連結成分の最内点を必ず含める"] --> P2["残りは成分の内側から最遠点サンプリング"]
    end

    subgraph PRIOR["初期マスク"]
        direction TB
        M1["内側 core = 前景 (p=1)"] --> M2["粗いマスクの外側1パッチまで = 不明 (p=0.5)<br/>それより外 = 背景 (p=0)"]
        M2 --> M3["入力画像の輝度でガイデッドフィルタ<br/>→ 不明の縁を画像の縁に沿わせる"]
        M3 --> M4["logit = (2p-1) × 8"]
    end
```

斑点（3パッチ未満の孤立成分）はクラスの推定に混ぜない（物体が背景へ広がるのを防ぐ）。

### スコア（重なりの解決用）

```
スコア = EdgeTAM の確信度 × (0.5 + 0.5 × 粗い前景らしさ)
```

## 設定

| キー（`configs/ov_config.yaml`） | 値 | 意味 |
|---|---|---|
| `image_size` | 512 | リサイズ後の一辺 |
| `k_clusters` / `n_iter` | 8 / 10 | KMeans |
| `foreground_threshold` | 0.5 | 前景の確信度の閾値 |
| `prompts_dir` | `prompts/candidates/color_free_bg` | 前景・背景プロンプト |
| `use_edgetam_clustering` | false | EdgeTAM 特徴でも判定して AND |
| `edgetam_foreground_points_per_class` | 4 | クラスごとの正例点 |
| `edgetam_background_point_count` | 4 | 負例点 |
| `edgetam_refine_iterations` | 3 | EdgeTAM の反復回数 |
| `semantic_max_side` | 1024 | マスク演算の作業解像度（長辺） |
| `ov_num_thread` | 4 | OpenVINO のスレッド数 |

コードに固定した値（近傍3・logit幅8・スコア重み0.5 など）は `core/common/params.py`。

## 処理時間の目安（OpenVINO・FP32・4スレッド）

[improvement_methods.md](improvement_methods.md) §9.3 の測定値（`use_edgetam_clustering: true`、中央値）。

```mermaid
flowchart LR
    A["PE<br/>86ms"] --> B["EdgeTAM<br/>エンコーダ<br/>34ms"] --> C["粗い判定<br/>11ms"] --> D["デコード<br/>40ms"] --> E["NumPy 後処理<br/>26ms"] --> F["合計 約206ms"]
```

## コードの対応

| 処理 | ファイル |
|---|---|
| 全体の流れ | `core/common/segmentation_runner.py` |
| ① 粗い推定 | `core/common/coarse_segmentation.py` |
| ② 精緻化 | `core/common/semantic_refiner.py` |
| 点の選択・連結成分・ガイデッドフィルタ | `core/common/mask_utils.py` |
| モデル推論の口 | `core/openvino/` / `core/torch/` |
