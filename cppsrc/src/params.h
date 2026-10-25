// =============================================================================
// 固定ハイパーパラメータ（core/common/params.py の C++ 版）
// yaml では調整せずコードに固定している値を、すべてここにまとめる。
// 値を変える場合はこのファイルだけを編集すること（他の箇所に数値を直書きしない）。
// =============================================================================
#pragma once

namespace pes::params {

// --- CLIP（粗い前景・セマンティックの判定） ---
// テキストとの類似度(cos)に掛ける係数（softmaxの温度の逆数）
constexpr float CLIP_LOGIT_SCALE = 100.0f;
// テキストエンコーダの入力長（pe_text_model.xml の入力形状 [N, 32] と同じ）
constexpr int TEXT_CONTEXT_LENGTH = 32;

// --- KMeans ---
constexpr int KMEANS_SEED = 42;

// --- OpenCV ---
// モデル読み込み時に cv::setNumThreads へ渡すスレッド数
constexpr int OPENCV_NUM_THREADS = 2;

// --- EdgeTAM の特徴での前景確信度の伝播（粗い判定。use_edgetam_clustering） ---
// EdgeTAM の image_embed のコサイン類似度が上位の k 個のパッチを近傍とし、前景確信度を conf ← (1-α)·conf0 + α·(近傍の平均) で伝播する。
constexpr double EDGE_PROPAGATION_ALPHA = 0.8;
constexpr int EDGE_PROPAGATION_NEIGHBORS = 8;
constexpr int EDGE_PROPAGATION_ITERATIONS = 3;

// --- パッチ単位の幾何（粗い結果はCLIPのパッチ格子上にある） ---
// 近傍の大きさ。「内側」「細い領域」の判定（この近傍で縮めて残るか）と、粗いマスクの平滑化に使う。
constexpr int PATCH_NEIGHBORHOOD = 3;
// 背景の負例点を前景から離す距離（パッチ数）。縁・影の太り(1パッチ)を避けるため2
constexpr int BACKGROUND_POINT_MARGIN = 2;
// ノイズとして除去する領域の最小面積（パッチ面積に対する倍率）
constexpr double MIN_REGION_PATCH_AREA = 1.0;
// 前景クラスタに混ざる斑点を除くための、孤立成分の最小パッチ数
constexpr int MIN_ISOLATED_PATCHES = 3;

// --- EdgeTAMに渡す初期マスク（粗いマスクを事前確率として与える） ---
// 初期マスクのロジット幅。確率 p のとき logit = (2p - 1) * PRIOR_LOGIT_RANGE
constexpr float PRIOR_LOGIT_RANGE = 8.0f;
// core（物体内部）の外側にある粗いマスクに与える確率。0.5=前景とも背景とも言えない
constexpr float PRIOR_NON_CORE_PROBABILITY = 0.5f;
// 初期マスクの縁を入力画像に沿わせるガイデッドフィルタの正則化
constexpr float GUIDED_FILTER_EPS = 1e-2f;
// EdgeTAMのロジットがこの値を超えた画素を前景とする
constexpr float MASK_LOGIT_THRESHOLD = 0.0f;
// EdgeTAMのデコーダ出力マスクの値の範囲（postprocess の clip）
constexpr float MASK_LOGIT_CLIP = 32.0f;

// --- 重なり解決 ---
// スコア = EdgeTAMの確信度 × ((1 - w) + w × 粗い前景らしさ) の w
constexpr float SCORE_PRIOR_WEIGHT = 0.5f;

// --- 前景内部の細い隙間の穴埋め ---
// 隙間は 1 パッチ = この画素数に縮小したマスクの閉処理で探す
constexpr int GAP_FILL_PIXELS_PER_PATCH = 8;

// --- ノイズ除去 ---
// マスクの輪郭を平滑化するメディアンフィルタのカーネルサイズ
constexpr int MEDIAN_BLUR_KSIZE = 3;

}  // namespace pes::params
