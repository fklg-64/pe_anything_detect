# =============================================================================
# 固定ハイパーパラメータ
# yamlでは調整せずコードに固定している値を、すべてここにまとめる。
# 値を変える場合はこのファイルだけを編集すること（他の箇所に数値を直書きしない）。
# 可視化の見た目だけに関わる値は visualization.py にある。
# =============================================================================

import numpy as np


# --- CLIP（粗い前景・セマンティックの判定） ---
# テキストとの類似度(cos)に掛ける係数（softmaxの温度の逆数）
CLIP_LOGIT_SCALE = 100.0

# --- KMeans ---
KMEANS_SEED = 42
FAISS_NUM_THREADS = 2

# --- OpenCV ---
# OpenVINO backend が起動時に cv2.setNumThreads へ渡すスレッド数
OPENCV_NUM_THREADS = 2

# --- EdgeTAM の特徴での前景確信度の伝播（粗い判定。use_edgetam_clustering） ---
# EdgeTAM の image_embed のコサイン類似度が上位の k 個のパッチを近傍とし、前景確信度を conf ← (1-α)·conf0 + α·(近傍の平均) で伝播する。
# α は 0.7〜0.9、k は 4〜8 で結果が同程度（平坦な範囲から選んだ）。反復は 3 回で十分（5 回でも変わらない）
EDGE_PROPAGATION_ALPHA = 0.8
EDGE_PROPAGATION_NEIGHBORS = 8
EDGE_PROPAGATION_ITERATIONS = 3

# --- パッチ単位の幾何（粗い結果はCLIPのパッチ格子上にある） ---
# 近傍の大きさ。「内側」「細い領域」の判定（この近傍で縮めて残るか）と、粗いマスクの平滑化に使う。
# 3 = 1パッチ分の近傍。粗いマスクは縁・影で最大1パッチ太るという前提に基づく
PATCH_NEIGHBORHOOD = 3
PATCH_STRUCTURE = np.ones((PATCH_NEIGHBORHOOD, PATCH_NEIGHBORHOOD), dtype=bool)
# 背景の負例点を前景から離す距離（パッチ数）。縁・影の太り(1パッチ)を避けるため2
BACKGROUND_POINT_MARGIN = 2
# ノイズとして除去する領域の最小面積（パッチ面積に対する倍率）。1パッチ未満は粗いマスクの分解能以下
MIN_REGION_PATCH_AREA = 1.0

# 前景クラスタに混ざる斑点を除くための、孤立成分の最小パッチ数。これ未満の孤立成分はクラスの推定に
# 混ぜず、個別に範囲を制限して扱う（1〜2パッチの孤立成分は、物体より背景の斑点である場合が多い）
MIN_ISOLATED_PATCHES = 3

# --- EdgeTAMに渡す初期マスク（粗いマスクを事前確率として与える） ---
# 初期マスクのロジット幅。確率 p のとき logit = (2p - 1) * PRIOR_LOGIT_RANGE （p=0.5が「不明」で0）
PRIOR_LOGIT_RANGE = 8.0
# core（物体内部）の外側にある粗いマスクに与える確率。0.5=前景とも背景とも言えない
PRIOR_NON_CORE_PROBABILITY = 0.5
# 初期マスクの縁を入力画像に沿わせるガイデッドフィルタ（He ら 2010）の正則化。ガイドは入力画像の輝度（0〜1）で、半径は粗いマスクの信頼限界（1パッチ）。
# 値は 0.001〜0.01 で結果が変わらない（0.1 では下がる）
GUIDED_FILTER_EPS = 1e-2
# EdgeTAMのロジットがこの値を超えた画素を前景とする
MASK_LOGIT_THRESHOLD = 0.0
# EdgeTAMのデコーダ出力マスクを縮小する倍率（分母。高さ・幅を 1/N にする）
EDGETAM_MASK_DOWNSCALE = 8

# --- 重なり解決 ---
# スコア = EdgeTAMの確信度 × ((1 - w) + w × 粗い前景らしさ) の w。0.5で両者を等しく扱う
SCORE_PRIOR_WEIGHT = 0.5

# --- 前景内部の細い隙間の穴埋め ---
# 粗い前景の内部（core）にある背景の隙間は、幅が 1 パッチ未満なら粗い判定の分解能以下なので前景にする（クラス境界の割れ。docs/semantic_refiner.md §6.4）。
# 隙間は 1 パッチ = この画素数に縮小したマスクの閉処理で探す（全解像度の閉処理は約 10ms かかるため。位置は ±1 画素程度の差）
GAP_FILL_PIXELS_PER_PATCH = 8

# --- ノイズ除去 ---
# マスクの輪郭を平滑化するメディアンフィルタのカーネルサイズ
MEDIAN_BLUR_KSIZE = 3

# --- インスタンス分離（instance_split。同じクラスの接触した物体を 1 個ずつに分ける。docs/instance_segmentation.md） ---
# 画像面積に対する、インスタンスの最小面積の割合（これ未満は捨てる）
INSTANCE_MIN_AREA_RATIO = 0.0002
# 距離変換の watershed を、この倍率に縮小して行う（全解像度では重いため）
INSTANCE_EDT_SCALE = 0.25
# 過分割に使う距離変換の山の高さ（領域の最大半径に対する割合）。小さいほど細かく分ける。どちらの方式も、ここで分けた断片が出発点になる
INSTANCE_OVERSEG_H_FRAC = 0.1
# 提案方式（proposals）: 提案のマスクのうち、クラスの領域に収まる割合の下限 / 既に採った物体に覆われていない新しい面積の割合の下限 / 重複とみなす IoU
INSTANCE_PROPOSAL_INSIDE = 0.85
INSTANCE_PROPOSAL_NEW_FRACTION = 0.5
INSTANCE_PROPOSAL_NMS_IOU = 0.5
# 結合方式（merge）: 隣り合う断片の一方を指したときのマスクが、もう一方の面積をこの割合以上覆えば同じ物体の候補。
# かつ、断片の境界の長さが小さい方の直径のこの割合以上（くびれが無い）なら結合する
INSTANCE_MERGE_COVER = 0.5
INSTANCE_MERGE_NECK = 0.7
