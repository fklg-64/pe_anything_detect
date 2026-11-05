# =============================================================================
# 固定ハイパーパラメータ
# yamlでは調整せずコードに固定している値を、すべてここにまとめる。
# 値を変える場合はこのファイルだけを編集すること（他の箇所に数値を直書きしない）。
# 結果の画像・一覧HTMLの見た目だけに関わる値は、末尾の「結果の保存・表示」にある。
# =============================================================================

# --- CLIP（粗い前景・セマンティックの判定） ---
# テキストとの類似度(cos)に掛ける係数（softmaxの温度の逆数）
CLIP_LOGIT_SCALE = 100.0

# --- KMeans ---
KMEANS_SEED = 42
FAISS_NUM_THREADS = 2

# --- OpenCV ---
# OpenVINO backend が起動時に cv2.setNumThreads へ渡すスレッド数
OPENCV_NUM_THREADS = 2

# --- EdgeTAM の特徴での前景確信度の伝播（粗い判定） ---
# EdgeTAM の image_embed のコサイン類似度が上位の k 個のパッチを近傍とし、前景確信度を conf ← (1-α)·conf0 + α·(近傍の平均) で伝播する。
# α は 0.7〜0.9、k は 4〜8 で結果が同程度（平坦な範囲から選んだ）。反復は 3 回で十分（5 回でも変わらない）
EDGE_PROPAGATION_ALPHA = 0.8
EDGE_PROPAGATION_NEIGHBORS = 8
EDGE_PROPAGATION_ITERATIONS = 3

# --- 前景のクラスの付け直し（region_relabel） ---
# 前景のパッチを PE + EdgeTAM の特徴で KMeans する「ブロック」の数。ブロックごとの代表トークン（CLS なし）と前景プロンプトの照合が、パッチごとのクラスの根拠になる。
# 細かいほど、種類の違う物体が別のクラスになる（種類別合成 120 枚の V: 8 個 0.424 → 32 個 0.563 → 128 個 0.609）。代わりに、1 つの物体が複数クラスに割れる画像も増える
# （単一物体 351 枚で 63 → 67 → 68 枚）。前景のマスクは変えず、前景 IoU は平均で変わらない。処理時間は 8 → 32 で約 +1ms（検証: tools/experiments/fine_coarse/）。
# 判定し直し（rejudge_regions）が「小さすぎて独立した根拠を持てない領域」を決める面積の基準は、ブロック数と別に k_clusters（前景の 1/k_clusters）のまま
# （ブロック数に連動させると、小さな領域に別の名前が付いて割れが増える）
RELABEL_BLOCK_COUNT = 32

# --- 画像の正規化（RGB。(画素値/255 - 平均) / 標準偏差） ---
EDGETAM_MEAN = (0.485, 0.456, 0.406)
EDGETAM_STD = (0.229, 0.224, 0.225)
PE_MEAN = (0.5, 0.5, 0.5)
PE_STD = (0.5, 0.5, 0.5)

# --- モデル入力の大きさ ---
# EdgeTAM エンコーダの neck は最も粗い特徴（1/32）を 2 倍に拡大して 1/16 の特徴に足すので、入力の辺は 32 の倍数でないと大きさが合わず落ちる
EDGETAM_INPUT_ALIGNMENT = 32

# --- パッチ単位の幾何（粗い結果はCLIPのパッチ格子上にある） ---
# 近傍の大きさ。「内側」（core。この近傍で縮めて残る所）の判定と、初期マスクの「不明」の帯（この近傍で膨らませる）、粗いマスクの平滑化に使う。
# 3 = 1パッチ分の近傍。粗いマスクは縁・影で最大1パッチ太るという前提に基づく
PATCH_NEIGHBORHOOD = 3
# 背景の負例点を前景から離す距離（パッチ数）。縁・影の太り(1パッチ)を避けるため2
BACKGROUND_POINT_MARGIN = 2
# 前景クラスタに混ざる斑点を除くための、孤立成分の最小パッチ数。これ未満の孤立成分はクラスの推定に
# 混ぜず、個別に範囲を制限して扱う（1〜2パッチの孤立成分は、物体より背景の斑点である場合が多い）
MIN_ISOLATED_PATCHES = 3

# --- EdgeTAMに渡す初期マスク（粗いマスクを事前確率として与える） ---
# 初期マスクのロジット幅。確率 p のとき logit = (2p - 1) * PRIOR_LOGIT_RANGE （p=0.5が「不明」で0）
PRIOR_LOGIT_RANGE = 8.0
# 初期マスクの縁を入力画像に沿わせるガイデッドフィルタ（He ら 2010）の正則化。ガイドは入力画像の輝度（0〜1）で、半径は粗いマスクの信頼限界（1パッチ）。
# 値は 0.001〜0.01 で結果が変わらない（0.1 では下がる）
GUIDED_FILTER_EPS = 1e-2
# EdgeTAMのデコーダ出力（低解像度ロジット）を、絶対値がこの値以下に収める（外れ値で補間・比較が荒れないように）
EDGETAM_LOGIT_CLIP = 32.0
# EdgeTAM エンコーダの出力 (high_res_feats0, high_res_feats1, image_embed) の、入力画像に対する縮小倍率（縦横それぞれ 1/N。512 入力で 128・64・32）。
# 初期マスク・ガイデッドフィルタのガイドは high_res_feats0 の大きさ。デコーダーの出力（低解像度ロジット）の縮小倍率も high_res_feats0 と同じ
EDGETAM_FEATURE_STRIDES = (4, 8, 16)

# --- 重なり解決 ---
# スコア = EdgeTAMの確信度 × ((1 - w) + w × 粗い前景らしさ) の w。0.5で両者を等しく扱う
SCORE_PRIOR_WEIGHT = 0.5

# --- 前景内部の細い隙間の穴埋め ---
# 粗い前景の内部（core）にある背景の隙間は、幅が 1 パッチ未満なら粗い判定の分解能以下なので前景にする（クラス境界の割れ。docs/detailed_segmentation.md「5. 後処理」）。
# 隙間は 1 パッチ = この画素数に縮小したマスクの閉処理で探す（全解像度の閉処理は約 10ms かかるため。位置は ±1 画素程度の差）
GAP_FILL_PIXELS_PER_PATCH = 8

# --- ノイズ除去 ---
# マスクの輪郭を平滑化するメディアンフィルタのカーネルサイズ
MEDIAN_BLUR_KSIZE = 3
# 小さな連結成分が多いとき（この数を超えるとき）は、成分ごとに消すより、全画素を表で引く方が速い（clean_mask）
CLEAN_MASK_MAX_PER_COMPONENT = 48

# --- 規則的に並んだ小物体（格子）の分離（lattice.py / lattice_separator.py） ---
# 物体どうしの隙間が 1 パッチ未満だと、粗い前景は 1 つの塊になり、EdgeTAM もクラス単位のデコードでは点に従って塊を返す（隙間が埋まる）。
# 画像の勾配の自己相関で格子（トレイ・巣・ブリスターなど）を見つけたら、セルごとの箱で物体を 1 個ずつデコードして分ける
# 格子の検出（自己相関）をする画像の長辺（画素）。毎回実行する判定なので小さくする（256 で約 11ms、512 で約 60ms）。通ったときだけ画像の大きさで精密化する
LATTICE_DETECT_SIZE = 256
# 格子と判定する自己相関（格子ベクトル v と 2v の小さいほう）の下限。格子（syringe の巣）は 0.43〜0.8、格子でない画像（MVTec・合成・COCO・D2S・実画像）は 0.32 以下
LATTICE_MIN_STRENGTH = 0.4
# 前景に入るセルの数の下限。これより少ない物体は、隙間が 1 パッチ以上あれば分かれるので、クラス単位のデコードで足りる
LATTICE_MIN_CELLS = 16
# 格子ベクトルの最小の長さ（入力画像の画素）。これより短い（0.75 パッチ未満）ピークは、物体の縁の二重線などで、物体の間隔ではない
LATTICE_MIN_LAG = 12
# 自己相関のピークを探す近傍の半径（自己相関の画素）
LATTICE_PEAK_RADIUS = 7
# 格子ベクトルの候補にするピークの強さ（最も強いピークに対する比）。この比以上のピークのうち最も短いものを選ぶ（2v・3v のピークが最も強くても、v を選べる）
LATTICE_PEAK_RATIO = 0.6
# セルのテンプレート（格子の周期で重ねた物体らしさ）の分割数（1 辺）
LATTICE_PHASE_BINS = 16
# セルごとのデコードで EdgeTAM の特徴を切り出す窓の大きさ（image_embed のトークン数。1 トークン = 16 画素）。窓が大きいほど正確（8: IoU 0.88, 12: 0.91, 16: 0.92, 全体 0.94）だが遅い
LATTICE_WINDOW_TOKENS = 12
# 格子の位相（物体の中心）を、デコード結果の重心で直すときに使うセルの数の上限（全セルではなく、等間隔に選んだ一部で足りる）
LATTICE_PHASE_PROBES = 16

# デコーダーの予測 IoU（マスクと正解の IoU の予測値）が、この値以上のマスクを物体として採る。0.5 は、物体と一致した（マッチした）とみなす IoU の基準
LATTICE_MIN_PREDICTED_IOU = 0.5

# --- インスタンス分離（instance_splitter.py。config の instance_split が真のときだけ動く。同じクラスの接触した物体を 1 個ずつに分ける。docs/instance_segmentation.md） ---
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

# --- 密集した長方形の物体のインスタンス分離（rect_size.py / rect_tiling.py / rect_separator.py。config の dense_instances が真のときだけ動く） ---
# 向き・大きさが混在して接した細長い長方形の物体（積んだ荷物・並べた部品）を 1 個ずつに分ける。学習はしない。物体の大きさ（長辺 L・短辺 S）は画像に 1 つと仮定し、
#   ① 隣り合う平行な縁の間隔のヒストグラムから L・S を求め（rect_size.py）、② その大きさの長方形を前景に敷き詰め（rect_tiling.py）、
#   ③ 長方形のマスク + 軸に平行な箱 + 中心の点で EdgeTAM をデコードして 1 個ずつ切り出す（rect_separator.py）。
# 値は tools/experiments/dense_instances/ で選んだ（合成の積み荷 60 枚（開発 36・評価 24）と実画像 5 枚で選び、設定を固定してから独立した合成 30 枚で確かめた。
# ただし RECT_DOUBLING は、独立した 30 枚の結果も見てから選んだ。報告: outputs/dense_instances/mixed.html 9 章）。
# 長い直線の縁を持つ細長い長方形向き。散らばった小さい部品（短辺 24〜41 画素）では大きさが当たらない。画素数は、すべて作業解像度 image_size = 512 の画素
RECT_IMAGE_SIZE = 512

# ① 大きさ: 隣り合う平行な縁の間隔のヒストグラム
# 画像を回す向きの刻み（度）。RECT_SCAN_HALF・RECT_SCAN_ANGLE_STEP と合わせて時間と精度を選ぶ（合成 90 枚 + 実画像 5 枚の F1 と、本体込みの処理時間）:
#   5 度刻み・走査 128 画素 / 15 度（C1）0.805・633ms、7.5・96 / 15（C2）0.785・554ms、10・96 / 20（C3。既定）0.775・504ms、10・走査なし（C5）0.760・404ms
RECT_GAP_ANGLE_STEP = 10.0
# 縁の強さを求める前の平滑化（ガウシアンの標準偏差。画素）
RECT_GRADIENT_SIGMA = 1.0
# 前景（縁が前景の上にあるか調べる範囲）を膨らませる正方形の一辺（画素）。物体の縁は前景の外周にあるので、少し広げる
RECT_FOREGROUND_DILATE = 5
# 縦に続く縁だけを残す最小値フィルタの長さ（画素）。S 用は長辺どうしの継ぎ目（長い）だけ、L 用は短辺の端（短い）まで数える。256 に縮小して数えると S の当たりが 0.85 に落ちる
RECT_SHORT_EDGE_LINE = 33
RECT_LONG_EDGE_LINE = 9
# 行の上の縁の極大を探す幅（画素）
RECT_EDGE_NMS = 9
# 数える縁の間隔の範囲（画素）。下限 RECT_MIN_GAP 未満は縁の二重線などで、物体の大きさではない
RECT_MIN_GAP = 16
RECT_MAX_GAP = 300
# ヒストグラムの平滑化（標準偏差。間隔の画素）
RECT_HISTOGRAM_SMOOTH_SIGMA = 1.0
# S: ヒストグラムの極大のうち、最大のこの割合以上のものの中で最も小さい間隔（基本周期）を採る。隣り合う縁だけ数えるので、2 個分の間隔は現れない（0.6〜0.95 で結果は変わらない）
RECT_PEAK_RATIO = 0.8
# L の候補: S のこの倍より長い間隔の極大のうち、（間隔 ^ RECT_LONG_GAP_POWER × 値）が最大のもの。間隔で重みづけるのは、同じ行の長さを占める割合に近づけるため（1〜3 で当たりが 0.55〜0.88 に動く。2 を選んだ）
RECT_LONG_MIN_RATIO = 1.4
RECT_LONG_GAP_POWER = 2.0

# ② L の補正: L の候補が 2 個分に見えるとき（端と端が接して並ぶ配置）だけ、敷き詰めの質で直す。候補の梯子は、60 から 1.2 倍刻みの 9 段（60 … 258 画素）のうち、
# L の候補の RECT_SCAN_LOW_RATIO 倍以上（かつ S の RECT_SCAN_MIN_ASPECT 倍以上）、RECT_SCAN_HIGH_RATIO 倍以下のものと、L の候補自身
RECT_LADDER_BASE = 60.0
RECT_LADDER_RATIO = 1.2
RECT_LADDER_STEPS = 9
RECT_SCAN_LOW_RATIO = 0.45
RECT_SCAN_MIN_ASPECT = 1.3
RECT_SCAN_HIGH_RATIO = 1.15
# 梯子の走査の敷き詰め: 画像を縮小する大きさ（画素）・向きの刻み（度）・候補の上限
RECT_SCAN_HALF = 96
RECT_SCAN_ANGLE_STEP = 20.0
RECT_SCAN_TOP = 500
# L の候補が敷き詰めの質で選んだ長さのこの倍以上なら、L の候補は 2 個分とみなして敷き詰めの長さを採る（1 個分と 2 個分の中間。1.3〜1.7 で結果はほぼ変わらない）
RECT_DOUBLING = 1.5

# ② 敷き詰め: 長方形 (L, S, 向き) を各位置に置いた点数（前景の割合 ^ RECT_COVERAGE_POWER × 4 辺の縁の強さ）の極大から、点数の高い順に、前景の割合が RECT_MIN_COVERAGE 以上で、
# すでに選んだ長方形との重なりが自分の面積の RECT_MAX_OVERLAP 以下のものを選ぶ（貪欲）。縁の強さは、画像の縁の強さの RECT_EDGE_SCALE_PERCENTILE 分位点で割って RECT_EDGE_CLIP で打ち切る
# 最終の敷き詰め: 画像を縮小する大きさ（画素）・向きの刻み（度）・候補の上限
RECT_TILE_HALF = 256
RECT_TILE_ANGLE_STEP = 15.0
RECT_TILE_TOP = 800
RECT_COVERAGE_POWER = 4
RECT_MIN_COVERAGE = 0.8
RECT_MAX_OVERLAP = 0.2
RECT_EDGE_SCALE_PERCENTILE = 90
RECT_EDGE_CLIP = 1.5
# 点数の極大を探す正方形の一辺（画素。縮小した画像の）
RECT_CANDIDATE_WINDOW = 5

# ③ デコード: 初期マスクは長方形の内側 +PRIOR_LOGIT_RANGE・外側 -PRIOR_LOGIT_RANGE（長方形は RECT_MASK_SHRINK 倍に縮める）。箱は縮めた長方形の外接矩形（軸に平行）、点は中心。
# 向きを戻さず再エンコードなしで 1 個ずつ切り出せる（正解の姿勢で recall 1.00。向き ±10°・大きさ 0.7〜1.2 倍まで許容）
RECT_MASK_SHRINK = 0.85
# デコーダーの 3 つの候補のうち、使う候補の番号（検証はすべてこの候補で行った）
RECT_MASK_CANDIDATE = 1
# デコードの窓（image_embed のトークン数。1 トークン = 16 画素）: 箱の長辺 / 16 + RECT_WINDOW_MARGIN_TOKENS を RECT_WINDOW_TOKEN_STEP の倍数に切り上げ、RECT_WINDOW_MIN_TOKENS 〜 RECT_WINDOW_MAX_TOKENS に収める。
# 窓の大きさごとに一括でデコードする
RECT_WINDOW_MARGIN_TOKENS = 3
RECT_WINDOW_TOKEN_STEP = 4
RECT_WINDOW_MIN_TOKENS = 8
RECT_WINDOW_MAX_TOKENS = 32
# 物体として採るマスクの下限: 面積（画素）と予測 IoU（0.5 は、物体と一致したとみなす IoU の基準）
RECT_MIN_AREA = 20
RECT_MIN_PREDICTED_IOU = 0.5
# 重複の除去: 予測 IoU の高い順に採り、採った結果と IoU が RECT_NMS_IOU 以上、または小さい方の面積の RECT_NMS_CONTAIN 以上が重なるものを捨てる
RECT_NMS_IOU = 0.3
RECT_NMS_CONTAIN = 0.5

# --- OpenVINO 実行環境（core/openvino/runtime_utils.py） ---
# pip 版 openvino との衝突を避けるため sys.path / PYTHONPATH から外すシステム版 OpenVINO の場所
SYSTEM_OPENVINO_DIR = "/opt/intel/openvino"
# NumPy の BLAS（OpenBLAS）のスレッド数。BLAS は既定で論理コア数（20）のスレッドを起こし、行列積（`@`）の後もしばらくスピン待ちするので、
# OpenVINO のスレッド（ピン止め）とコアを取り合い、直後の推論が 2〜4 倍遅くなる（行列積 2ms の後のデコードが 7ms → 29ms）。
# 本パイプラインの行列積は小さい（1024×256）ので、1 スレッドで 4ms 程度で済み、結果も同じ
BLAS_NUM_THREADS = 1

# --- 画像の読み込み ---
# データセット定義yamlのディレクトリを探索するときに画像とみなす拡張子
IMAGE_EXTENSIONS = {".png", ".jpg", ".bmp"}

# --- 重みの自動ダウンロード ---
# edgetam_model_path に重みが無いときの取得先（Hugging Face の公式配布）
EDGETAM_WEIGHTS_URL = "https://huggingface.co/facebook/EdgeTAM/resolve/main/edgetam.pt"

# =============================================================================
# 結果の保存・表示（visualization.py / gallery.py。セグメンテーションの結果には影響しない）
# =============================================================================

# セマンティック可視化の色 (RGB)。0番は背景で常にシアン。足りなければ 1 番以降を繰り返す
SEMANTIC_COLORS = (
    (0, 255, 255),
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (255, 128, 0), (128, 0, 255), (255, 255, 255),
    (128, 64, 0), (255, 128, 192), (0, 128, 0), (128, 128, 128),
)
# KMeans のクラスタ・粗いセマンティックラベルの着色に使う、PIL のパレット（R, G, B を 1 色ずつ並べたもの。41 色。0 番は黒）
CLUSTER_COLOR_PALETTE = (
      0,   0,   0,
    255,   0,   0,
      0, 255,   0,
      0,   0, 255,
    255, 255,   0,
      0, 255, 255,
    255,   0, 255,
    192, 192, 192,
    128,   0,   0,
    128, 128,   0,
      0, 128,   0,
    128,   0, 128,
      0,   0, 128,
    255, 165,   0,
    255, 105, 180,
    255,  69,   0,
    255,  20, 147,
     34, 139,  34,
      0, 255, 127,
    255, 182, 193,
    255, 215,   0,
    139,  69,  19,
    244, 164,  96,
     72,  61, 139,
      0, 191, 255,
     70, 130, 180,
    147, 112, 219,
    153,  50, 204,
    255, 228, 225,
    255,   0, 255,
    255, 228, 181,
    255,  99,  71,
      0, 250, 154,
    221, 160, 221,
    238, 130, 238,
    186,  85, 211,
    255, 240, 245,
    219, 112, 147,
    255,  69,   0,
    240, 128, 128,
    255, 105, 180,
)
# インスタンスマップ（密集物体の分離）の色: インスタンスごとに色相をこの割合ずつずらす（黄金比。隣り合う番号が似た色にならない）。彩度・明度は固定
INSTANCE_HUE_STEP = 0.618033988749895
INSTANCE_SATURATION = 0.75
INSTANCE_VALUE = 1.0
# インスタンス分離（instance_split）の可視化: 色相の刻み（HSV の H は 0〜179。黄金角に近い値で、隣り合う番号の色が離れる）、重畳の不透明度、番号の文字の大きさ
INSTANCE_SPLIT_HUE_STEP = 47
INSTANCE_SPLIT_OVERLAY_ALPHA = 0.5
INSTANCE_SPLIT_LABEL_SCALE = 1.0
# 前景オーバーレイ（粗い前景マスク・EdgeTAM最終マスク）の赤の不透明度
FOREGROUND_OVERLAY_ALPHA = 0.3
# 前景オーバーレイに描くプロンプト点の半径(px)と色(BGR)。正例=緑、負例=赤
POINT_RADIUS = 20
POINT_COLOR_POSITIVE = (0, 255, 0)
POINT_COLOR_NEGATIVE = (0, 0, 255)
# セマンティック結果（重畳 + 凡例）のmatplotlib図
RESULT_FIGSIZE = (10, 6)
RESULT_DPI = 150
LEGEND_FONTSIZE = 9
# 精緻化の中間画像を入れる、保存先の下のフォルダ名
DEBUG_DIR_NAME = "debug"

# 一覧HTML（gallery.py）に並べる画像（列見出し, ファイル名）。元画像は保存先ではなく入力画像のパスを参照する
GALLERY_IMAGES = (
    ("foreground_map", "foreground_map.png"),
    ("coarse_semantic_segmentation", "coarse_semantic_segmentation.png"),
    ("foreground_overlay", "foreground_overlay.png"),
    ("semantic_result", "semantic_result.png"),
)
GALLERY_FILE_NAME = "index.html"
# 元画像を保存先へコピーするときのファイル名（拡張子は元のまま）。output_dir ごと移動しても参照が切れないようにする
ORIGINAL_STEM = "original"
# 一覧HTMLの列の最小幅(px)。列は画面幅に合わせて等分され、画像は列幅に合わせて拡縮する（足りなければ横スクロール）
GALLERY_MIN_COLUMN_WIDTH = 280
# 一覧HTMLの1ページあたりの表示枚数の選択肢（先頭が既定）
GALLERY_PAGE_SIZES = (50, 100, 200)
