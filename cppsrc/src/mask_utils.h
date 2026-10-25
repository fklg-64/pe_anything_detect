// マスク・座標・点サンプリングの純粋なヘルパー関数（状態を持たない）（core/common/mask_utils.py）。
// 2 次元の配列は cv::Mat。2 値マスクは CV_8UC1 で 0 / 1（OpenCV の関数は 0 以外を真として扱う）。
// scipy.ndimage は OpenCV の関数で置き換えた。OpenCV に無い「最近傍の特徴点の位置つきの距離変換」だけ自前（nearest_feature_index）。
#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace pes {

// マスクの連結成分1つ分
struct Component {
    cv::Point peak;   // 最内点 (x, y)。距離変換が最大の位置
    cv::Mat core;     // 成分を1パッチ内側へ縮めたマスク（縮めると消える細い成分では成分そのもの）
    cv::Mat region;   // 成分のマスク
};

// --- 点のサンプリング ---
// 最遠点サンプリングで k 点を選ぶ。最初は重心に最も近い点で、以降は選択済みの点から最も遠い点を足していく。0 <= k <= points.size()
std::vector<cv::Point> farthest_point_sampling(const std::vector<cv::Point>& points, int k);
// 真の画素から最遠点サンプリングで最大 k 点を抽出する（行優先の順に並べた座標 (x, y) から）
std::vector<cv::Point> sample_points(const cv::Mat& candidate, int k);
// 正例点を返す。各連結成分の最内点(peaks)を必ず含め、残りは core 上の最遠点で k 点まで補う
std::vector<cv::Point> prompt_points(const cv::Mat& core, const std::vector<cv::Point>& peaks, int k);
// 画像端につながる背景のうち、前景から十分離れた位置から負例点を count 点抽出する。foreground_map（CV_32F、空でもよい）を渡すと、
// 前景確信度が最も低いパッチから選ぶ
std::vector<cv::Point> select_background_points(const cv::Mat& foreground_binary, int count, const cv::Mat& foreground_map);

// --- マスクの解析・整形 ---
// マスクの連結成分（8 近傍）ごとに (最内点, core, 成分マスク) を返す
std::vector<Component> connected_components(const cv::Mat& mask);
// マスクの外接矩形（上下左右に margin 画素の余白つき、画像内に収める）。空なら false
bool bounding_box(const cv::Mat& mask, int margin, cv::Rect* box);
// 小さな連結成分（面積 min_area 未満）を除去し、輪郭をメディアンフィルタで平滑化する。0/1 のマスクを返す。
// 処理は前景の外接矩形（余白 1 画素）の中だけで行う。box（NULL 可）には処理した矩形を返す（空なら Rect()）
cv::Mat clean_mask(const cv::Mat& mask, int min_area, cv::Rect* box = nullptr);
// セマンティックマップ（CV_8UC1）上の小領域ノイズを除去する（in-place）
void remove_small_regions(cv::Mat& semantic_map, int min_area);
// 粗い前景の内部にある、幅が 1 パッチ未満の背景の隙間を前景にする（semantic_map を in-place で更新）。foreground: パッチ格子の粗い前景マスク
void fill_narrow_gaps(cv::Mat& semantic_map, const cv::Mat& foreground);

// 各画素から最も近い「特徴画素（is_feature が 0 でない画素）」の位置を、行優先の通し番号 (y * cols + x) で返す（CV_32S）。
// 特徴画素が 1 つも無ければ全部 -1。scipy.ndimage.distance_transform_edt(return_indices=True) 相当（距離が同じ候補の選び方だけ異なりうる）
cv::Mat nearest_feature_index(const cv::Mat& is_feature);

// --- パッチ格子 <-> 作業解像度の変換 ---
// パッチ近傍(PATCH_NEIGHBORHOOD 四方)の平均。境界は scipy.ndimage の既定(reflect)と同じ
cv::Mat patch_mean(const cv::Mat& patch_map);
// ガイデッドフィルタ（He ら 2010）の輝度（1チャンネル）ガイド版
cv::Mat guided_filter(const cv::Mat& guide, const cv::Mat& src, int radius, float eps);
// パッチ格子のマップを size=(幅, 高さ) へ拡大する（float32 で返す）
cv::Mat resize_map(const cv::Mat& patch_map, cv::Size size, int interpolation);

// bool 風のマスク（0 以外が真）を 0/1 の CV_8UC1 にする
cv::Mat to_binary01(const cv::Mat& mask);
// 論理演算（0/1 マスク）
cv::Mat mask_not(const cv::Mat& a);

// --- kNN グラフ（EdgeTAM の特徴での前景確信度の伝播） ---
// 行を正規化した疎行列（CSR）。対称で、重みは 1 / 次数
struct KnnGraph {
    int n = 0;
    std::vector<int> indptr;        // n + 1
    std::vector<int> indices;       // 隣接する番号（行ごとに昇順）
    std::vector<float> weights;     // 各要素の重み（float32。Python 版と同じ）
};
// 特徴 (n × dim)（L2 正規化済み）のコサイン類似度が上位 k 個の近傍で、対称の kNN グラフを作る
KnnGraph knn_graph(const float* features, int n, int dim, int k);
// values をグラフ上で value ← (1-α)·values0 + α·(近傍の平均) と iterations 回伝播する
std::vector<float> propagate(const std::vector<float>& values, const KnnGraph& graph, double alpha, int iterations);

}  // namespace pes
