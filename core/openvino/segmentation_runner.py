import re
import time
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import faiss
import numpy as np
from PIL import Image

from core.utils import Timer, load_image, preprocess_image
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.edgetam_runner import OVEdgeTamRunner

from scipy.ndimage import binary_dilation, binary_erosion, convolve, distance_transform_edt
from scipy.ndimage import label as ndi_label

import numpy as np

# =============================================================================
# 固定ハイパーパラメータ
# yamlでは調整せずコードに固定している値を、すべてここにまとめる。
# 値を変える場合はこのブロックだけを編集すること（他の箇所に数値を直書きしない）。
# =============================================================================

# --- CLIP（粗い前景・セマンティックの判定） ---
# テキストとの類似度(cos)に掛ける係数（softmaxの温度の逆数）
CLIP_LOGIT_SCALE = 100.0

# --- KMeans ---
KMEANS_SEED = 42
FAISS_NUM_THREADS = 2

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
# EdgeTAMのロジットがこの値を超えた画素を前景とする
MASK_LOGIT_THRESHOLD = 0.0
# EdgeTAMのデコーダ出力マスクを縮小する倍率（分母。高さ・幅を 1/N にする）
EDGETAM_MASK_DOWNSCALE = 8

# --- 重なり解決 ---
# スコア = EdgeTAMの確信度 × ((1 - w) + w × 粗い前景らしさ) の w。0.5で両者を等しく扱う
SCORE_PRIOR_WEIGHT = 0.5

# --- ノイズ除去 ---
# マスクの輪郭を平滑化するメディアンフィルタのカーネルサイズ
MEDIAN_BLUR_KSIZE = 3

# --- 可視化（結果の見た目のみ。セグメンテーション結果には影響しない） ---
# セマンティック可視化の色 (RGB)。0番は背景で常にシアン
SEMANTIC_COLORS = np.array([
    [0, 255, 255],
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [255, 128, 0], [128, 0, 255], [255, 255, 255],
    [128, 64, 0], [255, 128, 192], [0, 128, 0], [128, 128, 128],
], dtype=np.uint8)
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

# 選択済みのポイントから最遠点を繰り返し抽出
def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """
    points: (M, 2) の座標配列。各行は (x, y)
    k:      抽出する点の数

    return: (k, 2) の座標配列
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M, 2)")

    m = len(points)

    if not 0 <= k <= m:
        raise ValueError("k must be between 0 and len(points)")

    if k == 0:
        return points[:0].copy()

    if k == m:
        return points.copy()

    # 距離計算用
    pts = points.astype(np.float64, copy=False)

    # 最初は重心に最も近い点を選ぶ
    center = pts.mean(axis=0)
    first = np.argmin(np.sum((pts - center) ** 2, axis=1))

    selected = np.empty(k, dtype=np.intp)
    selected[0] = first

    # 各点から、選択済みの点までの最小二乗距離
    min_dist2 = np.sum((pts - pts[first]) ** 2, axis=1)
    min_dist2[first] = -1

    for i in range(1, k):
        # 選択済みの点から最も遠い候補を選ぶ
        idx = np.argmax(min_dist2)
        selected[i] = idx

        # 新しく選んだ点との距離で最小距離を更新
        dist2 = np.sum((pts - pts[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

        # 重複選択を防ぐ
        min_dist2[selected[:i + 1]] = -1

    return points[selected]

# マスク画像からノイズを除去する関数
def clean_mask(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """
    mask: 2次元の二値マスク（0/1 または 0/255）
    min_area: 除去する連結成分の最小面積（ピクセル数）

    return: 0/1 の二値マスク
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)

    # 小さな独立成分を除去
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # 面積が足りる連結成分だけを残す（背景ラベル0は常に除外）
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    cleaned = keep[labels].astype(np.uint8)

    # 輪郭の細かなノイズを平滑化
    cleaned = cv2.medianBlur(cleaned, MEDIAN_BLUR_KSIZE)

    return cleaned

class SegmentationRunner:
    """抽出された特徴量・クラスタリングを用いた推論・マスク生成の実行クラス"""

    def __init__(
        self, 
        extractor: OVCLIPFeatureExtractor, 
        edgetam: OVEdgeTamRunner, 
        config: yaml
    ):
        # モデル
        self.extractor = extractor
        self.edgetam = edgetam

        # 設定値の初期化
        self.init_value(config)
        
        self.register_fgbg_text_features()

    def init_value(self, config: yaml):
        # config関係
        self.prompts_dir = config["prompts_dir"]
        self.k_clusters = config["k_clusters"]
        self.niter = config["n_iter"]
        self.foreground_threshold = config["foreground_threshold"]
        self.output_dir = config["output_dir"]
        self.color_palette = config["color_palette"]
        self.et_fds = config["edgetam_high_res_feature_downscale_factor"]
        self.fg_pt_per_cls = config["edgetam_foreground_points_per_class"]
        self.bg_pt_cnt = config["edgetam_background_point_count"]
        self.refine_iterations = config["edgetam_refine_iterations"]
        self.semantic_max_side = config["semantic_max_side"]
        self.overlay_alpha = config["overlay_alpha"]
        self.debug = config["debug"]
        self.show_result = config["show_result"]

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size
        # edgetamの高解像度特徴量サイズ
        self.et_high_res_feature_size: int = self.edgetam.target_size // self.et_fds

        # プロンプトとプロンプトのテキスト特徴量
        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: Optional[np.ndarray] = None

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    # =========================================================================
    # 1. テキストプロンプト & 特徴量準備
    # =========================================================================

    def load_prompts(self, path: str, mode: int=0) -> List[str]:
        if mode == 0:
            with open(path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data["base"] + data["custom"]
        elif mode == 1:
            prompts = []
            with open(path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for color in data["color"]:
                for shape in data["shape"]:
                    article = "an" if color[0].lower() in "aeiou" else "a"
                    prompts += [f"a photo of {article} {color} {shape} object."]
            return prompts + data["custom"]
        elif mode == 2:
            prompts = []
            with open(path, mode="r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for cond in data["condition"]:
                for mat in data["material"]:
                    for ctx in data["context"]:
                        article = "an" if cond[0].lower() in "aeiou" else "a"
                        prompt = f"a photo of {article} {cond} {mat} {ctx}."
                        prompts.append(prompt)
            return prompts + data["custom"]
    
    def common_process(self, prompts: List[str]) -> np.ndarray:
        text_features = self.extractor.extract_text_features(prompts)
        return text_features
    
    def register_fgbg_text_features(self) -> None:
        foreground_prompts_path = Path(self.prompts_dir) / "foreground.yaml"
        background_prompts_path = Path(self.prompts_dir) / "background.yaml"

        self.foreground_prompts = self.load_prompts(foreground_prompts_path, mode=1)
        self.background_prompts = self.load_prompts(background_prompts_path, mode=0)

        fg_feats = self.common_process(self.foreground_prompts)
        bg_feats = self.common_process(self.background_prompts)

        self.fgbg_text_features = np.vstack([fg_feats, bg_feats])

    # =========================================================================
    # 2. クラスタリング & 可視化ユーティリティ
    # =========================================================================

    def fit_predict_kmeans(
        self, 
        features: np.ndarray, 
        k: int = 12, 
        niter: int = 10
    ) -> np.ndarray:
        N, D = features.shape
        features_contig = (
            np.ascontiguousarray(features, dtype=np.float32)
            if not features.flags["C_CONTIGUOUS"]
            else features
        )
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)
        kmeans = faiss.Kmeans(
            d=D, k=k, niter=niter, spherical=False, verbose=False, seed=KMEANS_SEED
        )
        kmeans.train(features_contig)
        _, cluster_labels = kmeans.index.search(features_contig, 1)
        return cluster_labels.reshape(-1)

    def colorize_clusters_PIL(self, cluster_labels: np.ndarray) -> Image.Image:
        color_map = Image.fromarray(cluster_labels.astype(np.uint8))
        color_map.putpalette(self.color_palette)
        return color_map

    # =========================================================================
    # 3. セマンティックセグメンテーションの精緻化 & 座標変換
    # =========================================================================

    @staticmethod
    def _sample_points(candidate: np.ndarray, k: int) -> np.ndarray:
        """True の画素から最遠点サンプリングで最大 k 点を抽出する。座標は (x, y)。"""
        points = np.argwhere(candidate)[:, ::-1]
        return farthest_point_sampling(points, min(k, len(points)))

    @staticmethod
    def _prompt_points(core: np.ndarray, peaks: List[Tuple[int, int]], k: int) -> np.ndarray:
        """正例点を返す。各連結成分の最内点(peaks)を必ず含め、残りは core 上の最遠点で k 点まで補う。

        最内点を必ず含めることで、細い物体や小さい物体にも漏れなく点が置かれる。
        """
        selected = [np.asarray(p, dtype=np.float64) for p in peaks]
        points = np.argwhere(core)[:, ::-1].astype(np.float64)
        if len(points) and len(selected) < k:
            min_d2 = np.min(((points[:, None, :] - np.asarray(selected)[None]) ** 2).sum(-1), axis=1)
            while len(selected) < k and min_d2.max() > 0:
                idx = int(np.argmax(min_d2))
                selected.append(points[idx])
                min_d2 = np.minimum(min_d2, ((points - points[idx]) ** 2).sum(-1))
        return np.asarray(selected).reshape(-1, 2).astype(int)

    def _components(self, mask: np.ndarray) -> List[Tuple[Tuple[int, int], np.ndarray, np.ndarray]]:
        """マスクの連結成分ごとに (最内点(x, y), core, 成分マスク) を返す。

        最内点は距離変換の最大点。core は成分を1パッチ内側へ縮めたもので、
        縮めると消える細い成分では成分そのものを core とする。
        """
        n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        result = []
        for i in range(1, n):
            region = comp == i
            edt = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
            y, x = np.unravel_index(np.argmax(edt), edt.shape)
            core = binary_erosion(region, structure=PATCH_STRUCTURE, border_value=1)
            result.append(((int(x), int(y)), core if core.any() else region, region))
        return result

    def _save_debug_image(self, name: str, data: np.ndarray, width: int, height: int) -> None:
        """デバッグ用の中間画像を save_dir/debug に元画像サイズで保存する。"""
        debug_dir = self.save_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        image = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(debug_dir / f"{name}.png"), (image * 255).astype(np.uint8))

    def _remove_small_regions(self, semantic_map: np.ndarray, min_area: int) -> None:
        """セマンティックマップ上の小領域ノイズを除去する（in-place）。

        1. クラスごとに小さな連結成分を背景へ戻す。
        2. 背景に残った小さな穴を、周囲で最も多いクラスで埋める。
        """
        height, width = semantic_map.shape
        for class_id in np.unique(semantic_map):
            if class_id == 0:
                continue
            region = semantic_map == class_id
            keep = clean_mask(region, min_area).astype(bool)
            semantic_map[region & ~keep] = 0

        n, comp, stats, _ = cv2.connectedComponentsWithStats(
            (semantic_map == 0).astype(np.uint8), connectivity=8
        )
        ring_kernel = np.ones((3, 3), dtype=np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                continue
            x, y, w, h = stats[i, :4]
            x0, y0 = max(x - 1, 0), max(y - 1, 0)
            x1, y1 = min(x + w + 1, width), min(y + h + 1, height)
            hole = comp[y0:y1, x0:x1] == i
            ring = cv2.dilate(hole.astype(np.uint8), ring_kernel).astype(bool) & ~hole
            neighbors = semantic_map[y0:y1, x0:x1][ring]
            neighbors = neighbors[neighbors > 0]
            if neighbors.size:
                semantic_map[y0:y1, x0:x1][hole] = np.bincount(neighbors).argmax()

    def refine_coarse_segmentation(
        self, semantic_labels: np.ndarray, foreground_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """パッチ単位の粗い結果を整える。返り値は (前景マスク, セマンティックラベル)。

        前景に囲まれた背景（反射・透明部などの穴）は前景とし、最寄りの前景ラベルを与える。
        最終的な形はEdgeTAMが決めるため、穴埋めは粗いマスクを物体単位でつなぐ目的に限る。

        続いて、縁・影由来の細いクラス領域を取り除く。

        物体の縁や影のパッチは背景と物体のトークンが混ざり、別クラスタ（別プロンプト）になりやすい。
        幅が2パッチ以下（3x3で縮めると消える）のクラス領域が他クラスの前景に接していれば、
        最も長く接するクラスへ統合する。小さい領域から順に処理する。
        """
        original_fg = foreground_mask != 0
        fg = original_fg.copy()
        labels = semantic_labels.copy()

        # 画像端に接しない背景の連結成分 = 穴
        n, comp = cv2.connectedComponents((~fg).astype(np.uint8), connectivity=4)
        border_ids = np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))
        holes = (comp > 0) & ~np.isin(comp, border_ids)
        if holes.any() and original_fg.any():
            _, (iy, ix) = distance_transform_edt(~original_fg, return_indices=True)
            labels[holes] = labels[iy, ix][holes]
            fg |= holes

        regions = []
        for class_id in np.unique(labels[fg]):
            n, comp, stats, _ = cv2.connectedComponentsWithStats(
                (fg & (labels == class_id)).astype(np.uint8), connectivity=4
            )
            regions += [(stats[i, cv2.CC_STAT_AREA], class_id, comp == i) for i in range(1, n)]

        for _, class_id, region in sorted(regions, key=lambda r: r[0]):
            region &= labels == class_id
            if not region.any() or binary_erosion(region, structure=PATCH_STRUCTURE, border_value=1).any():
                continue
            ring = binary_dilation(region, structure=PATCH_STRUCTURE) & ~region
            neighbors = labels[ring & fg]
            neighbors = neighbors[neighbors != class_id]
            if neighbors.size:
                labels[region] = np.bincount(neighbors).argmax()

        return fg.astype(foreground_mask.dtype), labels

    def _select_background_points(self, foreground_binary: np.ndarray) -> np.ndarray:
        """画像端につながる背景のうち、前景から十分離れた位置から負例点を抽出する。

        物体内部の穴や影・縁の近くに負例を置くと、物体が欠けるため避ける。
        """
        background = ~foreground_binary
        labeled, _ = ndi_label(background)
        border_ids = np.unique(np.concatenate([labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]]))
        outer = np.isin(labeled, border_ids[border_ids > 0])
        if not outer.any():
            outer = background
        dist = distance_transform_edt(outer)
        for margin in (BACKGROUND_POINT_MARGIN, 1):
            candidate = outer & (dist >= margin)
            if candidate.any():
                break
        return self._sample_points(candidate, self.bg_pt_cnt)

    def _estimate_class_logit(
        self,
        feats: Tuple,
        prior_map: np.ndarray,
        foreground_points: np.ndarray,
        background_coords: List[List[float]],
        width: int,
        height: int,
        tag: str,
    ) -> Tuple[Optional[np.ndarray], List[List[float]], List[int]]:
        """正例点と共通の負例点でEdgeTAMを反復実行し、低解像度ロジット(128x128)を返す。

        prior_map: 初期マスクの前景確率。ロジットは (2p-1)*PRIOR_LOGIT_RANGE（0.5=不明）。
        """
        coords = self.transform_coords(foreground_points.tolist(), width, height, self.feature_size)
        labels = [1] * len(coords)
        coords = coords + background_coords
        labels = labels + [0] * len(background_coords)

        mask_input = cv2.resize(
            (2.0 * prior_map - 1.0) * PRIOR_LOGIT_RANGE,
            (self.et_high_res_feature_size, self.et_high_res_feature_size),
            interpolation=cv2.INTER_LINEAR,
        )[None, ...]

        logit: Optional[np.ndarray] = None
        for it in range(self.refine_iterations):
            with Timer(f"    ┌ edgetam {tag} {it + 1}回目マスク推定"):
                _, scores, logits = self._predict_edgetam_decoder(
                    point_coords=coords,
                    point_labels=labels,
                    feats=feats,
                    height=height,
                    width=width,
                    mask_input=mask_input,
                )
            if len(scores) == 0:
                break
            best_idx = int(np.argmax(scores))
            mask_input = logits[best_idx:best_idx + 1]
            logit = logits[best_idx]
        return logit, coords, labels

    def refine_semantic_segmentation(
        self,
        image_raw: np.ndarray,
        foreground_map: np.ndarray,
        foreground_mask: np.ndarray,
        semantic_labels: np.ndarray,
        labels_list: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, List[int], List[List[float]], List[int]]:
        """粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合する。

        クラスごとにEdgeTAMで前景マスクを推定し、重なり部分は
        「EdgeTAMの確信度 × 粗い前景らしさ(foreground_map)」が高いクラスを優先する。
        背景は全クラス共通でクラス0とし、前景は1から振る。
        どのクラスのマスクにも入らなかった粗い連結成分は、その成分だけで再推定する。

        Returns:
            semantic_map: (height, width) uint8。0=背景、1~=前景クラス。
            prompt_indices: クラス1~に対応する、前景プロンプトのインデックス。
            point_coords / point_labels: EdgeTAMに与えたプロンプト点（画像座標、全クラス分）。
        """
        foreground_binary = foreground_mask != 0
        semantic_ids: np.ndarray = np.unique(semantic_labels[foreground_binary])
        prompt_indices: List[int] = [int(labels_list[i]) for i in semantic_ids]

        # マスク演算は長辺が semantic_max_side 以下の作業解像度で行い、最後に元画像サイズへ戻す
        scale = min(1.0, self.semantic_max_side / max(width, height))
        work_w, work_h = max(1, round(width * scale)), max(1, round(height * scale))

        semantic_map = np.zeros((work_h, work_w), dtype=np.uint8)
        point_coords: List[List[float]] = []
        point_labels: List[int] = []
        if len(semantic_ids) == 0:
            return np.zeros((height, width), dtype=np.uint8), prompt_indices, point_coords, point_labels

        debug = getattr(self, "debug", False)
        kernel = PATCH_STRUCTURE.astype(np.float32)
        size = (work_w, work_h)

        # EdgeTAMの特徴量抽出（全クラス共通）
        with Timer("    ┌ edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

        # 背景の負例点は全クラス共通
        background_coords = self.transform_coords(
            self._select_background_points(foreground_binary).tolist(), width, height, self.feature_size
        )

        # 粗いマスクの信頼限界は1パッチ。1パッチ未満の領域はノイズとして扱う
        min_area = max(1, int(MIN_REGION_PATCH_AREA * work_w * work_h / self.feature_size ** 2))
        best_score = np.full((work_h, work_w), -1.0, dtype=np.float32)

        def to_full(patch_map: np.ndarray, interpolation: int) -> np.ndarray:
            return cv2.resize(patch_map.astype(np.float32), size, interpolation=interpolation)

        def refine_region(coarse_mask, comps, tag):
            """粗いマスクに対するEdgeTAMの2値マスク(作業解像度)と、重なり解決用スコアを返す。"""
            core = np.zeros_like(coarse_mask, dtype=bool)
            for _, core_c, _ in comps:
                core |= core_c
            coarse_map = convolve(coarse_mask.astype(np.uint8), kernel) / kernel.sum()
            core_map = convolve(core.astype(np.uint8), kernel) / kernel.sum()
            # core内は前景、core外の粗いマスクは不明(PRIOR_NON_CORE_PROBABILITY)、粗いマスク外は背景
            prior_map = np.maximum(core_map, PRIOR_NON_CORE_PROBABILITY * coarse_map)

            fg_points = self._prompt_points(core, [c[0] for c in comps], self.fg_pt_per_cls)
            logit, coords, labels = self._estimate_class_logit(
                feats, prior_map, fg_points, background_coords, width, height, tag
            )
            point_coords.extend(coords)
            point_labels.extend(labels)

            coarse_full = to_full(coarse_mask, cv2.INTER_NEAREST) > 0
            if logit is None:
                return coarse_full, to_full(coarse_map * foreground_map, cv2.INTER_LINEAR), core
            logit_full = cv2.resize(logit.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
            binary = clean_mask(logit_full > MASK_LOGIT_THRESHOLD, min_area).astype(bool)
            # 重なり解決用スコア = EdgeTAMの確信度 × ((1-w) + w × 粗い前景らしさ)
            confidence = 1.0 / (1.0 + np.exp(-logit.astype(np.float32)))
            prior_low = cv2.resize(
                (coarse_map * foreground_map).astype(np.float32), confidence.shape[::-1], interpolation=cv2.INTER_LINEAR
            )
            return binary, to_full(confidence * ((1.0 - SCORE_PRIOR_WEIGHT) + SCORE_PRIOR_WEIGHT * prior_low), cv2.INTER_LINEAR), core

        class_components = {}
        for class_id, semantic_id in enumerate(semantic_ids, start=1):
            coarse_mask = (semantic_labels == semantic_id) & foreground_binary
            comps = self._components(coarse_mask)
            class_components[class_id] = comps
            # 孤立した極小成分（背景クラスタの斑点）はクラスの推定に混ぜない。混ぜると、斑点に置いた点を
            # 経由して物体のマスクが周囲の背景へ広がる。斑点は後段で個別に（範囲を制限して）扱う
            main = [c for c in comps if c[2].sum() >= MIN_ISOLATED_PATCHES] or comps
            coarse_main = np.any([c[2] for c in main], axis=0)
            binary, score, core = refine_region(coarse_main, main, f"クラス{class_id}")

            # 重なり部分はスコアの高いクラスを優先
            win = binary & (score > best_score)
            semantic_map[win] = class_id
            best_score[win] = score[win]

            if debug:
                self._save_debug_image(f"c{class_id}_coarse_mask", coarse_mask, *size)
                self._save_debug_image(f"c{class_id}_core", core, *size)
                self._save_debug_image(f"c{class_id}_edgetam_mask", binary, *size)
                self._save_debug_image(f"c{class_id}_score", score, *size)

        # 取りこぼし防止: 最内点がどのクラスにも覆われていない粗い連結成分は、単独で再推定する
        for class_id, comps in class_components.items():
            for peak, core_c, region in comps:
                px = min(int((peak[0] + 0.5) * work_w / self.feature_size), work_w - 1)
                py = min(int((peak[1] + 0.5) * work_h / self.feature_size), work_h - 1)
                if semantic_map[py, px] != 0:
                    continue
                binary, _, _ = refine_region(region, [(peak, core_c, region)], f"クラス{class_id}(再推定)")
                if region.sum() < MIN_ISOLATED_PATCHES:
                    # 極小成分は、成分を1パッチ広げた範囲までに制限する
                    allowed = binary_dilation(region, structure=PATCH_STRUCTURE)
                    binary &= to_full(allowed, cv2.INTER_NEAREST) > 0
                semantic_map[binary & (semantic_map == 0)] = class_id
                if debug:
                    self._save_debug_image(f"c{class_id}_rescue_{peak[0]}_{peak[1]}", binary, *size)

        if debug:
            self._save_debug_image(
                "semantic_map_before_cleanup", semantic_map / max(len(semantic_ids), 1), *size
            )

        # 重ね合わせ後のノイズ除去
        self._remove_small_regions(semantic_map, min_area)

        if (work_w, work_h) != (width, height):
            semantic_map = cv2.resize(semantic_map, (width, height), interpolation=cv2.INTER_NEAREST)

        return semantic_map, prompt_indices, point_coords, point_labels

    @staticmethod
    def _clean_prompt(prompt: str) -> str:
        """"a photo of a " などの固有表現と末尾のピリオドを除いたクラス名にする。"""
        return re.sub(r"^\s*a photo of (?:an?\s+)?", "", prompt, flags=re.IGNORECASE).rstrip(" .")

    def semantic_class_names(self, prompt_indices: List[int]) -> List[str]:
        """クラス0=background、クラス1~=前景プロンプト名のリストを返す。"""
        all_prompts = self.foreground_prompts + self.background_prompts
        return ["background"] + [self._clean_prompt(all_prompts[i]) for i in prompt_indices]

    @staticmethod
    def semantic_colors(num_classes: int) -> np.ndarray:
        """クラス数分の色 (RGB, uint8)。足りなければ繰り返し、0番は常にシアン。"""
        fg = SEMANTIC_COLORS[1:]
        return np.vstack([SEMANTIC_COLORS[:1], fg[np.arange(num_classes - 1) % len(fg)]])[:num_classes]

    def _save_semantic_result(
        self,
        save_dir: Path,
        image_raw: np.ndarray,
        semantic_map: np.ndarray,
        class_names: List[str],
    ) -> None:
        """セマンティックマップの保存と、元画像への重畳＋凡例の表示。"""
        colors = self.semantic_colors(len(class_names))
        color_image = colors[semantic_map]

        # クラスIDそのものを保存（パレット付きPNG）
        index_image = Image.fromarray(semantic_map, mode="P")
        index_image.putpalette(colors.flatten().tolist())
        index_image.save(save_dir / "semantic_map.png")

        # 元画像に色を重畳
        overlay = cv2.addWeighted(
            image_raw, 1.0 - self.overlay_alpha, color_image, self.overlay_alpha, 0
        )
        cv2.imwrite(str(save_dir / "semantic_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 重畳画像 + 色とプロンプトの凡例
        import matplotlib
        if not self.show_result:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        ratios = np.bincount(semantic_map.ravel(), minlength=len(class_names)) / semantic_map.size
        handles = [
            Patch(facecolor=colors[i] / 255.0, edgecolor="black",
                  label=f"{i}: {name} ({ratios[i] * 100:.1f}%)")
            for i, name in enumerate(class_names)
        ]
        fig, ax = plt.subplots(figsize=RESULT_FIGSIZE)
        ax.imshow(overlay)
        ax.axis("off")
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=LEGEND_FONTSIZE)
        fig.savefig(save_dir / "semantic_result.png", bbox_inches="tight", dpi=RESULT_DPI)
        if self.show_result:
            plt.show()
        plt.close(fig)

    def convert_coordinate(self, point: int, img_size: int, feat_size: int) -> float:
        return point * img_size / feat_size

    def transform_coords(self, pts: np.ndarray, width: int, height: int, feature_size: int) -> List[List[float]]:
        return [
            [
                self.convert_coordinate(x + 0.5, width, feature_size),
                self.convert_coordinate(y + 0.5, height, feature_size)
            ]
            for x, y in pts
        ]

    # =========================================================================
    # 4. EdgeTAM 推論バックエンド共通インターフェース (OpenVINO対応化用)
    # =========================================================================

    def _predict_edgetam_encoder(self, image_raw: Image.Image) -> Optional[Tuple]:
        """EdgeTAM Encoder の実行（PyTorch / Custom / 将来のOpenVINOの違いをカプセル化）"""
        return self.edgetam.encode(
            preprocess_image(
                image_raw, 
                self.edgetam.target_size, 
                self.edgetam.mean, self.edgetam.std
            )
        )

    def _predict_edgetam_decoder(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Optional[Tuple],
        height: int,
        width: int,
        mask_input: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """EdgeTAM Decoder の実行（PyTorch / Custom / 将来のOpenVINOの違いをカプセル化）"""
        if mask_input is None:
            mask_logits_tensor = np.array([0], dtype=np.float32)
        else:
            mask_logits_tensor = mask_input

        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self.edgetam.preprocess(
            (height, width),
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
            mask_logits=mask_logits_tensor,
        )

        with Timer("    ┌ デコード"):
            high_res_feats0, high_res_feats1, image_embed = feats
            low_res_masks, iou_predictions = self.edgetam.decode(
                high_res_feats0, high_res_feats1, image_embed,
                point_coords=point_coords_norm,
                point_labels=point_labels_norm,
                mask_input=mask_input_tensor,
                has_mask=has_mask,
            )

        with Timer("    ┌ リサイズ"):
            masks, scores, logits = self.edgetam.postprocess(
                low_res_masks, iou_predictions, (height // EDGETAM_MASK_DOWNSCALE, width // EDGETAM_MASK_DOWNSCALE)
            )

        return masks, scores, logits

    # =========================================================================
    # 5. 各パイプライン・サブタスク処理
    # =========================================================================

    def _feature_extraction(
        self, 
        image_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_features, patch_features, cls2patch_attn, fused_patch_features = (
            self.extractor.extract_image_features(image_tensor)
        )
        return image_features, patch_features, fused_patch_features

    def _image_classification(
        self, 
        image_features: np.ndarray
    ):
        # クラス分類
        text_probs = CLIP_LOGIT_SCALE * image_features @ self.fgbg_text_features.T
        text_probs = self._softmax(text_probs, axis=-1)

    def _generate_foreground_map(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """CLIP特徴量とKMeansによるベース前景マップの抽出"""

        # パッチレベルの特徴量を取得
        patch_tokens = fused_patch_features[0, 1:, :]

        cluster_index_list: List[int] = []
        fg_prompts_num: int = len(self.foreground_prompts)
        bg_prompts_num: int = len(self.background_prompts)
        foreground_map: np.ndarray = np.zeros((self.feature_size * self.feature_size,), dtype=np.float32)
        with Timer("    ┌ Kmeansクラスタ処理時間"):
            cluster_labels = self.fit_predict_kmeans(patch_tokens, k=self.k_clusters, niter=self.niter)

        with Timer("    ┌ クラスタごとのCLSトークン抽出処理時間"):
            label_token_list: List[np.ndarray] = self.extractor.extract_cluster_tokens(patch_features, cluster_labels)

        for label_idx, label_features in enumerate(label_token_list):
            # feature normalization
            label_features: np.ndarray = label_features.reshape(1, -1)
            label_norm: np.ndarray = np.linalg.norm(label_features, axis=-1, keepdims=True) + 1e-8
            label_features: np.ndarray = label_features / label_norm

            # similarity / logits
            logits = (CLIP_LOGIT_SCALE * label_features @ self.fgbg_text_features.T)

            # foreground / background を分離
            fg_logits = logits[:, :fg_prompts_num]
            bg_logits = logits[:, fg_prompts_num:]

            # グループスコアは最も近いプロンプトの類似度。プロンプトは「言い換え」ではなく「選択肢」の
            # 列挙なので、平均ではなく最大を使う（プロンプト数を増減しても判定が偏らない）
            fg_score = np.max(fg_logits, axis=-1)
            bg_score = np.max(bg_logits, axis=-1)

            # FG vs BG の2クラスsoftmax
            group_logits = np.concatenate([fg_score[:, None], bg_score[:, None]], axis=-1)
            group_probs = self._softmax(group_logits, axis=-1)
            foreground_confidence = float(group_probs[0, 0])

            # 前景か背景か
            if foreground_confidence > self.foreground_threshold:
                foreground_index = np.argmax(fg_logits, axis=1)
                cluster_index_list += [foreground_index[0]]
            else:
                background_index = np.argmax(bg_logits, axis=1)
                cluster_index_list += [fg_prompts_num + background_index[0]]
                
            # クラスタ内の全patchにconfidenceを付与
            foreground_map[cluster_labels == label_idx] += foreground_confidence

        # KMeansの結果を平均
        foreground_map = foreground_map.reshape(self.feature_size, self.feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)

        return (
            foreground_map,
            foreground_mask,
            cluster_labels, 
            cluster_index_list
        )

    def _run_edgetam_refinement(
        self,
        foreground_points: np.ndarray,
        background_points: np.ndarray,
        image_raw: Image.Image,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[int]]:
        """マスク生成"""

        # edgeTAMの特徴量抽出
        with Timer("    ┌ edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

        # 前景が存在しない場合はマスクなしとして処理を終了
        current_mask_list: List[np.ndarray] = []
        point_coords: List[List[float]] = []
        point_labels: List[int] = []
        if len(foreground_points) == 0:
            resized_mask: np.ndarray = np.zeros((height, width), dtype=np.uint8)
            return resized_mask, current_mask_list, [], []

        # 前景のポイントのみでセグメンテーション
        point_coords += self.transform_coords(foreground_points.tolist(), width, height, self.feature_size)
        point_labels += [1] * len(foreground_points)

        with Timer("    ┌ edgetam1回目マスク推定"):
            masks, scores, logits = self._predict_edgetam_decoder(
                point_coords=point_coords,
                point_labels=point_labels,
                feats=feats,
                height=height,
                width=width
            )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]
        current_mask_list += [current_mask]

        # 背景のポイントも追加してセグメンテーション
        point_coords += self.transform_coords(background_points.tolist(), width, height, self.feature_size)
        point_labels += [0] * len(background_points)

        with Timer("    ┌ edgetam2回目マスク推定"):
            masks, scores, logits = self._predict_edgetam_decoder(
                point_coords=point_coords,
                point_labels=point_labels,
                feats=feats,
                height=height,
                width=width,
                mask_input=low_res_logits
            )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]
        current_mask_list += [current_mask]

        # マスク画像を元画像サイズにリサイズ
        resized_mask: np.ndarray = cv2.resize(
            current_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_mask, current_mask_list, point_coords, point_labels

    def semantic_segmentation(
        self,
        cluster_labels: np.ndarray,
        cluster_index_list: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        cluster_indices = np.asarray(cluster_index_list)
        semantic_indices = cluster_indices[cluster_labels]

        labels_list, semantic_labels = np.unique(
            semantic_indices,
            return_inverse=True,
        )

        return labels_list, semantic_labels.reshape(self.feature_size, self.feature_size)

    def _save_results(
        self,
        save_dir: Path, 
        image_raw: Image.Image,
        width: int,
        height: int,
        semantic_map: np.ndarray,
        class_names: List[str],
        foreground_map: np.ndarray,
        foreground_mask: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        cluster_labels: np.ndarray, 
        semantic_labels: np.ndarray, 
        raw_semantic_labels: np.ndarray,
    ) -> None:
        """処理結果の画像・マスク・オーバーレイの書き出し"""
        # 最終セマンティックセグメンテーション結果（重畳 + 凡例）
        self._save_semantic_result(save_dir, image_raw, semantic_map, class_names)

        # クラスごとの最終マスクを保存
        for class_id in range(1, len(class_names)):
            mask_img = ((semantic_map == class_id) * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{class_id}.png"), mask_img)

        # レイヤー結合した特徴マップでのクラスタリング結果
        fused_cluster_labels_2d = cluster_labels.reshape(self.feature_size, self.feature_size)
        fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
        fused_cluster_rgb = fused_cluster_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        fused_cluster_rgb.save(save_dir / f"layer_fused_k{self.k_clusters}.png")

        # セマンティックセグメンテーション処理
        semantic_labels_2d = semantic_labels.reshape(self.feature_size, self.feature_size)
        semantic_labels_rgb = self.colorize_clusters_PIL(semantic_labels_2d)
        semantic_labels_rgb = semantic_labels_rgb.resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        semantic_labels_rgb.save(save_dir / "coarse_semantic_segmentation.png")

        # 整形前の粗いセマンティックセグメンテーション
        raw_labels_rgb = self.colorize_clusters_PIL(raw_semantic_labels.reshape(self.feature_size, self.feature_size))
        raw_labels_rgb.resize((width, height), resample=Image.Resampling.NEAREST).save(
            save_dir / "coarse_semantic_segmentation_raw.png"
        )

        # 前景度合いマップを保存
        foreground_map_2d = (
            (foreground_map * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        map_image = Image.fromarray(foreground_map_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        map_image.save(save_dir / "foreground_map.png")

        # 初期前景マスク画像を保存
        foreground_mask_2d = (
            (foreground_mask * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        mask_image = Image.fromarray(foreground_mask_2d).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask_image.save(save_dir / "foreground_mask.png")

        # オーバーレイ画像保存 (CLIP前景プロンプト点付き)
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_image)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = mask_cv
        overlay = cv2.addWeighted(image_cv, 1.0 - FOREGROUND_OVERLAY_ALPHA, colored_mask, FOREGROUND_OVERLAY_ALPHA, 0)
        
        if np.sum(foreground_mask == 1) > 0:
            for point_label, point_coord in zip(point_labels, point_coords):
                color = POINT_COLOR_POSITIVE if point_label == 1 else POINT_COLOR_NEGATIVE
                cv2.circle(
                    overlay, 
                    (int(point_coord[0]), int(point_coord[1])), 
                    POINT_RADIUS, color, -1
                )
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)

        # オーバーレイ画像（EdgeTAM 最終マスク版）
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = ((semantic_map > 0) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_cv, 1.0 - FOREGROUND_OVERLAY_ALPHA, colored_mask, FOREGROUND_OVERLAY_ALPHA, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay_edgetam.png"), overlay)

        print(f"Saving results to {save_dir}")

    # =========================================================================
    # 6. メイン実行エントリーポイント
    # =========================================================================

    def run(self, image_path: Union[str, Path]) -> None:
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if image_path.exists():
            print(f"====== 画像パス: {image_path.name} ======")
        else:
            print(f"====== 存在しません: {image_path.name} ======")
            return

        self.save_dir = Path(self.output_dir) / image_path.stem
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 画像読み込み
        image_raw = load_image(image_path)
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        # 画像前処理
        with Timer("画像前処理時間"):
            image_tensor = preprocess_image(image_raw, self.extractor.target_size, self.extractor.mean, self.extractor.std)

        # 特徴量抽出
        with Timer("特徴量抽出処理時間"):
            image_features, patch_features, fused_patch_features = self._feature_extraction(image_tensor)

        # 画像分類
        with Timer("画像分類処理時間"):
            self._image_classification(image_features)

        # CLIP + KMeans による初期前景マップの推定
        with Timer("粗い前景マスク取得処理時間"):
            foreground_map, foreground_mask, cluster_labels, cluster_index_list  = self._generate_foreground_map(patch_features, fused_patch_features)

        # セマンティックセグメンテーションのラベルを取得
        with Timer("セマンティックセグメンテーション処理時間"):
            labels_list, semantic_labels = self.semantic_segmentation(cluster_labels, cluster_index_list)

        # パッチ単位の粗い結果を整形（穴埋め・小さな島の統合・境界パッチのラベル統一）
        with Timer("粗いセグメンテーション整形処理時間"):
            raw_semantic_labels = semantic_labels
            foreground_mask, semantic_labels = self.refine_coarse_segmentation(semantic_labels, foreground_mask)

        # 粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合
        with Timer("詳細セマンティックセグメンテーション処理時間"):
            semantic_map, prompt_indices, point_coords, point_labels = self.refine_semantic_segmentation(
                image_raw, foreground_map, foreground_mask, semantic_labels, labels_list, width, height
            )

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        # 処理結果の保存
        self._save_results(
            save_dir=self.save_dir,
            image_raw=image_raw,
            width=width,
            height=height,
            semantic_map=semantic_map,
            class_names=self.semantic_class_names(prompt_indices),
            foreground_map=foreground_map,
            foreground_mask=foreground_mask,
            point_coords=point_coords,
            point_labels=point_labels,
            cluster_labels=cluster_labels,
            semantic_labels=semantic_labels,
            raw_semantic_labels=raw_semantic_labels,
        )