"""1 つの領域（クラス、または取りこぼしの成分）を EdgeTAM で推定する。

粗いマスク（パッチ格子）から、
  ① 前景の点を物体の上に選び、② 初期マスク（事前確率）を作り、③ EdgeTAM のデコードを反復し、④ 作業解像度の 2 値マスクとスコアにする。
推定は入力（RefineInput）を読むだけで何も書き換えないので、別スレッドで重ねて実行できる（結果の反映は SemanticRefiner の仕事）。
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from core.utils import Timer
from core.common.edgetam_session import EdgeTamSession, Features
from core.common.mask_utils import (
    Component,
    clean_mask,
    dilate,
    guided_filter,
    patch_mean,
    prompt_points,
    resize_map,
    resize_mask,
    select_background_points,
)
from core.common.object_points import (
    background_features,
    normalize_features,
    object_prompt_points,
    objectness_map,
)
from core.common.params import (
    EDGETAM_FEATURE_STRIDES,
    GUIDED_FILTER_EPS,
    PRIOR_LOGIT_RANGE,
    SCORE_PRIOR_WEIGHT,
)


@dataclass(frozen=True)
class RefineInput:
    """1 枚の画像の精緻化で、全領域に共通の入力（読むだけ）。"""

    feats: Features                           # EdgeTAMエンコーダの出力
    background_coords: List[List[float]]      # 背景の負例点（元画像の画素座標）
    foreground_map: np.ndarray                # 粗い前景らしさ（パッチ格子）
    width: int                                # 元画像の大きさ
    height: int
    size: Tuple[int, int]                     # 作業解像度 (幅, 高さ)。マスク演算はこの大きさで行い、最後に元画像の大きさへ戻す
    min_area: int                             # ノイズとして除去する最小面積（作業解像度の画素数）
    guide: np.ndarray                         # 初期マスクの縁を整えるガイド（入力画像の輝度。高解像度特徴量と同じ大きさ、float32、0〜1）
    normalized_feats: List[np.ndarray]        # EdgeTAMの特徴量3層を画素ごとにL2正規化したもの（前景の点を物体の上に選ぶのに使う）
    background_means: Optional[List[np.ndarray]]  # 上の各層の、背景の特徴の平均。作れなければ None


class RegionResult(NamedTuple):
    """1 つの領域の推定結果。"""

    binary: np.ndarray                 # EdgeTAM の 2 値マスク（作業解像度）
    score: Optional[np.ndarray]        # 重なり解決用スコア（need_score が偽なら None）
    core: np.ndarray                   # 領域の core（パッチ格子）
    coords: List[List[float]]          # EdgeTAM に与えた点（元画像の画素座標。正例 -> 負例の順）
    labels: List[int]                  # 上の点のラベル（1=正例、0=負例）
    logit: Optional[np.ndarray]        # 低解像度ロジット（デコーダーの結果が無ければ None）


class RegionEstimator:
    def __init__(self, session: EdgeTamSession, feature_size: int, config: dict):
        self.session = session
        self.foreground_points_per_class = config["edgetam_foreground_points_per_class"]
        self.background_point_count = config["edgetam_background_point_count"]
        self.refine_iterations = config["edgetam_refine_iterations"]
        self.semantic_max_side = config["semantic_max_side"]
        # パッチ格子 (高さ, 幅)。入力画像の全体を覆う
        self.grid: Tuple[int, int] = (feature_size, feature_size)
        # EdgeTAMの高解像度特徴量の大きさ (幅, 高さ)。cv2.resize に渡す向き。初期マスクとガイドの大きさ
        high_res = session.target_size // EDGETAM_FEATURE_STRIDES[0]
        self.high_res_size: Tuple[int, int] = (high_res, high_res)
        # 初期マスクの縁を整えるガイデッドフィルタの半径 = 粗いマスクの信頼限界（1パッチ）が、高解像度特徴量の何画素か
        self.guide_radius: int = max(1, high_res // feature_size)

    # =========================================================================
    # 画像ごとの準備（全領域に共通）
    # =========================================================================

    def prepare(
        self,
        resized: np.ndarray,
        feats: Features,
        foreground_map: np.ndarray,
        foreground_binary: np.ndarray,
        width: int,
        height: int,
    ) -> RefineInput:
        """全領域に共通の入力（背景の負例点・ガイド・作業解像度など）を作る。"""
        # マスク演算は長辺が semantic_max_side 以下の作業解像度で行い、最後に元画像サイズへ戻す
        scale = min(1.0, self.semantic_max_side / max(width, height))
        work_w, work_h = max(1, round(width * scale)), max(1, round(height * scale))
        grid_h, grid_w = self.grid

        # 前景の点を物体の上に選ぶための、正規化した特徴量と背景の特徴（全領域に共通なので、画像に 1 回だけ求める）
        normalized_feats = normalize_features(feats)
        background_means = background_features(normalized_feats, foreground_binary)

        # 背景の負例点は全領域に共通
        background_coords = self.transform_coords(
            select_background_points(foreground_binary, self.background_point_count, foreground_map).tolist(), width, height
        )
        guide = cv2.resize(resized, self.high_res_size, interpolation=cv2.INTER_AREA).astype(np.float32).mean(axis=-1) / 255.0
        return RefineInput(
            feats=feats,
            background_coords=background_coords,
            foreground_map=foreground_map,
            width=width,
            height=height,
            size=(work_w, work_h),
            # 粗いマスクの信頼限界は1パッチ。1パッチ未満の領域はノイズとして扱う
            min_area=max(1, int(work_w * work_h / (grid_h * grid_w))),
            guide=guide,
            normalized_feats=normalized_feats,
            background_means=background_means,
        )

    def transform_coords(self, pts: List[List[int]], width: int, height: int) -> List[List[float]]:
        """パッチ格子の座標 (x, y) を、各パッチの中心に当たる元画像の座標へ変換する。"""
        grid_h, grid_w = self.grid
        return [
            [(x + 0.5) * width / grid_w, (y + 0.5) * height / grid_h]
            for x, y in pts
        ]

    # =========================================================================
    # 領域の推定
    # =========================================================================

    def estimate(
        self, inputs: RefineInput, coarse_mask: np.ndarray, comps: List[Component], tag: str, need_score: bool = True
    ) -> RegionResult:
        """1 つの領域の推定（前景点・事前マスク・EdgeTAM のデコード・マスクの整形）。inputs は読むだけで、変えない。

        coarse_mask: 領域の粗いマスク（パッチ格子）。comps: その連結成分。tag: 処理時間の表示名。
        need_score: 偽なら、重なり解決用のスコアを求めない（重なりを解決しない取りこぼしの再推定では使わない）。
        """
        core = np.zeros_like(coarse_mask, dtype=bool)
        for component in comps:
            core |= component.core
        coarse_map = patch_mean(coarse_mask)
        prior_map = self.build_prior_map(inputs, coarse_mask, coarse_map, core)

        fg_points = self.foreground_points(inputs, coarse_mask, comps, core)
        logit, coords, labels = self.estimate_logit(inputs, prior_map, fg_points, tag)

        # 粗い前景らしさ（粗いマスクの周辺平均 × 前景確信度）
        coarse_foreground_score = (coarse_map * inputs.foreground_map).astype(np.float32) if need_score else None
        if logit is None:
            coarse_full = resize_mask(coarse_mask, inputs.size)
            score = resize_map(coarse_foreground_score, inputs.size, cv2.INTER_LINEAR) if need_score else None
            return RegionResult(coarse_full, score, core, coords, labels, None)

        logit_full = cv2.resize(logit.astype(np.float32), inputs.size, interpolation=cv2.INTER_LINEAR)
        binary = clean_mask(logit_full > 0, inputs.min_area).view(bool)
        if not need_score:
            return RegionResult(binary, None, core, coords, labels, logit)

        # 重なり解決用スコア = EdgeTAMの確信度 × ((1-w) + w × 粗い前景らしさ)
        confidence = 1.0 / (1.0 + np.exp(-logit.astype(np.float32)))
        prior_low = cv2.resize(coarse_foreground_score, confidence.shape[::-1], interpolation=cv2.INTER_LINEAR)
        score = confidence * ((1.0 - SCORE_PRIOR_WEIGHT) + SCORE_PRIOR_WEIGHT * prior_low)
        return RegionResult(binary, resize_map(score, inputs.size, cv2.INTER_LINEAR), core, coords, labels, logit)

    def foreground_points(
        self, inputs: RefineInput, coarse_mask: np.ndarray, comps: List[Component], core: np.ndarray
    ) -> np.ndarray:
        """領域の正例点（パッチ座標）。EdgeTAM の高解像度特徴量で物体らしさを求め、粗いマスクの縁も含めて物体の上の画素から選ぶ。

        背景の特徴が作れないとき（前景が画像の全体など）は、core 上のパッチの中心から選ぶ。
        """
        peaks = [c.peak for c in comps]
        score = None
        if inputs.background_means is not None:
            score = objectness_map(inputs.normalized_feats, inputs.background_means, core)
        if score is None:
            return prompt_points(core, peaks, self.foreground_points_per_class)
        return object_prompt_points(score, coarse_mask, peaks, self.foreground_points_per_class, self.grid)

    def build_prior_map(
        self, inputs: RefineInput, coarse_mask: np.ndarray, coarse_map: np.ndarray, core: np.ndarray
    ) -> np.ndarray:
        """EdgeTAMに初期マスクとして与える、前景確率のマップ（高解像度特徴量と同じ大きさ、0〜1）を作る。

        core内は前景、core外の粗いマスクとその外側1パッチ(信頼限界)は不明(確率 0.5)、それより外は背景。
        外側1パッチを背景にすると、粗いマスクが物体より内側に入ったとき（縁のパッチは物体と背景のトークンが混ざり落ちやすい）縁が削れる。
        不明の縁は、入力画像の縁に沿って物体側・背景側へ倒す（高解像度特徴量の大きさに拡大してガイデッドフィルタ）。
        パッチ格子の縛りと、小物体で外周が背景へ膨らむのを避けるため。

        coarse_map: coarse_mask のパッチ近傍平均（patch_mean）。
        """
        core_map = patch_mean(core)
        unknown = np.maximum(coarse_map, dilate(coarse_mask).astype(np.float32))
        prior_map = np.maximum(core_map, 0.5 * unknown)
        upsampled = cv2.resize(prior_map, self.high_res_size, interpolation=cv2.INTER_LINEAR)
        return np.clip(guided_filter(inputs.guide, upsampled, self.guide_radius, GUIDED_FILTER_EPS), 0.0, 1.0)

    def estimate_logit(
        self, inputs: RefineInput, prior_map: np.ndarray, foreground_points: np.ndarray, tag: str
    ) -> Tuple[Optional[np.ndarray], List[List[float]], List[int]]:
        """正例点と共通の負例点でEdgeTAMを反復実行し、(低解像度ロジット(128x128), 点の座標, 点のラベル) を返す。

        prior_map: 初期マスクの前景確率（EdgeTAMの高解像度特徴量と同じ大きさ）。ロジットは (2p-1)*PRIOR_LOGIT_RANGE（0.5=不明）。
        前回の最良のロジットを次の初期マスクにして、edgetam_refine_iterations 回繰り返す。
        """
        coords = self.transform_coords(foreground_points.tolist(), inputs.width, inputs.height)
        labels = [1] * len(coords)
        coords = coords + inputs.background_coords
        labels = labels + [0] * len(inputs.background_coords)

        mask_input = cv2.resize(
            (2.0 * prior_map - 1.0) * PRIOR_LOGIT_RANGE, self.high_res_size, interpolation=cv2.INTER_LINEAR,
        )[None, ...]

        logit: Optional[np.ndarray] = None
        for it in range(self.refine_iterations):
            with Timer(f"    ┌ edgetam {tag} {it + 1}回目マスク推定"):
                scores, logits = self.session.decode_prompts(
                    coords, labels, inputs.feats, inputs.height, inputs.width, mask_input
                )
            if len(scores) == 0:
                break
            best_idx = int(np.argmax(scores))
            mask_input = logits[best_idx:best_idx + 1]
            logit = logits[best_idx]
        return logit, coords, labels
