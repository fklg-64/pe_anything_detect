"""粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合する。"""

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from core.utils import Timer, normalize_image
from core.common.backend import EdgeTamBackend
from core.common.edgetam_helper import EdgeTamHelper
from core.common.image_layout import ImageLayout
from core.common.lattice_separator import LatticeSeparator
from core.common.mask_utils import (
    Component,
    clean_mask,
    connected_components,
    fill_narrow_gaps,
    guided_filter,
    patch_mean,
    patch_structure,
    prompt_points,
    remove_small_regions,
    resize_map,
    select_background_points,
)
from core.common.params import (
    EDGETAM_HIGH_RES_FEATURE_DOWNSCALE,
    EDGETAM_MASK_DOWNSCALE,
    GUIDED_FILTER_EPS,
    MASK_LOGIT_THRESHOLD,
    MIN_ISOLATED_PATCHES,
    MIN_REGION_PATCH_AREA,
    PRIOR_LOGIT_RANGE,
    PRIOR_NON_CORE_PROBABILITY,
    SCORE_PRIOR_WEIGHT,
)


@dataclass
class RefineContext:
    """1枚の画像を精緻化する間、各段階で共有する入力と、更新していく結果。"""

    # 入力（全クラス共通）
    feats: Tuple                           # EdgeTAMエンコーダの出力
    background_coords: List[List[float]]   # 背景の負例点（画像座標）
    foreground_map: np.ndarray             # 粗い前景らしさ（パッチ格子）
    width: int                             # 元画像サイズ
    height: int
    size: Tuple[int, int]                  # 作業解像度 (幅, 高さ)
    min_area: int                          # ノイズとして除去する最小面積（作業解像度の画素数）
    guide: np.ndarray                      # 初期マスクの縁を整えるガイド（入力画像の輝度。EdgeTAMの高解像度特徴量と同じ大きさ、float32、0〜1）
    grid: Tuple[int, int]                  # パッチ格子 (高さ, 幅)。画像の全体を覆う（パディングは含まない）
    input_hw: Tuple[int, int]              # モデルに入力した画像の内容の大きさ (高さ, 幅)。プロンプト点の座標変換に使う

    # 結果（段階ごとに更新する）
    semantic_map: np.ndarray               # 作業解像度。0=背景、1~=前景クラス
    best_score: np.ndarray                 # 各画素で勝ったクラスのスコア
    point_coords: List[List[float]] = field(default_factory=list)
    point_labels: List[int] = field(default_factory=list)
    logits: List[np.ndarray] = field(default_factory=list)  # EdgeTAM の低解像度ロジット（入力の 1/4）。クラス・再推定ごと
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)  # 中間画像（名前 -> 0〜1 の配列）。保存は呼び出し側


class RefineOutput(NamedTuple):
    """SemanticRefiner.refine の結果。タプルとして取り出すこともできる。"""

    semantic_map: np.ndarray           # (height, width) uint8。0=背景、1~=前景クラス
    prompt_indices: List[int]          # クラス1~に対応する、前景プロンプトのインデックス
    point_coords: List[List[float]]    # EdgeTAMに与えたプロンプト点（画像座標、全クラス分）
    point_labels: List[int]            # 上の点のラベル（1=正例、0=負例）
    debug_images: Dict[str, np.ndarray]  # 精緻化の中間画像（名前 -> 0〜1 の配列。作業解像度かパッチ格子の大きさ）


class SemanticRefiner:
    def __init__(self, edgetam: Union[EdgeTamBackend, EdgeTamHelper], feature_size: int, config: dict):
        self.edgetam = edgetam
        self.feature_size = feature_size
        self.foreground_points_per_class = config["edgetam_foreground_points_per_class"]
        self.background_point_count = config["edgetam_background_point_count"]
        self.refine_iterations = config["edgetam_refine_iterations"]
        self.semantic_max_side = config["semantic_max_side"]
        # 格子状に並んだ小物体を 1 個ずつに分ける（lattice_separator.py）
        self.lattice_separator = LatticeSeparator(edgetam, config["foreground_threshold"])
        # 初期マスクの縁を整えるガイデッドフィルタの半径 = 粗いマスクの信頼限界（1パッチ）が、高解像度特徴量の何画素か
        self.guide_radius: int = max(1, (edgetam.target_size // EDGETAM_HIGH_RES_FEATURE_DOWNSCALE) // feature_size)

    # =========================================================================
    # メインの流れ
    # =========================================================================

    def refine(
        self,
        resized: np.ndarray,
        foreground_map: np.ndarray,
        foreground_mask: np.ndarray,
        semantic_labels: np.ndarray,
        labels_list: np.ndarray,
        width: int,
        height: int,
        feats: Optional[Tuple] = None,
        layout: Optional[ImageLayout] = None,
    ) -> RefineOutput:
        """粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合する。

        クラスごとにEdgeTAMで前景マスクを推定し、重なり部分は
        「EdgeTAMの確信度 × 粗い前景らしさ(foreground_map)」が高いクラスを優先する。
        背景は全クラス共通でクラス0とし、前景は1から振る。
        どのクラスのマスクにも入らなかった粗い連結成分は、その成分だけで再推定する。

        resized: リサイズ済み（正規化前）の画像。EdgeTAMエンコーダの入力と、初期マスクの縁を整えるガイドになる。
        feats: EdgeTAMエンコーダの出力（layout がパディングを持つときは、有効な領域に切り出し済み）。
               粗い判定で既に計算しているときに渡すと、エンコーダを再実行しない（None なら実行する）。
        layout: モデル入力のレイアウト。None なら image_size 四方（従来どおり）。パッチ格子・ロジット・プロンプト点はすべて、
                パディングを除いた画像の内容（元画像の全体に対応する）を基準にする。

        Returns: RefineOutput（各フィールドの意味はそちら）。
        """
        foreground_binary = foreground_mask != 0
        semantic_ids: np.ndarray = np.unique(semantic_labels[foreground_binary])
        prompt_indices: List[int] = [int(labels_list[i]) for i in semantic_ids]
        if len(semantic_ids) == 0:
            return RefineOutput(np.zeros((height, width), dtype=np.uint8), prompt_indices, [], [], {})

        content = resized if layout is None else layout.content(resized)

        # EdgeTAMの特徴量抽出（全クラス共通）
        if feats is None:
            with Timer("    ┌ edgetam特徴量抽出"):
                feats = self.predict_encoder(resized)
                if layout is not None:
                    feats = layout.crop_edgetam(feats)

        ctx = self._make_context(content, feats, foreground_map, foreground_binary, width, height, layout)

        # クラスごとに推定し、どのクラスにも入らなかった粗い連結成分は単独で再推定する
        class_components = self._refine_classes(ctx, semantic_ids, semantic_labels, foreground_binary)
        self._rescue_uncovered(ctx, class_components)
        ctx.debug_images["semantic_map_before_cleanup"] = ctx.semantic_map / max(len(semantic_ids), 1)

        # 重ね合わせ後のノイズ除去
        remove_small_regions(ctx.semantic_map, ctx.min_area)
        # クラス境界の割れ（前景の内部の細い隙間）を埋める
        fill_narrow_gaps(ctx.semantic_map, foreground_binary)
        # 格子状に並んだ小物体（隙間が 1 パッチ未満で、1 つの塊になる）を、セルごとに 1 個ずつに分ける
        self._separate_lattice(ctx, content, feats)

        return RefineOutput(self._to_image_size(ctx), prompt_indices, ctx.point_coords, ctx.point_labels, ctx.debug_images)

    def _make_context(
        self,
        content: np.ndarray,
        feats: Tuple,
        foreground_map: np.ndarray,
        foreground_binary: np.ndarray,
        width: int,
        height: int,
        layout: Optional[ImageLayout],
    ) -> RefineContext:
        """全クラス共通の入力（背景の負例点・ガイド・作業解像度など）と、空の結果を持つ RefineContext を作る。"""
        # マスク演算は長辺が semantic_max_side 以下の作業解像度で行い、最後に元画像サイズへ戻す
        scale = min(1.0, self.semantic_max_side / max(width, height))
        work_w, work_h = max(1, round(width * scale)), max(1, round(height * scale))
        grid = (self.feature_size, self.feature_size) if layout is None else layout.valid_grid

        # 背景の負例点は全クラス共通
        background_coords = self.transform_coords(
            select_background_points(foreground_binary, self.background_point_count, foreground_map).tolist(), width, height, grid
        )
        return RefineContext(
            feats=feats,
            background_coords=background_coords,
            foreground_map=foreground_map,
            width=width,
            height=height,
            size=(work_w, work_h),
            # 粗いマスクの信頼限界は1パッチ。1パッチ未満の領域はノイズとして扱う
            min_area=max(1, int(MIN_REGION_PATCH_AREA * work_w * work_h / (grid[0] * grid[1]))),
            guide=self._make_guide(content),
            grid=grid,
            input_hw=content.shape[:2],
            semantic_map=np.zeros((work_h, work_w), dtype=np.uint8),
            best_score=np.full((work_h, work_w), -1.0, dtype=np.float32),
        )

    def _separate_lattice(self, ctx: RefineContext, content: np.ndarray, feats: Tuple) -> None:
        """格子状に並んだ物体があれば、semantic_map（in-place）をセルごとの物体に分ける（作動は稀）。"""
        if not ctx.logits:
            return
        lattice = self.lattice_separator.separate(
            ctx.semantic_map, content, np.maximum.reduce(ctx.logits), feats, ctx.foreground_map
        )
        if lattice is not None:
            ctx.debug_images["lattice_instances"] = (lattice["instance_logit"] > 0).astype(np.float32)

    @staticmethod
    def _to_image_size(ctx: RefineContext) -> np.ndarray:
        """作業解像度のセマンティックマップを、元画像の大きさへ戻す（同じ大きさならそのまま）。"""
        if ctx.size == (ctx.width, ctx.height):
            return ctx.semantic_map
        return cv2.resize(ctx.semantic_map, (ctx.width, ctx.height), interpolation=cv2.INTER_NEAREST)

    def _high_res_size(self, input_hw: Tuple[int, int]) -> Tuple[int, int]:
        """EdgeTAMの高解像度特徴量の大きさ (幅, 高さ)。cv2.resize に渡す向き。"""
        return input_hw[1] // EDGETAM_HIGH_RES_FEATURE_DOWNSCALE, input_hw[0] // EDGETAM_HIGH_RES_FEATURE_DOWNSCALE

    def _make_guide(self, content: np.ndarray) -> np.ndarray:
        """初期マスクの縁を整えるガイド。入力画像（パディングを除いた内容）の輝度を、EdgeTAMの高解像度特徴量の大きさに縮小したもの（float32、0〜1）。"""
        return cv2.resize(content, self._high_res_size(content.shape[:2]), interpolation=cv2.INTER_AREA).astype(np.float32).mean(axis=-1) / 255.0

    def _refine_classes(
        self,
        ctx: RefineContext,
        semantic_ids: np.ndarray,
        semantic_labels: np.ndarray,
        foreground_binary: np.ndarray,
    ) -> dict:
        """クラスごとにEdgeTAMでマスクを推定し、スコアの高いクラスを優先して semantic_map に書き込む。

        Returns: クラス番号 -> 粗い連結成分のリスト（取りこぼしの再推定で使う）
        """
        class_components = {}
        for class_id, semantic_id in enumerate(semantic_ids, start=1):
            coarse_mask = (semantic_labels == semantic_id) & foreground_binary
            comps = connected_components(coarse_mask)
            class_components[class_id] = comps

            # 孤立した極小成分（背景クラスタの斑点）はクラスの推定に混ぜない。混ぜると、斑点に置いた点を
            # 経由して物体のマスクが周囲の背景へ広がる。斑点は後段で個別に（範囲を制限して）扱う
            main = [c for c in comps if c.region.sum() >= MIN_ISOLATED_PATCHES] or comps
            coarse_main = np.any([c.region for c in main], axis=0)
            binary, score, core = self._refine_region(ctx, coarse_main, main, f"クラス{class_id}")

            # 重なり部分はスコアの高いクラスを優先
            win = binary & (score > ctx.best_score)
            ctx.semantic_map[win] = class_id
            ctx.best_score[win] = score[win]

            ctx.debug_images[f"c{class_id}_coarse_mask"] = coarse_mask
            ctx.debug_images[f"c{class_id}_core"] = core
            ctx.debug_images[f"c{class_id}_edgetam_mask"] = binary
            ctx.debug_images[f"c{class_id}_score"] = score
        return class_components

    def _rescue_uncovered(self, ctx: RefineContext, class_components: dict) -> None:
        """取りこぼし防止: 最内点がどのクラスにも覆われていない粗い連結成分は、単独で再推定する。"""
        work_w, work_h = ctx.size
        for class_id, comps in class_components.items():
            for component in comps:
                peak, region = component.peak, component.region
                grid_h, grid_w = ctx.grid
                px = min(int((peak[0] + 0.5) * work_w / grid_w), work_w - 1)
                py = min(int((peak[1] + 0.5) * work_h / grid_h), work_h - 1)
                if ctx.semantic_map[py, px] != 0:
                    continue
                binary, _, _ = self._refine_region(ctx, region, [component], f"クラス{class_id}(再推定)")
                if region.sum() < MIN_ISOLATED_PATCHES:
                    # 極小成分は、成分を1パッチ広げた範囲までに制限する
                    allowed = binary_dilation(region, structure=patch_structure())
                    binary &= resize_map(allowed, ctx.size, cv2.INTER_NEAREST) > 0
                ctx.semantic_map[binary & (ctx.semantic_map == 0)] = class_id
                ctx.debug_images[f"c{class_id}_rescue_{peak[0]}_{peak[1]}"] = binary

    def _refine_region(
        self, ctx: RefineContext, coarse_mask: np.ndarray, comps: List[Component], tag: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """粗いマスクに対するEdgeTAMの2値マスク(作業解像度)・重なり解決用スコア・coreを返す。

        EdgeTAMに与えた点は ctx.point_coords / point_labels に追記する。
        """
        core = np.zeros_like(coarse_mask, dtype=bool)
        for component in comps:
            core |= component.core
        coarse_map = patch_mean(coarse_mask)
        prior_map = self._build_prior_map(ctx, coarse_mask, coarse_map, core)

        fg_points = prompt_points(core, [c.peak for c in comps], self.foreground_points_per_class)
        logit, coords, labels = self._estimate_class_logit(
            ctx.feats, prior_map, fg_points, ctx.background_coords, ctx.width, ctx.height, tag, ctx.grid, ctx.input_hw
        )
        ctx.point_coords.extend(coords)
        ctx.point_labels.extend(labels)
        if logit is not None:
            ctx.logits.append(logit)

        # 粗い前景らしさ（粗いマスクの周辺平均 × 前景確信度）
        coarse_foreground_score = (coarse_map * ctx.foreground_map).astype(np.float32)
        if logit is None:
            coarse_full = resize_map(coarse_mask, ctx.size, cv2.INTER_NEAREST) > 0
            return coarse_full, resize_map(coarse_foreground_score, ctx.size, cv2.INTER_LINEAR), core

        logit_full = cv2.resize(logit.astype(np.float32), ctx.size, interpolation=cv2.INTER_LINEAR)
        binary = clean_mask(logit_full > MASK_LOGIT_THRESHOLD, ctx.min_area).astype(bool)

        # 重なり解決用スコア = EdgeTAMの確信度 × ((1-w) + w × 粗い前景らしさ)
        confidence = 1.0 / (1.0 + np.exp(-logit.astype(np.float32)))
        prior_low = cv2.resize(coarse_foreground_score, confidence.shape[::-1], interpolation=cv2.INTER_LINEAR)
        score = confidence * ((1.0 - SCORE_PRIOR_WEIGHT) + SCORE_PRIOR_WEIGHT * prior_low)
        return binary, resize_map(score, ctx.size, cv2.INTER_LINEAR), core

    def _build_prior_map(
        self, ctx: RefineContext, coarse_mask: np.ndarray, coarse_map: np.ndarray, core: np.ndarray
    ) -> np.ndarray:
        """EdgeTAMに初期マスクとして与える、前景確率のマップ（高解像度特徴量と同じ大きさ、0〜1）を作る。

        core内は前景、core外の粗いマスクとその外側1パッチ(信頼限界)は不明(PRIOR_NON_CORE_PROBABILITY)、それより外は背景。
        外側1パッチを背景にすると、粗いマスクが物体より内側に入ったとき（縁のパッチは物体と背景のトークンが混ざり落ちやすい）縁が削れる。
        不明の縁は、入力画像の縁に沿って物体側・背景側へ倒す（高解像度特徴量の大きさに拡大してガイデッドフィルタ）。
        パッチ格子の縛りと、小物体で外周が背景へ膨らむのを避けるため。

        coarse_map: coarse_mask のパッチ近傍平均（patch_mean）。
        """
        core_map = patch_mean(core)
        unknown = np.maximum(coarse_map, binary_dilation(coarse_mask, patch_structure()).astype(np.float32))
        prior_map = np.maximum(core_map, PRIOR_NON_CORE_PROBABILITY * unknown)
        upsampled = cv2.resize(prior_map, self._high_res_size(ctx.input_hw), interpolation=cv2.INTER_LINEAR)
        return np.clip(guided_filter(ctx.guide, upsampled, self.guide_radius, GUIDED_FILTER_EPS), 0.0, 1.0)

    def _estimate_class_logit(
        self,
        feats: Tuple,
        prior_map: np.ndarray,
        foreground_points: np.ndarray,
        background_coords: List[List[float]],
        width: int,
        height: int,
        tag: str,
        grid: Optional[Tuple[int, int]] = None,
        input_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], List[List[float]], List[int]]:
        """正例点と共通の負例点でEdgeTAMを反復実行し、低解像度ロジット(128x128)を返す。

        prior_map: 初期マスクの前景確率（EdgeTAMの高解像度特徴量と同じ大きさ）。ロジットは (2p-1)*PRIOR_LOGIT_RANGE（0.5=不明）。
        grid, input_hw: パッチ格子 (高さ, 幅) と、モデルに入力した画像の内容の大きさ (高さ, 幅)。None なら image_size 四方。
        """
        coords = self.transform_coords(foreground_points.tolist(), width, height, grid)
        labels = [1] * len(coords)
        coords = coords + background_coords
        labels = labels + [0] * len(background_coords)

        high_res_size = self._high_res_size(input_hw or (self.edgetam.target_size, self.edgetam.target_size))
        mask_input = cv2.resize(
            (2.0 * prior_map - 1.0) * PRIOR_LOGIT_RANGE, high_res_size, interpolation=cv2.INTER_LINEAR,
        )[None, ...]

        # 前回のロジットを入力にして再推定を繰り返す
        logit: Optional[np.ndarray] = None
        for it in range(self.refine_iterations):
            with Timer(f"    ┌ edgetam {tag} {it + 1}回目マスク推定"):
                _, scores, logits = self.predict_decoder(
                    point_coords=coords,
                    point_labels=labels,
                    feats=feats,
                    height=height,
                    width=width,
                    mask_input=mask_input,
                    input_hw=input_hw,
                )
            if len(scores) == 0:
                break
            best_idx = int(np.argmax(scores))
            mask_input = logits[best_idx:best_idx + 1]
            logit = logits[best_idx]
        return logit, coords, labels

    # =========================================================================
    # EdgeTAM の実行と座標変換
    # =========================================================================

    def predict_encoder(self, resized: np.ndarray) -> Tuple:
        """EdgeTAM Encoder の実行。resized は resize_image 済み（正規化前）の画像。"""
        return self.edgetam.encode(
            normalize_image(resized, self.edgetam.mean, self.edgetam.std)
        )

    def predict_decoder(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Tuple,
        height: int,
        width: int,
        mask_input: Optional[np.ndarray] = None,
        input_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[None, np.ndarray, np.ndarray]:
        """EdgeTAM Decoder の実行。(masks=None, scores, 低解像度logits) を返す。

        masks は使わないので計算しない（リサイズ・二値化を省く）。
        """
        if mask_input is None:
            mask_logits_tensor = np.array([0], dtype=np.float32)
        else:
            mask_logits_tensor = mask_input

        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self.edgetam.preprocess(
            (height, width),
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
            mask_logits=mask_logits_tensor,
            input_hw=input_hw,
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
            return self.edgetam.postprocess(
                low_res_masks, iou_predictions, (height // EDGETAM_MASK_DOWNSCALE, width // EDGETAM_MASK_DOWNSCALE),
                need_masks=False,
            )

    def transform_coords(
        self, pts: List[List[int]], width: int, height: int, grid: Optional[Tuple[int, int]] = None
    ) -> List[List[float]]:
        """パッチ格子の座標 (x, y) を、各パッチの中心に当たる元画像の座標へ変換する。grid は (高さ, 幅)。None なら正方形の格子。"""
        grid_h, grid_w = grid or (self.feature_size, self.feature_size)
        return [
            [(x + 0.5) * width / grid_w, (y + 0.5) * height / grid_h]
            for x, y in pts
        ]
