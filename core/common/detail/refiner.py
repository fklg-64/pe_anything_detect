"""粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合する。

流れ（refine）:
  1. 準備                  全領域に共通の入力（背景の負例点・ガイド・作業解像度）を作る          RegionEstimator.prepare
  2. クラスごとの推定      クラスごとに EdgeTAM で推定し、重なりはスコアの高いクラスを優先する    _refine_classes
  3. 取りこぼしの再推定    どのクラスにも覆われなかった粗い成分だけを、単独で推定し直す          _rescue_uncovered
  4. 後処理                小領域の除去 -> 前景内部の細い隙間を埋める -> 格子状の物体を 1 個ずつに分ける   mask_cleanup.py・_separate_lattice
  5. 最終の名前付け        EdgeTAM が切り出した（物体, クラス）の領域ごとに、その画素の代表トークンで名前を決め直す（前景のマスクは変えない）  naming.RegionNamer
  6. 元画像の大きさへ戻す

領域の推定（RegionEstimator.estimate）は何も書き換えないので、worker_threads があれば、次の領域のデコードと重ねて実行する。
推定の結果を semantic_map に反映するのは常にこのクラスで、クラス番号の順（取りこぼしは検出した順）に行うので、出力はスレッドの有無で変わらない。
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from core.common.detail.lattice_separator import LatticeSeparator
from core.common.detail.mask_cleanup import fill_narrow_gaps, remove_small_regions
from core.common.detail.naming import RegionNamer
from core.common.detail.region_estimator import RefineInput, RegionEstimator, RegionResult
from core.common.edgetam_session import EdgeTamSession, Features
from core.common.mask_utils import Component, connected_components, dilate, resize_mask
from core.common.parallel import JobGroup, Workers
from core.common.params import MIN_ISOLATED_PATCHES
from core.utils import Timer


@dataclass
class RefineState:
    """精緻化の途中経過。領域の推定結果を反映して更新していく（反映は SemanticRefiner だけが行う）。"""

    semantic_map: np.ndarray               # 作業解像度。0=背景、1~=前景クラス
    best_score: np.ndarray                 # 各画素で勝ったクラスのスコア
    point_coords: List[List[float]] = field(default_factory=list)   # EdgeTAMに与えた点（元画像の画素座標。全領域分）
    point_labels: List[int] = field(default_factory=list)
    logits: List[np.ndarray] = field(default_factory=list)          # EdgeTAM の低解像度ロジット（入力の 1/4）。領域ごと
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)  # 中間画像（名前 -> 0〜1 の配列）。保存は呼び出し側


class RefineOutput(NamedTuple):
    """SemanticRefiner.refine の結果。タプルとして取り出すこともできる。"""

    semantic_map: np.ndarray           # (height, width) uint8。0=背景、1~=前景クラス
    prompt_indices: List[int]          # クラス1~に対応する、前景プロンプトのインデックス
    point_coords: List[List[float]]    # EdgeTAMに与えたプロンプト点（画像座標、全クラス分）
    point_labels: List[int]            # 上の点のラベル（1=正例、0=負例）
    debug_images: Dict[str, np.ndarray]  # 精緻化の中間画像（名前 -> 0〜1 の配列。作業解像度かパッチ格子の大きさ）


class SemanticRefiner:
    def __init__(self, session: EdgeTamSession, feature_size: int, config: dict, workers: Workers, namer: RegionNamer):
        self.session = session
        self.estimator = RegionEstimator(session, feature_size, config)
        # 格子状に並んだ小物体を 1 個ずつに分ける（lattice_separator.py）
        self.lattice_separator = LatticeSeparator(session, config["foreground_threshold"])
        # 最終の名前付け（naming.py）
        self.namer = namer
        self.workers = workers

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
        feats: Features,
        patch_features: Optional[np.ndarray] = None,
    ) -> RefineOutput:
        """粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合する。

        クラスごとにEdgeTAMで前景マスクを推定し、重なり部分は
        「EdgeTAMの確信度 × 粗い前景らしさ(foreground_map)」が高いクラスを優先する。
        背景は全クラス共通でクラス0とし、前景は1から振る。
        どのクラスのマスクにも入らなかった粗い連結成分は、その成分だけで再推定する。

        resized: リサイズ済み（正規化前）の画像。EdgeTAMエンコーダの入力と、初期マスクの縁を整えるガイドになる。
        feats: EdgeTAMエンコーダの出力（粗い判定の前に SegmentationRunner が 1 回だけ実行したもの）。
        patch_features: 最終の名前付けの材料になる、PE のパッチ特徴（粗い段階の extract_features の 1 つめ）。
            None なら名前付けをせず、粗い段階のクラスのまま。

        Returns: RefineOutput（各フィールドの意味はそちら）。
        """
        foreground_binary = foreground_mask != 0
        semantic_ids: np.ndarray = np.unique(semantic_labels[foreground_binary])
        prompt_indices: List[int] = [int(labels_list[i]) for i in semantic_ids]
        if len(semantic_ids) == 0:
            return RefineOutput(np.zeros((height, width), dtype=np.uint8), prompt_indices, [], [], {})

        # 1. 準備
        inputs = self.estimator.prepare(resized, feats, foreground_map, foreground_binary, width, height)
        work_w, work_h = inputs.size
        state = RefineState(
            semantic_map=np.zeros((work_h, work_w), dtype=np.uint8),
            best_score=np.full((work_h, work_w), -1.0, dtype=np.float32),
        )

        # 2.・3. クラスごとに推定し、どのクラスにも入らなかった粗い連結成分は単独で再推定する
        class_components = self._refine_classes(inputs, state, semantic_ids, semantic_labels, foreground_binary)
        self._rescue_uncovered(inputs, state, class_components)
        state.debug_images["semantic_map_before_cleanup"] = state.semantic_map / max(len(semantic_ids), 1)

        # 4. 後処理: 重ね合わせ後のノイズ除去 -> クラス境界の割れ（前景の内部の細い隙間）を埋める
        remove_small_regions(state.semantic_map, inputs.min_area, self.workers)
        fill_narrow_gaps(state.semantic_map, foreground_binary)
        #    格子状に並んだ小物体（隙間が 1 パッチ未満で、1 つの塊になる）を、セルごとに 1 個ずつに分ける
        self._separate_lattice(inputs, state, resized, feats)

        # 5. EdgeTAM が切り出した物体ごとに、名前を決め直す（前景のマスクは変えない）
        if patch_features is not None:
            with Timer("    ┌ 最終の名前付け処理時間"):
                state.semantic_map, prompt_indices = self.namer.name(state.semantic_map, prompt_indices, patch_features)

        # 6. 元画像の大きさへ戻す（作業解像度が同じならそのまま）
        semantic_map = state.semantic_map
        if inputs.size != (width, height):
            semantic_map = cv2.resize(semantic_map, (width, height), interpolation=cv2.INTER_NEAREST)
        return RefineOutput(semantic_map, prompt_indices, state.point_coords, state.point_labels, state.debug_images)

    # =========================================================================
    # 2. クラスごとの推定
    # =========================================================================

    def _refine_classes(
        self,
        inputs: RefineInput,
        state: RefineState,
        semantic_ids: np.ndarray,
        semantic_labels: np.ndarray,
        foreground_binary: np.ndarray,
    ) -> Dict[int, List[Component]]:
        """クラスごとにEdgeTAMでマスクを推定し、スコアの高いクラスを優先して semantic_map に書き込む。

        先に全クラスの推定を投入し（worker_threads があれば、デコードと整形が重なって進む）、結果はクラス番号の順に反映する。

        Returns: クラス番号 -> 粗い連結成分のリスト（取りこぼしの再推定で使う）
        """
        class_components: Dict[int, List[Component]] = {}
        with JobGroup(self.workers) as jobs:
            queue = []  # (クラス番号, 粗いマスク, 推定の仕事)
            for class_id, semantic_id in enumerate(semantic_ids, start=1):
                coarse_mask = (semantic_labels == semantic_id) & foreground_binary
                comps = connected_components(coarse_mask)
                class_components[class_id] = comps

                main = self._main_components(comps)
                coarse_main = np.any([c.region for c in main], axis=0)
                job = partial(self.estimator.estimate, inputs, coarse_main, main, f"クラス{class_id}")
                queue.append((class_id, coarse_mask, jobs.submit(job)))

            for class_id, coarse_mask, job in queue:
                result = job.result()
                self._record_prompts(state, result)

                # 重なり部分はスコアの高いクラスを優先
                win = result.binary & (result.score > state.best_score)
                state.semantic_map[win] = class_id
                state.best_score[win] = result.score[win]

                state.debug_images[f"c{class_id}_coarse_mask"] = coarse_mask
                state.debug_images[f"c{class_id}_core"] = result.core
                state.debug_images[f"c{class_id}_edgetam_mask"] = result.binary
                state.debug_images[f"c{class_id}_score"] = result.score
        return class_components

    # =========================================================================
    # 3. 取りこぼしの再推定
    # =========================================================================

    def _rescue_uncovered(self, inputs: RefineInput, state: RefineState, class_components: Dict[int, List[Component]]) -> None:
        """取りこぼし防止: 最内点がどのクラスにも覆われていない粗い連結成分は、単独で再推定する。

        再推定の入力は semantic_map を読まないので、クラスの反映が済んだ時点で覆われていない成分は、先にまとめて推定を始める（投機的な推定）。
        結果は検出した順に反映し、反映の時点で（先の再推定に）覆われていた成分の結果は捨てる。
        """
        with JobGroup(self.workers) as jobs:
            queue = []  # (クラス番号, 成分, 最内点の作業解像度の座標, 推定の仕事)
            for class_id, comps in class_components.items():
                # 前景点は、成分自身ではなくクラスの本体の特徴を基準に選ぶ（成分が背景のパッチなら、物体らしい画素が無く、点も推定も無い）
                prototype = np.any([c.core for c in self._main_components(comps)], axis=0)
                for component in comps:
                    peak = self._peak_in_work_size(inputs, component)
                    if self._is_covered(state, peak):
                        continue
                    job = partial(
                        self.estimator.estimate, inputs, component.region, [component], f"クラス{class_id}(再推定)",
                        need_score=False, prototype=prototype,
                    )
                    queue.append((class_id, component, peak, jobs.submit(job)))

            for class_id, component, peak, job in queue:
                if self._is_covered(state, peak):  # 先の再推定で覆われたので、この成分は推定しなかったことにする
                    continue
                result = job.result()
                self._record_prompts(state, result)
                binary = result.binary
                if component.region.sum() < MIN_ISOLATED_PATCHES:
                    # 極小成分は、成分を1パッチ広げた範囲までに制限する
                    binary &= resize_mask(dilate(component.region), inputs.size)
                state.semantic_map[binary & (state.semantic_map == 0)] = class_id
                state.debug_images[f"c{class_id}_rescue_{component.peak[0]}_{component.peak[1]}"] = binary

    @staticmethod
    def _main_components(comps: List[Component]) -> List[Component]:
        """クラスの本体の成分。孤立した極小成分（背景クラスタの斑点）は、クラスの推定に混ぜない。

        混ぜると、斑点に置いた点を経由して物体のマスクが周囲の背景へ広がる。斑点は後段で個別に（範囲を制限して）扱う。
        """
        return [c for c in comps if c.region.sum() >= MIN_ISOLATED_PATCHES] or comps

    def _peak_in_work_size(self, inputs: RefineInput, component: Component) -> Tuple[int, int]:
        """成分の最内点（パッチ格子）に当たる、作業解像度の画素 (x, y)。"""
        work_w, work_h = inputs.size
        grid_h, grid_w = self.estimator.grid
        return (
            min(int((component.peak[0] + 0.5) * work_w / grid_w), work_w - 1),
            min(int((component.peak[1] + 0.5) * work_h / grid_h), work_h - 1),
        )

    @staticmethod
    def _is_covered(state: RefineState, peak: Tuple[int, int]) -> bool:
        """作業解像度の画素 peak=(x, y) が、すでにどれかのクラスに覆われているか。"""
        return state.semantic_map[peak[1], peak[0]] != 0

    @staticmethod
    def _record_prompts(state: RefineState, result: RegionResult) -> None:
        """領域の推定で EdgeTAM に与えた点とロジットを state に追記する。領域を反映する順に呼ぶ。"""
        state.point_coords.extend(result.coords)
        state.point_labels.extend(result.labels)
        if result.logit is not None:
            state.logits.append(result.logit)

    # =========================================================================
    # 4. 後処理
    # =========================================================================

    def _separate_lattice(self, inputs: RefineInput, state: RefineState, resized: np.ndarray, feats: Features) -> None:
        """格子状に並んだ物体があれば、semantic_map（in-place）をセルごとの物体に分ける（作動は稀）。"""
        if not state.logits:
            return
        lattice = self.lattice_separator.separate(
            state.semantic_map, resized, np.maximum.reduce(state.logits), feats, inputs.foreground_map
        )
        if lattice is not None:
            state.debug_images["lattice_instances"] = (lattice["instance_logit"] > 0).astype(np.float32)
