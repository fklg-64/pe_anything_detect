"""格子状に並んだ小物体を、セルごとの箱プロンプトで 1 個ずつ分ける。

クラス単位のデコードは、隙間が 1 パッチ未満の物体の集まりを 1 つの塊にする（EdgeTAM は点に従い、事前マスクはほとんど効かない）。
格子（lattice.py）が見つかったら、各セルの箱（隣の物体までの距離を一辺とする）を箱プロンプトとして、物体を 1 個ずつデコードする。
デコーダーは入力の大きさが可変なので、特徴を箱の周りの窓（LATTICE_WINDOW_TOKENS）で切り出して全セルを一括で実行する（画像全体を
セルの数だけデコードするより約 10 倍速い）。
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.utils import Timer
from core.common import lattice as lattice_utils
from core.common.edgetam_session import EdgeTamSession
from core.common.params import (
    EDGETAM_FEATURE_STRIDES,
    EDGETAM_LOGIT_CLIP,
    LATTICE_MIN_PREDICTED_IOU,
    LATTICE_PHASE_PROBES,
    LATTICE_WINDOW_TOKENS,
)


class LatticeSeparator:
    """semantic_map（作業解像度）の格子状の前景を、セルごとの物体に分ける。"""

    def __init__(self, session: EdgeTamSession, foreground_threshold: float):
        self.session = session
        self.foreground_threshold = foreground_threshold

    # =========================================================================
    # メインの流れ
    # =========================================================================

    def separate(
        self,
        semantic_map: np.ndarray,
        content: np.ndarray,
        logit: np.ndarray,
        feats: Tuple[np.ndarray, np.ndarray, np.ndarray],
        foreground_map: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """格子が見つかれば semantic_map を in-place で更新して、中間結果（デバッグ用）を返す。見つからなければ何もせず None。

        content: モデルに入力した画像（RGB, uint8）。logit: 全クラスの最大の低解像度ロジット（入力の 1/4）。
        feats: EdgeTAM エンコーダの出力（high_res_feats0, high_res_feats1, image_embed）。foreground_map: 粗い前景確信度（パッチ格子）。
        """
        region = semantic_map > 0
        with Timer("    ┌ 格子の検出"):
            lattice = lattice_utils.detect(content, region)
        mask = logit > 0
        if lattice is None or not mask.any():
            return None

        input_hw = content.shape[:2]
        half = np.array(lattice_utils.cell_half_size(lattice))
        phase = lattice_utils.find_phase(lattice, logit, mask, input_hw)
        centers = lattice_utils.cell_centers(lattice, phase, region, input_hw)
        if len(centers) == 0:
            return None
        centers = self._correct_phase(feats, lattice, phase, centers, region, half, input_hw)
        if len(centers) == 0:
            return None

        with Timer("    ┌ 格子のセルごとのデコード"):
            low, scores, origins = self._decode(feats, centers, half)
        # デコーダーの予測 IoU が、物体として当てはまっているかの確認。セルの 4 分の 1 以上で予測 IoU が LATTICE_MIN_PREDICTED_IOU 未満（下位 4 分位点が未満）なら、
        # 格子に見えたのは物体の並びではない（背景の模様など）ので、何も変えない。個々のセルも、予測 IoU が低ければそのセルは変えない
        confidence = scores.max(axis=1)
        if np.percentile(confidence, 25) < LATTICE_MIN_PREDICTED_IOU:
            return None
        instance_logit, owner = self._compose(low, scores, origins, input_hw, semantic_map.shape, confidence >= LATTICE_MIN_PREDICTED_IOU)
        self._apply(semantic_map, instance_logit, owner, centers, half, input_hw, foreground_map)
        return {"instance_logit": instance_logit, "centers": centers, "lattice": lattice}

    def _correct_phase(
        self,
        feats: Tuple[np.ndarray, np.ndarray, np.ndarray],
        lattice: lattice_utils.Lattice,
        phase: np.ndarray,
        centers: np.ndarray,
        region: np.ndarray,
        half: np.ndarray,
        input_hw: Tuple[int, int],
    ) -> np.ndarray:
        """格子の位相を、一部のセルのデコード結果の重心で直して、セルの中心 (N, 2) を作り直す。

        箱プロンプトのデコード結果は物体の中心に寄るので、等間隔に選んだ LATTICE_PHASE_PROBES 個以内のセルをデコードし、
        重心とのずれ（格子の係数で、±0.5 周期に畳んだ中央値）を位相に足す。重心が 3 個未満しか求まらなければ直さない。
        """
        probes = centers[np.linspace(0, len(centers) - 1, min(len(centers), LATTICE_PHASE_PROBES)).astype(int)]
        with Timer("    ┌ 格子の位相の補正"):
            low, scores, origins = self._decode(feats, probes, half)
        centroids = self._centroids(low, scores, origins)
        found = ~np.isnan(centroids[:, 0])
        if found.sum() < 3:
            return centers
        grid_basis = lattice_utils.basis(lattice)
        offset = np.linalg.solve(grid_basis, (centroids[found] - probes[found]).T).T
        offset -= np.round(offset)
        phase = phase + np.median(offset, axis=0) @ grid_basis.T
        return lattice_utils.cell_centers(lattice, phase, region, input_hw)

    # =========================================================================
    # セルごとのデコード
    # =========================================================================

    def _decode(
        self, feats: Tuple[np.ndarray, np.ndarray, np.ndarray], centers: np.ndarray, half: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """セルの箱（中心 ± half。入力画像の画素）を、箱の周りの窓で切り出した特徴で一括デコードする。

        返り値: (低解像度ロジット (N, 候補数, h, w), 予測 IoU (N, 候補数), 窓の左上（入力画像の画素 (x, y)）(N, 2))。
        """
        feats0, feats1, embed = feats
        stride0, stride1, embed_stride = EDGETAM_FEATURE_STRIDES
        grid_h, grid_w = embed.shape[2:]
        win = min(LATTICE_WINDOW_TOKENS, grid_h, grid_w)
        ratio1, ratio0 = embed_stride // stride1, embed_stride // stride0
        windows0, windows1, windows_e, coords, origins = [], [], [], [], []
        for cx, cy in centers:
            tx = int(np.clip(round(cx / embed_stride - win / 2), 0, grid_w - win))
            ty = int(np.clip(round(cy / embed_stride - win / 2), 0, grid_h - win))
            windows_e.append(embed[0, :, ty:ty + win, tx:tx + win])
            windows1.append(feats1[0, :, ty * ratio1:(ty + win) * ratio1, tx * ratio1:(tx + win) * ratio1])
            windows0.append(feats0[0, :, ty * ratio0:(ty + win) * ratio0, tx * ratio0:(tx + win) * ratio0])
            ox, oy = tx * embed_stride, ty * embed_stride
            coords.append([[cx - half[0] - ox, cy - half[1] - oy], [cx + half[0] - ox, cy + half[1] - oy]])
            origins.append((ox, oy))
        n = len(centers)
        side = win * ratio0
        low, scores = self.session.decode_batch(
            np.stack(windows0), np.stack(windows1), np.stack(windows_e),
            np.array(coords, np.float32), np.tile(np.array([[2, 3]], np.int32), (n, 1)),  # ラベル 2・3 = 箱の左上・右下
            np.zeros((n, 1, side, side), np.float32), np.zeros((n,), np.float32),
        )
        return np.clip(low, -EDGETAM_LOGIT_CLIP, EDGETAM_LOGIT_CLIP), scores, np.array(origins, np.float64)

    @staticmethod
    def _best(low: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """各セルの、予測 IoU が最大の候補のロジット (N, h, w)。"""
        return low[np.arange(len(low)), np.argmax(scores, axis=1)]

    def _centroids(self, low: np.ndarray, scores: np.ndarray, origins: np.ndarray) -> np.ndarray:
        """各セルのデコード結果の重心（入力画像の画素 (x, y)）。マスクが空なら nan。"""
        best = self._best(low, scores)
        out = np.full((len(low), 2), np.nan)
        for k, mask in enumerate(best > 0):
            if mask.any():
                ys, xs = np.nonzero(mask)
                out[k] = (origins[k][0] + (xs.mean() + 0.5) * EDGETAM_FEATURE_STRIDES[0], origins[k][1] + (ys.mean() + 0.5) * EDGETAM_FEATURE_STRIDES[0])
        return out

    def _compose(
        self, low: np.ndarray, scores: np.ndarray, origins: np.ndarray, input_hw: Tuple[int, int], size: Tuple[int, int], use: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """セルごとのロジット（use が真のセルだけ）を作業解像度 size=(高さ, 幅) の 1 枚（各画素で最大）にする。もう 1 枚は、その画素で勝ったセルの番号（無ければ -1）。"""
        best = self._best(low, scores)
        height, width = size
        scale_x, scale_y = width / input_hw[1], height / input_hw[0]
        logit = np.full((height, width), -EDGETAM_LOGIT_CLIP, np.float32)
        owner = np.full((height, width), -1, np.int32)
        window = best.shape[-1] * EDGETAM_FEATURE_STRIDES[0]  # 窓の一辺（入力画像の画素）
        win_w, win_h = max(1, round(window * scale_x)), max(1, round(window * scale_y))
        for k in np.flatnonzero(use):
            x0, y0 = int(round(origins[k][0] * scale_x)), int(round(origins[k][1] * scale_y))
            patch = cv2.resize(best[k].astype(np.float32), (win_w, win_h), interpolation=cv2.INTER_LINEAR)
            h, w = min(win_h, height - y0), min(win_w, width - x0)
            region_logit, region_owner = logit[y0:y0 + h, x0:x0 + w], owner[y0:y0 + h, x0:x0 + w]
            wins = patch[:h, :w] > region_logit
            region_logit[wins] = patch[:h, :w][wins]
            region_owner[wins] = k
        return logit, owner

    # =========================================================================
    # semantic_map への反映
    # =========================================================================

    def _apply(
        self,
        semantic_map: np.ndarray,
        instance_logit: np.ndarray,
        owner: np.ndarray,
        centers: np.ndarray,
        half: np.ndarray,
        input_hw: Tuple[int, int],
        foreground_map: np.ndarray,
    ) -> None:
        """セルごとの物体マスクで、そのセルの箱の中のクラスの前景を置き換える（semantic_map を in-place で更新）。

        セルの物体マスク（そのセルが勝った画素のうち、箱の中でロジットが正のもの）が
        - 空（デコードに失敗）なら、そのセルは変えない。
        - それ以外は、箱の中のそのクラスの前景を消して、物体マスクを置く。
        - ただし粗い前景確信度の平均が閾値以下のセルは、物体の無い穴（例: 空のスロット）なので、置かずに消す。粗い判定が格子全体で信頼できるとき
          （確信度の中央値が閾値を超えるとき）に限る。当てにならない格子では、物体まで消すので使わない。
        """
        height, width = semantic_map.shape
        scale_x, scale_y = width / input_hw[1], height / input_hw[0]
        confidence = cv2.resize(foreground_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        original = semantic_map.copy()
        positive = instance_logit > 0

        cells = []  # (箱, クラス, 物体マスク, 前景確信度の平均)
        for k, (cx, cy) in enumerate(centers):
            x0, x1 = max(int(round((cx - half[0]) * scale_x)), 0), min(int(round((cx + half[0]) * scale_x)), width)
            y0, y1 = max(int(round((cy - half[1]) * scale_y)), 0), min(int(round((cy + half[1]) * scale_y)), height)
            if x1 <= x0 or y1 <= y0:
                continue
            class_id = original[min(int(cy * scale_y), height - 1), min(int(cx * scale_x), width - 1)]
            box = (slice(y0, y1), slice(x0, x1))
            mask = positive[box] & (owner[box] == k)
            if class_id == 0 or not mask.any():
                continue
            cells.append((box, class_id, mask, float(confidence[box][mask].mean())))

        # 物体の無い穴は、粗い前景確信度が低い。ただし確信度が当てにならない格子（半数以上のセルが低い）では、物体まで消すので使わない
        reliable = bool(cells) and np.median([c[3] for c in cells]) > self.foreground_threshold
        for box, class_id, mask, mean_confidence in cells:
            semantic_map[box][original[box] == class_id] = 0
            if not (reliable and mean_confidence <= self.foreground_threshold):
                semantic_map[box][mask & (semantic_map[box] == 0)] = class_id
