"""向き・大きさが混在して接した長方形の物体（積んだ荷物・並べた部品）を、1 個ずつのインスタンスに分ける（学習なし）。

本体の最終結果の前景（semantic_map > 0）に対して、次の順に行う（定数は params.py の RECT_*）:
  1. 大きさ (L, S) を求める: 隣り合う平行な縁の間隔のヒストグラム（rect_size.py）。L の候補が 2 個分に見えるときだけ、梯子の各長さの敷き詰めの質（rect_tiling.py）で直す
  2. その大きさの長方形を、前景に敷き詰める（rect_tiling.py）
  3. 長方形ごとに EdgeTAM をデコードする: 長方形のマスク（縮めたもの）+ 軸に平行な箱 + 中心の点。物体の向きを戻さず、再エンコードもしない
     （EdgeTAM は点では物体ごとに分けられず、箱なら分けられる。箱は軸に平行なので、傾いた物体には長方形のマスクを初期マスクに与えて補う）。
     特徴は長方形の周りの窓で切り出し、窓の大きさごとに一括でデコードする
  4. 予測 IoU の低いマスクを捨て、重複を除いて（包含 NMS）、1 枚のインスタンスマップにする
EdgeTAM の入力は image_size 四方（RECT_IMAGE_SIZE）なので、インスタンスマップも入力の座標で作り、最後に元画像の大きさへ拡大する。
"""

from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.utils import Timer
from core.common.edgetam_session import EdgeTamSession, Features
from core.common.params import (
    EDGETAM_FEATURE_STRIDES,
    EDGETAM_LOGIT_CLIP,
    PRIOR_LOGIT_RANGE,
    RECT_DOUBLING,
    RECT_IMAGE_SIZE,
    RECT_LADDER_BASE,
    RECT_LADDER_RATIO,
    RECT_LADDER_STEPS,
    RECT_MASK_CANDIDATE,
    RECT_MASK_SHRINK,
    RECT_MIN_AREA,
    RECT_MIN_PREDICTED_IOU,
    RECT_NMS_CONTAIN,
    RECT_NMS_IOU,
    RECT_SCAN_ANGLE_STEP,
    RECT_SCAN_HALF,
    RECT_SCAN_HIGH_RATIO,
    RECT_SCAN_LOW_RATIO,
    RECT_SCAN_MIN_ASPECT,
    RECT_SCAN_TOP,
    RECT_TILE_ANGLE_STEP,
    RECT_TILE_HALF,
    RECT_TILE_TOP,
    RECT_WINDOW_MARGIN_TOKENS,
    RECT_WINDOW_MAX_TOKENS,
    RECT_WINDOW_MIN_TOKENS,
    RECT_WINDOW_TOKEN_STEP,
)
from core.common.rect_size import estimate_gaps
from core.common.rect_tiling import EdgeMaps, RectCandidate, rect_corners, tile, tiling_quality
from core.common.result import InstanceResult


class LocalMask(NamedTuple):
    """入力の座標のマスクを、外接矩形（左上 (x, y)）だけ切り出して持つ（全画面の比較を避けるため）。"""

    mask: np.ndarray    # bool (h, w)
    x: int
    y: int

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


class RectSeparator:
    """前景の長方形の物体を、1 個ずつのインスタンスに分ける。"""

    def __init__(self, session: EdgeTamSession):
        if session.target_size != RECT_IMAGE_SIZE:
            raise ValueError(f"dense_instances は image_size={RECT_IMAGE_SIZE} でだけ検証している（今は {session.target_size}）")
        self.session = session

    # =========================================================================
    # メインの流れ
    # =========================================================================

    def separate(self, image: np.ndarray, semantic_map: np.ndarray, feats: Features) -> Optional[InstanceResult]:
        """image: 入力画像（image_size 四方, RGB, uint8）。semantic_map: 本体の最終結果（元画像の大きさ。0 = 背景）。feats: EdgeTAM エンコーダの出力。

        前景が無い・大きさが求まらないときは None。大きさが求まって 1 個も見つからなければ、インスタンスが空の結果。
        """
        foreground = cv2.resize(semantic_map, (RECT_IMAGE_SIZE, RECT_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) > 0
        if not foreground.any():
            return None
        with Timer("    ┌ 長方形の大きさの推定"):
            size = self._estimate_size(image, foreground)
        if size is None:
            return None
        length, width = size
        with Timer("    ┌ 長方形の敷き詰め"):
            chosen = tile(EdgeMaps(image, foreground, RECT_TILE_HALF), length, width, RECT_TILE_ANGLE_STEP, RECT_TILE_TOP)
        with Timer("    ┌ 長方形ごとのデコード"):
            masks, scores = self._decode(feats, chosen, length, width)
            masks, scores = self._select(masks, scores)
        return InstanceResult(self._instance_map(masks, semantic_map.shape), scores, size)

    # =========================================================================
    # 1. 大きさ
    # =========================================================================

    @staticmethod
    def _estimate_size(image: np.ndarray, foreground: np.ndarray) -> Optional[Tuple[float, float]]:
        """(L, S)。S は縁の間隔のヒストグラムの基本周期。L は、ヒストグラムの候補が敷き詰めの質で選んだ長さの RECT_DOUBLING 倍以上（2 個分に見える）なら敷き詰めの長さ、でなければ候補。"""
        gaps = estimate_gaps(image, foreground)
        if gaps is None:
            return None
        candidate, width = gaps
        ladder = [RECT_LADDER_BASE * RECT_LADDER_RATIO ** k for k in range(RECT_LADDER_STEPS)]
        lower, upper = max(RECT_SCAN_MIN_ASPECT * width, RECT_SCAN_LOW_RATIO * candidate), RECT_SCAN_HIGH_RATIO * candidate
        lengths = sorted({l for l in ladder if lower <= l <= upper} | {candidate})
        maps = EdgeMaps(image, foreground, RECT_SCAN_HALF)
        area = int(foreground.sum())
        best_length, best_quality = None, -1.0
        for length in lengths:
            chosen = tile(maps, length, width, RECT_SCAN_ANGLE_STEP, RECT_SCAN_TOP)
            if len(chosen) < 2:     # 1 個では質を比べられない
                continue
            quality = tiling_quality(chosen, length, width, area)
            if quality > best_quality:
                best_length, best_quality = length, quality
        if best_length is not None and candidate / best_length >= RECT_DOUBLING:
            return best_length, width
        return candidate, width

    # =========================================================================
    # 3. デコード
    # =========================================================================

    def _decode(self, feats: Features, chosen: Sequence[RectCandidate], length: float, width: float) -> Tuple[List[LocalMask], np.ndarray]:
        """長方形ごとに、長方形のマスク + 箱 + 中心の点でデコードする。返り値: (長方形ごとのマスク, 予測 IoU (N,))。"""
        count = len(chosen)
        if count == 0:
            return [], np.zeros(0)
        feats0, feats1, embed = feats
        stride0, stride1, embed_stride = EDGETAM_FEATURE_STRIDES
        grid_h, grid_w = embed.shape[2:]
        ratio1, ratio0 = embed_stride // stride1, embed_stride // stride0
        polygons = [rect_corners(c.cx, c.cy, length * RECT_MASK_SHRINK, width * RECT_MASK_SHRINK, c.theta) for c in chosen]
        boxes = [np.array([p.min(0), p.max(0)]) for p in polygons]
        windows = [self._window_tokens(box, grid_h, grid_w) for box in boxes]
        masks: List[Optional[LocalMask]] = [None] * count
        scores = np.zeros(count)
        for win in sorted(set(windows)):
            members = [i for i in range(count) if windows[i] == win]
            side = win * ratio0
            windows0, windows1, windows_e, coords, origins, priors = [], [], [], [], [], []
            for i in members:
                c = chosen[i]
                tx = int(np.clip(round(c.cx / embed_stride - win / 2), 0, grid_w - win))
                ty = int(np.clip(round(c.cy / embed_stride - win / 2), 0, grid_h - win))
                ox, oy = tx * embed_stride, ty * embed_stride
                windows_e.append(embed[0, :, ty:ty + win, tx:tx + win])
                windows1.append(feats1[0, :, ty * ratio1:(ty + win) * ratio1, tx * ratio1:(tx + win) * ratio1])
                windows0.append(feats0[0, :, ty * ratio0:(ty + win) * ratio0, tx * ratio0:(tx + win) * ratio0])
                coords.append(np.vstack([boxes[i], [c.cx, c.cy]]) - [ox, oy])
                prior = np.full((side, side), -PRIOR_LOGIT_RANGE, np.float32)    # 長方形の内側 +、外側 -（窓の低解像度の座標）
                cv2.fillPoly(prior, [np.round((polygons[i] - [ox, oy]) / stride0).astype(np.int32)], PRIOR_LOGIT_RANGE)
                priors.append(prior[None])
                origins.append((ox, oy))
            n = len(members)
            low, pred = self.session.decode_batch(
                np.stack(windows0), np.stack(windows1), np.stack(windows_e),
                np.array(coords, np.float32), np.tile(np.array([[2, 3, 1]], np.int32), (n, 1)),   # ラベル 2・3 = 箱の左上・右下、1 = 正例の点
                np.stack(priors), np.ones((n,), np.float32),
            )
            low = np.clip(low[:, RECT_MASK_CANDIDATE], -EDGETAM_LOGIT_CLIP, EDGETAM_LOGIT_CLIP)
            for k, i in enumerate(members):
                masks[i] = self._paste(low[k], origins[k], stride0)
                scores[i] = pred[k, RECT_MASK_CANDIDATE]
        return masks, scores

    @staticmethod
    def _window_tokens(box: np.ndarray, grid_h: int, grid_w: int) -> int:
        """箱を囲む窓の大きさ（image_embed のトークン数）。"""
        need = int(np.ceil(max(box[1] - box[0]) / EDGETAM_FEATURE_STRIDES[2])) + RECT_WINDOW_MARGIN_TOKENS
        win = int(np.clip(RECT_WINDOW_TOKEN_STEP * np.ceil(need / RECT_WINDOW_TOKEN_STEP), RECT_WINDOW_MIN_TOKENS, RECT_WINDOW_MAX_TOKENS))
        return min(win, grid_h, grid_w)

    @staticmethod
    def _paste(low: np.ndarray, origin: Tuple[int, int], stride: int) -> LocalMask:
        """窓の低解像度ロジット (s, s) を拡大して正の画素を取り、窓の左上 origin（入力の画素）に置いた、入力の座標のマスク（外接矩形だけ）にする。"""
        up = cv2.resize(low.astype(np.float32), (low.shape[1] * stride, low.shape[0] * stride), interpolation=cv2.INTER_LINEAR)
        ox, oy = origin
        inside = up[:RECT_IMAGE_SIZE - oy, :RECT_IMAGE_SIZE - ox] > 0
        rows, cols = np.flatnonzero(inside.any(1)), np.flatnonzero(inside.any(0))
        if len(rows) == 0:
            return LocalMask(np.zeros((0, 0), bool), ox, oy)
        return LocalMask(inside[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], ox + int(cols[0]), oy + int(rows[0]))

    # =========================================================================
    # 4. 選別とインスタンスマップ
    # =========================================================================

    @staticmethod
    def _overlaps(masks: Sequence[LocalMask]) -> np.ndarray:
        """マスクの対ごとの重なりの画素数 (N, N)（float32。対角は面積）。外接矩形が重なる対だけ数える。"""
        count = len(masks)
        inter = np.zeros((count, count), np.float32)
        for i in range(count):
            a = masks[i]
            inter[i, i] = a.area
            for j in range(i + 1, count):
                b = masks[j]
                x0, y0 = max(a.x, b.x), max(a.y, b.y)
                x1, y1 = min(a.x + a.mask.shape[1], b.x + b.mask.shape[1]), min(a.y + a.mask.shape[0], b.y + b.mask.shape[0])
                if x1 > x0 and y1 > y0:
                    inter[i, j] = inter[j, i] = np.count_nonzero(a.mask[y0 - a.y:y1 - a.y, x0 - a.x:x1 - a.x] & b.mask[y0 - b.y:y1 - b.y, x0 - b.x:x1 - b.x])
        return inter

    @classmethod
    def _select(cls, masks: List[LocalMask], scores: np.ndarray) -> Tuple[List[LocalMask], np.ndarray]:
        """面積と予測 IoU が足りるマスクを、予測 IoU の高い順に、重複（IoU が RECT_NMS_IOU 以上、または小さい方の RECT_NMS_CONTAIN 以上が重なる）を除いて採る。"""
        keep = [i for i in range(len(masks)) if masks[i].area >= RECT_MIN_AREA and scores[i] >= RECT_MIN_PREDICTED_IOU]
        if not keep:
            return [], np.zeros(0)
        inter = cls._overlaps([masks[i] for i in keep])
        area = np.diag(inter)
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)
        contain = inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1)
        picked: List[int] = []
        for i in np.argsort(-scores[keep]):
            if all(iou[i, j] < RECT_NMS_IOU and contain[i, j] < RECT_NMS_CONTAIN for j in picked):
                picked.append(int(i))
        return [masks[keep[i]] for i in picked], scores[keep][picked]

    @staticmethod
    def _instance_map(masks: List[LocalMask], size: Tuple[int, int]) -> np.ndarray:
        """マスク（予測 IoU の高い順）を 1 枚のラベルマップ（1 から）にして、元画像の大きさ size=(高さ, 幅) へ拡大する。

        重なる画素は面積の小さいマスクのものにする（大きいマスクが隣の物体まで飲み込んでも、小さいマスクの物体を残す）。
        1 画素に 1 個しか割り当てられないので、重なるマスクの一覧のままより F1 は少し下がる（合成 + 実画像 95 枚で 0.775 → 0.772）。予測 IoU の高い方にしても 0.772 で、
        実画像 5 枚は 0.654 → 0.628（面積の小さい方なら 0.654）、開発 + 評価の 60 枚は逆に面積の小さい方が 約 -0.004。どちらも一覧より下がり、差は小さい。
        """
        labels = np.zeros((RECT_IMAGE_SIZE, RECT_IMAGE_SIZE), np.uint16)
        for index in np.argsort([-m.area for m in masks], kind="stable"):
            m = masks[index]
            labels[m.y:m.y + m.mask.shape[0], m.x:m.x + m.mask.shape[1]][m.mask] = index + 1
        return cv2.resize(labels, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
