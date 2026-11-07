"""精緻化の後の、最終の名前付け。最終マップ（EdgeTAM が切り出した画素単位の物体）の領域ごとに、名前（前景プロンプト）をもう一度決め直す。

粗い段階のブロック（coarse/relabel.py）は物体の縁で背景と混ざり、小さな物体は 1 ブロックより小さいので判定し直せない。
最終マップの物体は EdgeTAM が物体と認めた画素なので、その画素だけから作った代表トークンは背景を含まず、物体ごとに名前を付けられる（種類の違う物体を分ける）。
前景のマスクは変えない。

  1. 領域       最終マップの前景の 8 連結成分（物体）× クラス。各パッチを、面積の半分以上を占める領域に割り当てる（final_regions）
  2. 根拠       領域のパッチだけの代表トークンと前景プロンプトの照合（RegionNamer.region_logits）
  3. 判定し直し 照合が最も強い名前にする（前景の 1/k_clusters 未満の小さな領域は、今の名前のまま）
  4. 統合       画像全体で、根拠の弱い小さいクラスを大きいクラスへまとめる（coarse/relabel.py の merge_weak_classes と同じ）
"""

from functools import partial
from typing import Callable, List, NamedTuple, Tuple

import cv2
import numpy as np

from core.common.backend import CLIPBackend
from core.common.coarse.relabel import merge_weak_classes
from core.common.params import GAP_FILL_PIXELS_PER_PATCH
from core.common.prompts import PromptBank


class RegionNamer:
    """最終マップの領域ごとに、PE の代表トークンで名前を決め直す。"""

    def __init__(self, extractor: CLIPBackend, prompts: PromptBank, grid: Tuple[int, int], min_region_divisor: int):
        self.extractor = extractor
        self.prompts = prompts
        self.grid = grid                            # パッチ格子 (高さ, 幅)
        self.min_region_divisor = min_region_divisor  # 判定し直さない小さな領域の基準（前景の 1/k_clusters。粗い段階の rejudge_regions と同じ）

    def name(self, semantic_map: np.ndarray, prompt_indices: List[int], patch_features: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """最終マップの（物体, クラス）の領域ごとに名前を決め直し、(新しい最終マップ, クラス 1~ のプロンプト番号) を返す。前景のマスクは変えない。

        patch_features: PE のパッチ特徴（領域の代表トークンの材料）。
        """
        return name_final_regions(
            semantic_map, prompt_indices, partial(self.region_logits, patch_features), self.grid, self.min_region_divisor
        )

    def region_logits(self, patch_features: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
        """名前付けの根拠。領域番号 (N,) -> 番号 0.. ごとの前景プロンプトのロジット (最大番号+1, 前景プロンプト数)。

        CLS を含まない代表トークン（extract_region_tokens。領域のパッチだけ）と、CLS 込みの代表トークン（extract_cluster_tokens。粗い判定のクラスタと同じ）の
        照合の和。CLS なしは領域ごとの違いをよく表し、CLS 込みは画像全体を共通の基準にして名前のばらつきを抑える（正解の物体の領域で、種類の分離 V が
        CLS なし 0.561・CLS 込み 0.527・和 0.580）。
        """
        tokens = self.extractor.extract_region_tokens(patch_features, region_labels)
        logits = self.prompts.foreground_logits(tokens)
        cls_tokens = np.asarray(self.extractor.extract_cluster_tokens(patch_features, region_labels))
        rows = min(len(logits), len(cls_tokens))
        logits[:rows] += self.prompts.foreground_logits(cls_tokens[:rows])
        return logits


# =============================================================================
# 最終マップの領域
# =============================================================================

class FinalRegions(NamedTuple):
    """最終マップの（物体, クラス）の領域。物体は前景の 8 連結成分（画素単位）。"""

    objects: np.ndarray       # (H, W) int32。画素ごとの物体番号（0 = 背景）
    keys: np.ndarray          # (R+1,) 領域の鍵 = 物体番号 * stride + クラス番号（最終マップの 1~）。0 番は背景
    stride: int               # 鍵のクラス番号の桁（= 最大のクラス番号 + 1）
    sample_area: np.ndarray   # (R+1,) 領域の標本の数（面積に比例する）
    patch_labels: np.ndarray  # (N,) パッチごとの領域番号。パッチの面積の半分以上を占める領域（無ければ 0）。
                              # 半分以上を占めるパッチが無い小さな領域には、その領域が最も多くを占める、まだどの領域にも属さないパッチを割り当てる


def final_regions(semantic_map: np.ndarray, grid: Tuple[int, int]) -> FinalRegions:
    """最終マップ (H, W)（0 = 背景、1~ = クラス）を、（物体, クラス）の領域に分け、パッチ格子 grid = (高さ, 幅) の各パッチを領域に割り当てる。

    領域がパッチに占める割合は、1 パッチを GAP_FILL_PIXELS_PER_PATCH 四方の点で標本にして数える（全画素で数えると 1024 四方で約 8ms）。
    標本に 1 点も入らない（パッチの 1/64 未満の）領域は、ここでは領域にならない（name_final_regions が同じクラスの最大の領域に合わせる）。
    """
    n_objects, objects = cv2.connectedComponents((semantic_map > 0).astype(np.uint8), connectivity=8)
    stride = int(semantic_map.max()) + 1
    grid_h, grid_w = grid
    height, width = semantic_map.shape
    step = GAP_FILL_PIXELS_PER_PATCH
    rows = np.minimum(((np.arange(grid_h * step) + 0.5) * height / (grid_h * step)).astype(np.int64), height - 1)
    cols = np.minimum(((np.arange(grid_w * step) + 0.5) * width / (grid_w * step)).astype(np.int64), width - 1)
    key = objects[np.ix_(rows, cols)].astype(np.int64) * stride + semantic_map[np.ix_(rows, cols)]
    sample_area = np.bincount(key.ravel(), minlength=n_objects * stride)
    keys = np.flatnonzero(sample_area)
    if keys[0] != 0:  # 背景の標本が無い画像でも、0 番を背景にする
        keys = np.concatenate([[0], keys])
    lookup = np.zeros(n_objects * stride, dtype=np.int64)
    lookup[keys] = np.arange(len(keys))
    region = lookup[key]

    # 各パッチを、面積の半分以上を占める領域に割り当てる
    num_patches = grid_h * grid_w
    patch = (np.arange(grid_h * step) // step)[:, None] * grid_w + (np.arange(grid_w * step) // step)[None, :]
    counts = np.bincount((region * num_patches + patch).ravel(), minlength=len(keys) * num_patches)
    fraction = counts.reshape(len(keys), num_patches) / float(step * step)
    owner = np.argmax(fraction[1:], axis=0) + 1
    labels = np.where(fraction[1:].max(axis=0) >= 0.5, owner, 0)
    for r in np.setdiff1d(np.arange(1, len(keys)), labels):
        p = int(np.argmax(fraction[r]))
        if labels[p] == 0 or fraction[labels[p], p] < fraction[r, p]:  # 半分以上を占める領域のパッチは取らない（割合が 0.5 未満なので）
            labels[p] = r
    return FinalRegions(objects, keys, stride, sample_area[keys], labels.astype(np.int64))


# =============================================================================
# 名前の決め直し
# =============================================================================

def name_final_regions(
    semantic_map: np.ndarray,
    prompt_indices: List[int],
    region_logits: Callable[[np.ndarray], np.ndarray],
    grid: Tuple[int, int],
    min_region_divisor: int,
) -> Tuple[np.ndarray, List[int]]:
    """最終マップの（物体, クラス）の領域ごとに名前（前景プロンプト）を決め直し、(新しい最終マップ, クラス 1~ のプロンプト番号) を返す。前景のマスクは変えない。

    region_logits: パッチごとの領域番号 (N,) -> 番号 0.. ごとの前景プロンプトのロジット (最大番号+1, 前景プロンプト数)。
    1. 判定し直し: 領域のパッチの代表トークンの照合が最も強いプロンプトにする。前景のパッチ数 / min_region_divisor（粗い段階の rejudge_regions と同じ
       k_clusters）より小さい領域は、独立した根拠を持てない（1〜2 パッチのトークンの照合はノイズが大きい）ので、今の名前のまま。
    2. 統合: 画像全体で、根拠が弱い小さいクラスを大きいクラスへまとめる（merge_weak_classes。対数オッズ + log 面積比）。
    名前ごとに 1 クラスにする。クラスの順は、今のクラスの順（残った名前）→ 新しい名前（出てきた順）。何も変わらなければ、入力をそのまま返す。
    """
    if not len(prompt_indices) or semantic_map.max() == 0:
        return semantic_map, prompt_indices
    regions = final_regions(semantic_map, grid)
    labels, stride = regions.patch_labels, regions.stride
    region_class = regions.keys % stride
    num_regions = len(regions.keys)
    foreground = labels > 0
    if not foreground.any():
        return semantic_map, prompt_indices
    logits = region_logits(labels)
    if len(logits) < num_regions:  # パッチを持たない（最大番号より大きい）領域の行
        logits = np.vstack([logits, np.zeros((num_regions - len(logits), logits.shape[1]), logits.dtype)])

    # 1. 判定し直し（小さな領域は今の名前のまま）
    prompts = np.concatenate([[-1], np.asarray(prompt_indices, dtype=np.int64)])  # クラス番号 -> プロンプト番号
    current = prompts[region_class]
    area = np.bincount(labels, minlength=num_regions)
    rejudge = area >= foreground.sum() / min_region_divisor
    rejudge[0] = False
    name = np.where(rejudge, np.argmax(logits, axis=1), current)

    # 2. 統合（パッチ単位の配列にして渡す。グループ = 画像全体、ブロック = 領域）
    merged = merge_weak_classes(np.where(foreground, name[labels], 0), foreground.astype(np.int64), labels, logits)
    moved = {int(name[r]): int(merged[np.argmax(labels == r)]) for r in np.unique(labels[foreground])}
    name = np.array([moved.get(int(n), int(n)) for n in name], dtype=np.int64)

    # 標本に入らなかった小さな領域は、同じクラスの最大の領域の名前にする（クラスに標本が無ければ、今の名前のまま）
    class_name = prompts.copy()
    largest = np.zeros(stride, dtype=np.int64)
    for r in range(1, num_regions):
        c = region_class[r]
        if regions.sample_area[r] > largest[c]:
            largest[c], class_name[c] = regions.sample_area[r], name[r]
    histogram = cv2.calcHist([semantic_map.astype(np.uint8)], [0], None, [stride], [0, stride]).ravel()  # 全画素の bincount より速い
    present_classes = np.flatnonzero(histogram[1:]) + 1
    names = [int(n) for n in name[1:]] + [int(class_name[c]) for c in present_classes]
    order = [p for p in dict.fromkeys(prompt_indices) if p in names] + [p for p in dict.fromkeys(names) if p not in prompt_indices]
    number = {p: k + 1 for k, p in enumerate(order)}
    class_lookup = np.zeros(stride, dtype=semantic_map.dtype)
    class_lookup[present_classes] = [number[int(class_name[c])] for c in present_classes]
    region_number = np.array([0] + [number[int(n)] for n in name[1:]])
    if order == list(prompt_indices) and np.array_equal(class_lookup[present_classes], present_classes):
        if np.array_equal(region_number[1:], class_lookup[region_class[1:]]):
            return semantic_map, prompt_indices  # 何も変わらない
    if np.array_equal(region_number[1:], class_lookup[region_class[1:]]):
        return class_lookup[semantic_map], order  # 同じクラスの物体はすべて同じ名前: クラスの表で書き換える
    key_lookup = class_lookup[np.arange(int(regions.objects.max() + 1) * stride) % stride]
    key_lookup[regions.keys[1:]] = region_number[1:]
    return key_lookup[regions.objects * stride + semantic_map], order  # 同じクラスの物体に違う名前: 物体ごとに書き換える
