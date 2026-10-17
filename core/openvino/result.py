from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SegmentationResult:
    """1枚の画像に対するセグメンテーション結果（最終結果と、可視化用の中間結果）。"""

    # 最終結果
    semantic_map: np.ndarray           # (height, width) uint8。0=背景、1~=前景クラス
    prompt_indices: List[int]          # クラス1~に対応する、前景プロンプトのインデックス
    point_coords: List[List[float]]    # EdgeTAMに与えたプロンプト点（画像座標、全クラス分）
    point_labels: List[int]            # 上の点のラベル（1=正例、0=負例）

    # 粗い推定の中間結果（パッチ格子 feature_size x feature_size 上）
    foreground_map: np.ndarray         # 前景らしさ
    foreground_mask: np.ndarray        # 整形後の粗い前景マスク
    cluster_labels: np.ndarray         # KMeansのクラスタ番号（平坦）
    raw_semantic_labels: np.ndarray    # 整形前の粗いセマンティックラベル
    semantic_labels: np.ndarray        # 整形後の粗いセマンティックラベル
