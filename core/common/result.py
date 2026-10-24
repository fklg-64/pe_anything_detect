from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional

import numpy as np


class ClusterView(NamedTuple):
    """KMeans のクラスタ分けと、クラスタごとの分類結果（パッチ格子上）。"""

    labels: np.ndarray       # パッチごとのクラスタ番号（平坦）
    index_list: List[int]    # クラスタ番号ごとの対応プロンプト番号（前景プロンプト -> 背景プロンプトの通し番号）


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
    cluster_labels: np.ndarray         # KMeansのクラスタ番号（平坦）。EdgeTAM 併用時は PE のクラスタ × 伝播後のプロンプトの組
    raw_semantic_labels: np.ndarray    # 整形前の粗いセマンティックラベル
    semantic_labels: np.ndarray        # 整形後の粗いセマンティックラベル
    cluster_index_list: List[int]      # cluster_labels のクラスタごとの対応プロンプト番号
    # EdgeTAM 併用時（use_edgetam_clustering）だけ、伝播前の PE だけのクラスタ分けと分類結果。併用しなければ None
    pe_clusters: Optional[ClusterView] = None

    # 精緻化の中間結果（名前 -> 0〜1 の2次元配列。保存時に元画像サイズへ拡大する）
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
