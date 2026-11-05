from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np


class ClusterView(NamedTuple):
    """KMeans のクラスタ分けと、クラスタごとの分類結果（パッチ格子上）。"""

    labels: np.ndarray       # パッチごとのクラスタ番号（平坦）
    index_list: List[int]    # クラスタ番号ごとの対応プロンプト番号（前景プロンプト -> 背景プロンプトの通し番号）


@dataclass
class InstanceResult:
    """密集した長方形の物体を 1 個ずつに分けた結果（config の dense_instances が真のとき。rect_separator.py）。"""

    instance_map: np.ndarray           # (height, width) uint16。0=なし、1~=インスタンス（EdgeTAM の予測 IoU の高い順。重なる画素は面積の小さい方）
    scores: np.ndarray                 # (インスタンス数,) 各インスタンスの EdgeTAM の予測 IoU
    size: Tuple[float, float]          # 敷き詰めた長方形の大きさ（長辺, 短辺）。image_size 四方の入力の画素


@dataclass
class SegmentationResult:
    """1枚の画像に対するセグメンテーション結果（最終結果と、可視化用の中間結果）。"""

    # 最終結果
    semantic_map: np.ndarray           # (height, width) uint8。0=背景、1~=前景クラス
    prompt_indices: List[int]          # クラス1~に対応する、前景プロンプトのインデックス
    point_coords: List[List[float]]    # EdgeTAMに与えたプロンプト点（画像座標、全クラス分）
    point_labels: List[int]            # 上の点のラベル（1=正例、0=負例）

    # 粗い推定の中間結果（パッチ格子（高さ x 幅）上）
    foreground_map: np.ndarray         # 前景らしさ
    foreground_mask: np.ndarray        # 粗い前景マスク
    cluster_labels: np.ndarray         # KMeansのクラスタ番号（平坦）。PE のクラスタ × 伝播後のプロンプトの組
    semantic_labels: np.ndarray        # 粗いセマンティックラベル
    cluster_index_list: List[int]      # cluster_labels のクラスタごとの対応プロンプト番号
    pe_clusters: ClusterView           # 伝播前の PE だけのクラスタ分けと分類結果

    # インスタンス分離（instance_split: true のときだけ）。semantic_map の同じクラスの領域を、物体 1 個ずつに分けたもの
    instance_map: Optional[np.ndarray] = None       # (height, width) int32。0=背景、1~=インスタンス
    instance_classes: Optional[List[int]] = None    # インスタンス k（1~）のクラス（semantic_map の値）。instance_classes[k - 1]

    # 精緻化の中間結果（名前 -> 0〜1 の2次元配列。保存時に元画像サイズへ拡大する）
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)

    # 密集した長方形の物体のインスタンス分離（dense_instances が偽、前景が無い、大きさが求まらないときは None）
    instances: Optional[InstanceResult] = None
