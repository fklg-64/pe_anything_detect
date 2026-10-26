"""CLIPによる粗い推定 -> EdgeTAMによる精緻化 の全体の流れ（画像1枚分）をつなぐ。"""

import time
from pathlib import Path
from typing import Union

import numpy as np

from core.utils import Timer, load_image, normalize_image
from core.common.coarse_segmentation import CoarseForeground, CoarseSegmenter
from core.common.backend import CLIPBackend, EdgeTamBackend
from core.common.edgetam_helper import EdgeTamHelper
from core.common.image_layout import RESIZE_MODES, make_layout
from core.common.result import SegmentationResult
from core.common.semantic_refiner import SemanticRefiner
from core.common.visualization import ResultSaver


class SegmentationRunner:
    """CLIPによる粗い推定 -> EdgeTAMによる精緻化 の一連のセグメンテーションを実行するクラス"""

    def __init__(
        self,
        extractor: CLIPBackend,
        edgetam: Union[EdgeTamBackend, EdgeTamHelper],
        config: dict,
    ):
        self.extractor = extractor
        self.edgetam = edgetam
        self.output_dir = config["output_dir"]
        # 粗い判定の前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える（誤検出を減らす。EdgeTAM のエンコードを先に 1 回だけ実行する）
        self.use_edgetam_clustering = config.get("use_edgetam_clustering", False)
        # モデルへ入力する画像サイズの決め方（square / fit / pad。image_layout.py）
        self.resize_mode = config.get("resize_mode", "square")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode は {RESIZE_MODES} のいずれかにしてください: {self.resize_mode!r}")

        self.coarse = CoarseSegmenter(extractor, config)
        self.refiner = SemanticRefiner(edgetam, extractor.feature_size, config)
        self.saver = ResultSaver(config, extractor.feature_size)

        # tools から参照されるプロンプト一覧
        self.foreground_prompts = self.coarse.foreground_prompts
        self.background_prompts = self.coarse.background_prompts

    def segment(self, image_raw: np.ndarray) -> SegmentationResult:
        """1枚の画像（RGB）をセグメンテーションする。ファイルへの保存はしない（保存は run / ResultSaver）。"""
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        # 画像前処理
        with Timer("画像前処理時間"):
            # リサイズ（パディング込み）はCLIPとEdgeTAMで共通（正規化だけがモデルごとに異なる）
            layout = make_layout((height, width), self.extractor.target_size, int(self.extractor.patch_size), self.resize_mode)
            resized = layout.apply(image_raw)
            image_tensor = normalize_image(resized, self.extractor.mean, self.extractor.std)
            # 以降のパッチ格子は、パディングを除いた（元画像の全体を覆う）領域
            self.coarse.grid = layout.valid_grid

        # EdgeTAMの特徴量抽出（粗い判定で EdgeTAM の特徴を使うときは、ここで 1 回だけ実行して精緻化にも渡す）
        edgetam_features = None
        if self.use_edgetam_clustering:
            with Timer("EdgeTAM特徴量抽出処理時間"):
                edgetam_features = layout.crop_edgetam(self.refiner.predict_encoder(resized))

        # 特徴量抽出
        with Timer("特徴量抽出処理時間"):
            _, patch_features, fused_patch_features = self.coarse.extract_features(image_tensor)
            # パディング部分のトークンは捨てる
            patch_features, fused_patch_features = layout.crop_tokens(patch_features), layout.crop_tokens(fused_patch_features)

        # CLIP + KMeans による初期前景マップの推定
        with Timer("粗い前景マスク取得処理時間"):
            if edgetam_features is None:
                generated = self.coarse.generate_foreground_map(patch_features, fused_patch_features)
            else:
                edgetam_image_embed = edgetam_features[2]  # (high_res_feats0, high_res_feats1, image_embed) の最後
                generated = self.coarse.generate_foreground_map(patch_features, fused_patch_features, edgetam_image_embed)
            # 4 要素のタプルを返す差し替え実装（tools）でも動くよう、名前付きにそろえる
            coarse_foreground = CoarseForeground(*generated)
            foreground_map, foreground_mask = coarse_foreground.foreground_map, coarse_foreground.foreground_mask
            cluster_labels, cluster_index_list = coarse_foreground.cluster_labels, coarse_foreground.cluster_index_list

        # セマンティックセグメンテーションのラベルを取得
        with Timer("セマンティックセグメンテーション処理時間"):
            labels_list, semantic_labels = self.coarse.assign_semantic_labels(cluster_labels, cluster_index_list)

        # パッチ単位の粗い結果を整形（穴埋め・小さな島の統合・境界パッチのラベル統一）
        with Timer("粗いセグメンテーション整形処理時間"):
            raw_semantic_labels = semantic_labels
            foreground_mask, semantic_labels = self.coarse.refine_coarse_segmentation(semantic_labels, foreground_mask)

        # 粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合
        with Timer("詳細セマンティックセグメンテーション処理時間"):
            refined = self.refiner.refine(
                resized, foreground_map, foreground_mask, semantic_labels, labels_list, width, height, edgetam_features, layout
            )

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        return SegmentationResult(
            semantic_map=refined.semantic_map,
            prompt_indices=refined.prompt_indices,
            point_coords=refined.point_coords,
            point_labels=refined.point_labels,
            foreground_map=foreground_map,
            foreground_mask=foreground_mask,
            cluster_labels=cluster_labels,
            raw_semantic_labels=raw_semantic_labels,
            semantic_labels=semantic_labels,
            cluster_index_list=cluster_index_list,
            pe_clusters=coarse_foreground.pe_clusters,
            debug_images=refined.debug_images,
        )

    def run(self, image_path: Union[str, Path]) -> None:
        """画像ファイルをセグメンテーションし、結果を output_dir/<画像名>/ に保存する。"""
        image_path = Path(image_path)

        if image_path.exists():
            print(f"====== 画像パス: {image_path.name} ======")
        else:
            print(f"====== 存在しません: {image_path.name} ======")
            return

        save_dir = Path(self.output_dir) / image_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        image_raw = load_image(image_path)
        result = self.segment(image_raw)

        self.saver.save(save_dir, image_raw, result, self.foreground_prompts + self.background_prompts)
