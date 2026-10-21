import time
from pathlib import Path
from typing import Union

import numpy as np

from core.utils import Timer, load_image, normalize_image, resize_image
from core.common.coarse_segmentation import CoarseSegmenter
from core.common.backend import CLIPBackend, EdgeTamBackend
from core.common.edgetam_helper import EdgeTamHelper
from core.common.result import SegmentationResult
from core.common.semantic_refiner import SemanticRefiner
from core.common.visualization import ResultSaver, semantic_class_names


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
        self.debug = config["debug"]
        # 粗い判定で、PE に加えて EdgeTAM の特徴でもクラスタ分けして前景確信度を AND する（誤検出を減らす。小物体の検出率が少し下がる）
        self.use_edgetam_clustering = config.get("use_edgetam_clustering", False)

        self.coarse = CoarseSegmenter(extractor, config)
        self.refiner = SemanticRefiner(edgetam, extractor.feature_size, config)
        self.saver = ResultSaver(config, extractor.feature_size)

        # tools から参照されるプロンプト一覧
        self.foreground_prompts = self.coarse.foreground_prompts
        self.background_prompts = self.coarse.background_prompts

    def segment(self, image_raw: np.ndarray, debug_dir: Union[Path, None] = None) -> SegmentationResult:
        """1枚の画像（RGB）をセグメンテーションする。ファイルへの保存はしない。

        debug_dir: デバッグ中間画像の保存先。config の debug が true で、かつ指定されたときだけ保存する。
        """
        height, width = image_raw.shape[:2]
        self.refiner.debug_dir = debug_dir if self.debug else None

        # 計測開始
        start = time.time()

        # 画像前処理
        with Timer("画像前処理時間"):
            # リサイズはCLIPとEdgeTAMで共通（正規化だけがモデルごとに異なる）
            resized = resize_image(image_raw, self.extractor.target_size)
            image_tensor = normalize_image(resized, self.extractor.mean, self.extractor.std)

        # EdgeTAMの特徴量抽出（粗い判定で EdgeTAM の特徴を使うときは、ここで 1 回だけ実行して精緻化にも渡す）
        edgetam_feats = None
        if self.use_edgetam_clustering:
            with Timer("EdgeTAM特徴量抽出処理時間"):
                edgetam_feats = self.refiner.predict_encoder(resized)

        # 特徴量抽出
        with Timer("特徴量抽出処理時間"):
            _, patch_features, fused_patch_features = self.coarse.extract_features(image_tensor)

        # CLIP + KMeans による初期前景マップの推定
        with Timer("粗い前景マスク取得処理時間"):
            foreground_map, foreground_mask, cluster_labels, cluster_index_list = (
                self.coarse.generate_foreground_map(
                    patch_features, fused_patch_features, *((edgetam_feats[2],) if edgetam_feats is not None else ())
                )
            )

        # セマンティックセグメンテーションのラベルを取得
        with Timer("セマンティックセグメンテーション処理時間"):
            labels_list, semantic_labels = self.coarse.assign_semantic_labels(cluster_labels, cluster_index_list)

        # パッチ単位の粗い結果を整形（穴埋め・小さな島の統合・境界パッチのラベル統一）
        with Timer("粗いセグメンテーション整形処理時間"):
            raw_semantic_labels = semantic_labels
            foreground_mask, semantic_labels = self.coarse.refine_coarse_segmentation(semantic_labels, foreground_mask)

        # 粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合
        with Timer("詳細セマンティックセグメンテーション処理時間"):
            semantic_map, prompt_indices, point_coords, point_labels = self.refiner.refine(
                resized, foreground_map, foreground_mask, semantic_labels, labels_list, width, height, edgetam_feats
            )

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        return SegmentationResult(
            semantic_map=semantic_map,
            prompt_indices=prompt_indices,
            point_coords=point_coords,
            point_labels=point_labels,
            foreground_map=foreground_map,
            foreground_mask=foreground_mask,
            cluster_labels=cluster_labels,
            raw_semantic_labels=raw_semantic_labels,
            semantic_labels=semantic_labels,
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
        result = self.segment(image_raw, debug_dir=save_dir / "debug")

        all_prompts = self.foreground_prompts + self.background_prompts
        self.saver.save(save_dir, image_raw, result, semantic_class_names(all_prompts, result.prompt_indices))
