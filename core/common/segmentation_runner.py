"""CLIPによる粗い推定 -> EdgeTAMによる精緻化 の全体の流れ（画像1枚分）をつなぐ。

処理の流れ（SegmentationRunner.segment。各段階の詳細は docs/overview.md）:
  1. 前処理               リサイズ（ImageLayout）と正規化
  2. EdgeTAM エンコーダ   use_edgetam_clustering のとき、粗い判定の前に 1 回だけ実行（精緻化でも使う）
  3. CLIP（PE）の特徴量   パッチ特徴・fused パッチ特徴
  4. 粗い前景/クラスタ    CoarseSegmenter.generate_foreground_map（KMeans + テキスト照合 + EdgeTAM 特徴での伝播 + クラスのまとめ）
  5. 粗いセマンティック   assign_semantic_labels -> refine_coarse_segmentation（パッチ単位）
  6. 精緻化               SemanticRefiner.refine（クラスごとに EdgeTAM デコーダ -> 1 枚のセマンティックマップ）
結果の保存は run() / ResultSaver（segment には含まれない）。
"""

import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from core.utils import Timer, load_image, normalize_image
from core.common.coarse_segmentation import CoarseForeground, CoarseSegmenter
from core.common.backend import CLIPBackend, EdgeTamBackend
from core.common.edgetam_helper import EdgeTamHelper
from core.common.image_layout import ImageLayout, make_layout
from core.common.params import RESIZE_MODES
from core.common.result import SegmentationResult
from core.common.semantic_refiner import SemanticRefiner
from core.common.gallery import ORIGINAL_STEM, write_gallery_html
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
        # 保存先フォルダ名 -> 画像パス（同名の別画像が上書きされないようにする）
        self._save_names: dict[str, Path] = {}
        # 粗い判定の前景確信度を、EdgeTAM の特徴が似たパッチどうしで伝播して揃える（誤検出を減らす。EdgeTAM のエンコードを先に 1 回だけ実行する）
        self.use_edgetam_clustering = config.get("use_edgetam_clustering", False)
        # モデルへ入力する画像サイズの決め方（square / fit / pad。image_layout.py）
        self.resize_mode = config.get("resize_mode", "square")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode は {RESIZE_MODES} のいずれかにしてください: {self.resize_mode!r}")

        # モデルの推論（OpenVINO・PyTorch は GIL を離す）と、NumPy / OpenCV の整形を重ねて実行する Python のスレッド（0 なら重ねず、従来どおり順に実行する）。
        # 結果は従来と同じ順に反映するので出力は変わらない。モデルの推論のスレッド数（ov_num_thread）は変わらない
        worker_threads = int(config.get("worker_threads", 0))
        self.pool = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="segment") if worker_threads > 0 else None

        self.coarse = CoarseSegmenter(extractor, config)
        self.refiner = SemanticRefiner(edgetam, extractor.feature_size, config, pool=self.pool)
        self.saver = ResultSaver(config, extractor.feature_size)
        self.save_gallery_html = config.get("save_gallery_html", False)
        self._gallery_times: dict[Path, float] = {}  # 一覧HTML用の 保存先 -> 処理時間(ms)

        # tools から参照されるプロンプト一覧
        self.foreground_prompts = self.coarse.foreground_prompts
        self.background_prompts = self.coarse.background_prompts

    def segment(self, image_raw: np.ndarray) -> SegmentationResult:
        """1枚の画像（RGB）をセグメンテーションする。ファイルへの保存はしない（保存は run / ResultSaver）。"""
        # 段階ごとの処理時間の表示は、終わってからまとめて出す（端末への出力を、計測する処理の内側で待たない）
        with Timer.deferred():
            return self._segment(image_raw)

    def _segment(self, image_raw: np.ndarray) -> SegmentationResult:
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        layout, resized, image_tensor = self._preprocess(image_raw)
        edgetam_features = self._encode_edgetam(layout, resized)
        # EdgeTAM の特徴の kNN グラフ（PE の結果に依らない）は、PE の推論と重ねて求める
        edge_graph = self._start_edge_graph(edgetam_features)
        patch_features, fused_patch_features = self._extract_clip_features(layout, image_tensor)
        coarse_foreground = self._estimate_coarse_foreground(patch_features, fused_patch_features, edgetam_features, edge_graph)
        cluster_labels, cluster_index_list = coarse_foreground.cluster_labels, coarse_foreground.cluster_index_list

        # セマンティックセグメンテーションのラベルを取得
        with Timer("セマンティックセグメンテーション処理時間"):
            labels_list, semantic_labels = self.coarse.assign_semantic_labels(cluster_labels, cluster_index_list)

        # パッチ単位の粗い結果を整形（穴埋め・小さな島の統合・境界パッチのラベル統一）
        with Timer("粗いセグメンテーション整形処理時間"):
            raw_semantic_labels = semantic_labels
            foreground_mask, semantic_labels = self.coarse.refine_coarse_segmentation(
                semantic_labels, coarse_foreground.foreground_mask
            )

        # 粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合
        with Timer("詳細セマンティックセグメンテーション処理時間"):
            refined = self.refiner.refine(
                resized, coarse_foreground.foreground_map, foreground_mask, semantic_labels, labels_list, width, height,
                edgetam_features, layout,
            )

        elapsed_time = time.time() - start
        Timer.log(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        return SegmentationResult(
            semantic_map=refined.semantic_map,
            prompt_indices=refined.prompt_indices,
            point_coords=refined.point_coords,
            point_labels=refined.point_labels,
            foreground_map=coarse_foreground.foreground_map,
            foreground_mask=foreground_mask,
            cluster_labels=cluster_labels,
            raw_semantic_labels=raw_semantic_labels,
            semantic_labels=semantic_labels,
            cluster_index_list=cluster_index_list,
            pe_clusters=coarse_foreground.pe_clusters,
            debug_images=refined.debug_images,
        )

    # -------------------------------------------------------------------------
    # segment の各段階
    # -------------------------------------------------------------------------

    def _preprocess(self, image_raw: np.ndarray) -> Tuple[ImageLayout, np.ndarray, np.ndarray]:
        """リサイズと正規化。(レイアウト, リサイズ済み画像（正規化前）, CLIP の入力テンソル) を返す。"""
        height, width = image_raw.shape[:2]
        with Timer("画像前処理時間"):
            # リサイズ（パディング込み）はCLIPとEdgeTAMで共通（正規化だけがモデルごとに異なる）
            layout = make_layout((height, width), self.extractor.target_size, int(self.extractor.patch_size), self.resize_mode)
            resized = layout.apply(image_raw)
            image_tensor = normalize_image(resized, self.extractor.mean, self.extractor.std)
            # 以降のパッチ格子は、パディングを除いた（元画像の全体を覆う）領域
            self.coarse.grid = layout.valid_grid
        return layout, resized, image_tensor

    def _encode_edgetam(self, layout: ImageLayout, resized: np.ndarray) -> Optional[Tuple]:
        """EdgeTAMの特徴量抽出。粗い判定で EdgeTAM の特徴を使うときだけ、ここで 1 回実行して精緻化にも渡す（使わなければ None）。"""
        if not self.use_edgetam_clustering:
            return None
        with Timer("EdgeTAM特徴量抽出処理時間"):
            return layout.crop_edgetam(self.refiner.predict_encoder(resized))

    def _start_edge_graph(self, edgetam_features: Optional[Tuple]):
        """EdgeTAM の image_embed の kNN グラフを、pool があれば別スレッドで求め始める（なければ、粗い判定の中で求める。None を返す）。"""
        if edgetam_features is None or self.pool is None:
            return None
        return self.pool.submit(self.coarse.edge_graph_of, edgetam_features[2])

    def _extract_clip_features(self, layout: ImageLayout, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CLIP（PE）のパッチ特徴と fused パッチ特徴（パディング部分のトークンは捨てる）。"""
        with Timer("特徴量抽出処理時間"):
            _, patch_features, fused_patch_features = self.coarse.extract_features(image_tensor)
            return layout.crop_tokens(patch_features), layout.crop_tokens(fused_patch_features)

    def _estimate_coarse_foreground(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray,
        edgetam_features: Optional[Tuple],
        edge_graph=None,
    ) -> CoarseForeground:
        """CLIP + KMeans による初期前景マップの推定（EdgeTAM の特徴があれば、前景確信度の伝播も行う）。"""
        with Timer("粗い前景マスク取得処理時間"):
            if edgetam_features is None:
                generated = self.coarse.generate_foreground_map(patch_features, fused_patch_features)
            else:
                edgetam_image_embed = edgetam_features[2]  # (high_res_feats0, high_res_feats1, image_embed) の最後
                generated = self.coarse.generate_foreground_map(
                    patch_features, fused_patch_features, edgetam_image_embed, None if edge_graph is None else edge_graph.result()
                )
            # 4 要素のタプルを返す差し替え実装（tools）でも動くよう、名前付きにそろえる
            return CoarseForeground(*generated)

    def _save_name(self, image_path: Path) -> str:
        """保存先のフォルダ名。画像名が別の画像と重なるときは 親フォルダ名_ファイル名 にする。"""
        resolved = image_path.resolve()
        name = image_path.stem
        if self._save_names.setdefault(name, resolved) != resolved:
            name = f"{image_path.parent.name}_{image_path.stem}"
            self._save_names.setdefault(name, resolved)
        return name

    def run(self, image_path: Union[str, Path]) -> None:
        """画像ファイルをセグメンテーションし、結果を output_dir/<画像名>/ に保存する。"""
        image_path = Path(image_path)

        if image_path.exists():
            print(f"====== 画像パス: {image_path.name} ======")
        else:
            print(f"====== 存在しません: {image_path.name} ======")
            return

        save_dir = Path(self.output_dir) / self._save_name(image_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        image_raw = load_image(image_path)
        start = time.time()
        result = self.segment(image_raw)
        elapsed_ms = (time.time() - start) * 1000

        self.saver.save(save_dir, image_raw, result, self.foreground_prompts + self.background_prompts)

        # 処理済みの画像の一覧HTMLを output_dir 直下に更新する（途中でも見られる）
        if self.save_gallery_html:
            shutil.copy2(image_path, save_dir / f"{ORIGINAL_STEM}{image_path.suffix}")
            self._gallery_times[save_dir] = elapsed_ms
            print(f"一覧HTML: {write_gallery_html(Path(self.output_dir), self._gallery_times)}")
