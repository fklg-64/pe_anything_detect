"""CLIPによる粗い推定 -> EdgeTAMによる精緻化 の全体の流れ（画像1枚分）をつなぐ。

処理の流れ（SegmentationRunner.segment。各段階の詳細は docs/overview.md）:

  1. 前処理               image_size 四方へのリサイズと正規化
  2. EdgeTAM エンコーダ   use_edgetam_clustering のとき、粗い判定の前に 1 回だけ実行（精緻化でも使う）
  3. CLIP（PE）の特徴量   パッチ特徴・fused パッチ特徴        <- この間に、EdgeTAM の特徴の kNN グラフを別スレッドで求める
  4. 粗い前景/クラスタ    CoarseSegmenter.generate_foreground_map（KMeans + テキスト照合 + EdgeTAM 特徴での伝播 + クラスのまとめ）
  5. 粗いセマンティック   assign_semantic_labels（パッチ単位のラベル）
  6. 精緻化               SemanticRefiner.refine（クラスごとに EdgeTAM デコーダ -> 1 枚のセマンティックマップ）
                          <- この間に、領域の推定（NumPy の整形）を、別の領域のデコードと重ねる

重ねる処理は worker_threads のスレッドで実行する（parallel.py。0 なら重ねず、順に実行する）。結果は決まった順に反映するので、出力は変わらない。
結果の保存は run() / ResultSaver（segment には含まれない）。
"""

import time
import shutil
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from core.utils import Timer, load_image, resize_image
from core.common.coarse_segmentation import CoarseForeground, CoarseSegmenter
from core.common.backend import CLIPBackend, EdgeTamBackend
from core.common.edgetam_helper import EdgeTamHelper
from core.common.edgetam_session import EdgeTamSession
from core.common.parallel import Job, Workers
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
        self.use_edgetam_clustering = config["use_edgetam_clustering"]
        # 画像は image_size 四方にリサイズする（アスペクト比は保たない）。パッチ格子が入力の全体を覆うよう、パッチサイズの倍数であること
        if extractor.target_size % extractor.patch_size:
            raise ValueError(f"image_size はパッチサイズ {extractor.patch_size} の倍数にしてください: {extractor.target_size}")

        # モデルの推論（OpenVINO・PyTorch は GIL を離す）と、NumPy / OpenCV の整形を重ねて実行する Python のスレッド（0 なら重ねず、従来どおり順に実行する）。
        # 結果は従来と同じ順に反映するので出力は変わらない。モデルの推論のスレッド数（ov_num_thread）は変わらない
        self.workers = Workers(int(config["worker_threads"]))

        self.session = EdgeTamSession(edgetam)
        self.coarse = CoarseSegmenter(extractor, config)
        self.refiner = SemanticRefiner(self.session, extractor.feature_size, config, self.workers)
        self.saver = ResultSaver(config)
        self.save_gallery_html = config["save_gallery_html"]
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

        resized = self._preprocess(image_raw)
        edgetam_features = self._encode_edgetam(resized)
        # EdgeTAM の特徴の kNN グラフ（PE の結果に依らない）は、PE の推論と重ねて求める
        edge_graph = self._start_edge_graph(edgetam_features)
        patch_features, fused_patch_features = self._extract_clip_features(resized)
        coarse_foreground = self._estimate_coarse_foreground(patch_features, fused_patch_features, edgetam_features, edge_graph)
        cluster_labels, cluster_index_list = coarse_foreground.cluster_labels, coarse_foreground.cluster_index_list

        # セマンティックセグメンテーションのラベルを取得
        with Timer("セマンティックセグメンテーション処理時間"):
            labels_list, semantic_labels = self.coarse.assign_semantic_labels(cluster_labels, cluster_index_list)

        foreground_mask = coarse_foreground.foreground_mask

        # 粗いセマンティックセグメンテーションをEdgeTAMで精緻化し、1枚のセマンティックマップに統合
        with Timer("詳細セマンティックセグメンテーション処理時間"):
            refined = self.refiner.refine(
                resized, coarse_foreground.foreground_map, foreground_mask, semantic_labels, labels_list, width, height,
                edgetam_features,
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
            semantic_labels=semantic_labels,
            cluster_index_list=cluster_index_list,
            pe_clusters=coarse_foreground.pe_clusters,
            debug_images=refined.debug_images,
        )

    # -------------------------------------------------------------------------
    # segment の各段階
    # -------------------------------------------------------------------------

    def _preprocess(self, image_raw: np.ndarray) -> np.ndarray:
        """リサイズ済み画像（正規化前）を返す。正規化はモデルごとに、各特徴抽出の中で行う。"""
        with Timer("画像前処理時間"):
            # リサイズは CLIP と EdgeTAM で共通
            return resize_image(image_raw, self.extractor.target_size)

    def _encode_edgetam(self, resized: np.ndarray) -> Optional[Tuple]:
        """EdgeTAMの特徴量抽出。粗い判定で EdgeTAM の特徴を使うときだけ、ここで 1 回実行して精緻化にも渡す（使わなければ None）。"""
        if not self.use_edgetam_clustering:
            return None
        with Timer("EdgeTAM特徴量抽出処理時間"):
            return self.session.encode(resized)

    def _start_edge_graph(self, edgetam_features: Optional[Tuple]) -> Optional[Job]:
        """EdgeTAM の image_embed の kNN グラフ作りを、別スレッドで始める（スレッドが無ければ、result() を呼んだときに実行する）。"""
        if edgetam_features is None:
            return None
        return self.workers.submit(partial(self.coarse.edge_graph_of, edgetam_features[2]))

    def _extract_clip_features(self, resized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CLIP（PE）のパッチ特徴と fused パッチ特徴（正規化も含む）。"""
        with Timer("特徴量抽出処理時間"):
            return self.coarse.extract_features(resized)

    def _estimate_coarse_foreground(
        self,
        patch_features: np.ndarray,
        fused_patch_features: np.ndarray,
        edgetam_features: Optional[Tuple],
        edge_graph: Optional[Job] = None,
    ) -> CoarseForeground:
        """CLIP + KMeans による初期前景マップの推定（EdgeTAM の特徴があれば、前景確信度の伝播も行う）。"""
        with Timer("粗い前景マスク取得処理時間"):
            if edgetam_features is None:
                generated = self.coarse.generate_foreground_map(patch_features, fused_patch_features)
            else:
                edgetam_image_embed = edgetam_features[2]  # (high_res_feats0, high_res_feats1, image_embed) の最後
                generated = self.coarse.generate_foreground_map(
                    patch_features, fused_patch_features, edgetam_image_embed, edge_graph.result()
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
            shutil.copyfile(image_path, save_dir / f"{ORIGINAL_STEM}{image_path.suffix}")
            self._gallery_times[save_dir] = elapsed_ms
            print(f"一覧HTML: {write_gallery_html(Path(self.output_dir), self._gallery_times)}")
