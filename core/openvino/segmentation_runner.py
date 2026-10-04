import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import faiss
import numpy as np
from PIL import Image

from core.utils import *
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.edgetam_runner import OVEdgeTamRunner

from scipy.ndimage import distance_transform_edt

class SegmentationRunner:
    """抽出された特徴量・クラスタリングを用いた推論・マスク生成の実行クラス"""

    def __init__(
        self, 
        extractor: OVCLIPFeatureExtractor, 
        edgetam: OVEdgeTamRunner, 
        config: yaml
    ):
        # モデル
        self.extractor = extractor
        self.edgetam = edgetam

        # 設定値の初期化
        self.init_value(config)
        
        self.register_fgbg_text_features()

    def init_value(self, config: yaml):
        # config関係
        self.prompts_dir = config["prompts_dir"]
        self.k_clusters = config["k_clusters"]
        self.niter = config["n_iter"]
        self.foreground_threshold = config["foreground_threshold"]
        self.output_dir = config["output_dir"]
        self.color_palette = config["color_palette"]
        self.init_n_points = config["init_n_points"]
        self.max_refine_steps = config["max_refine_steps"]

        # 特徴マップのサイズ
        self.feature_size: int = self.extractor.feature_size

        # プロンプトとプロンプトのテキスト特徴量
        self.foreground_prompts: List[str] = []
        self.background_prompts: List[str] = []
        self.fgbg_text_features: Optional[np.ndarray] = None

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    # =========================================================================
    # 1. テキストプロンプト & 特徴量準備
    # =========================================================================

    def load_prompts(self, path: str) -> List[str]:
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return data["prompts"]
    
    def common_process(self, prompts: List[str]) -> np.ndarray:
        text_features = self.extractor.extract_text_features(prompts)
        return text_features
    
    def register_fgbg_text_features(self) -> None:
        foreground_prompts_path = Path(self.prompts_dir) / "foreground.json"
        background_prompts_path = Path(self.prompts_dir) / "background.json"

        self.foreground_prompts = self.load_prompts(foreground_prompts_path)
        self.background_prompts = self.load_prompts(background_prompts_path)

        fg_feats = self.common_process(self.foreground_prompts)
        bg_feats = self.common_process(self.background_prompts)

        self.fgbg_text_features = np.vstack([fg_feats, bg_feats])

    # =========================================================================
    # 2. クラスタリング & 可視化ユーティリティ
    # =========================================================================

    def fit_predict_kmeans(
        self, 
        features: np.ndarray, 
        k: int = 12, 
        niter: int = 10
    ) -> np.ndarray:
        N, D = features.shape
        features_contig = (
            np.ascontiguousarray(features, dtype=np.float32)
            if not features.flags["C_CONTIGUOUS"]
            else features
        )
        faiss.omp_set_num_threads(2)
        kmeans = faiss.Kmeans(
            d=D, k=k, niter=niter, spherical=False, verbose=False, seed=42
        )
        kmeans.train(features_contig)
        _, cluster_labels = kmeans.index.search(features_contig, 1)
        return cluster_labels.reshape(-1)

    def colorize_clusters_PIL(self, cluster_labels: np.ndarray) -> Image.Image:
        color_map = Image.fromarray(cluster_labels.astype(np.uint8))
        color_map.putpalette(self.color_palette)
        return color_map

    # =========================================================================
    # 3. プロンプトポイント選出 & 座標変換 (ロジック不変)
    # =========================================================================

    # def select_n_prompt_points(
    #     self, 
    #     target_mask: np.ndarray, 
    #     opponent_mask: np.ndarray, 
    #     num_points: int = 1
    # ) -> np.ndarray:
    #     """
    #     対象領域（target_mask）の中から、条件を満たす点を順に num_points 個抽出する:
    #     1. 領域の内側に位置する（境界から遠い）
    #     2. 対立領域（opponent_mask）から遠い
    #     3. 既に選ばれている他のプロンプト点から遠い
    #     """
    #     if not np.any(target_mask) or num_points <= 0:
    #         return np.empty((0, 2), dtype=np.int32)

    #     dist_target = distance_transform_edt(target_mask)
    #     dist_opponent = distance_transform_edt(1 - opponent_mask)

    #     norm_target = dist_target / (np.max(dist_target) + 1e-8)
    #     norm_opponent = dist_opponent / (np.max(dist_opponent) + 1e-8)

    #     base_score_map = norm_target * norm_opponent
    #     base_score_map[target_mask == 0] = -1.0  # 対象領域外は除外

    #     selected_points = []
    #     h, w = target_mask.shape
    #     grid_y, grid_x = np.ogrid[:h, :w]

    #     for i in range(num_points):
    #         if i == 0:
    #             score_map = base_score_map.copy()
    #         else:
    #             dist_to_selected = np.ones((h, w), dtype=np.float32) * np.inf
    #             for py, px in selected_points:
    #                 d = np.sqrt((grid_y - py)**2 + (grid_x - px)**2)
    #                 dist_to_selected = np.minimum(dist_to_selected, d)
                
    #             norm_dist_selected = dist_to_selected / (np.max(dist_to_selected) + 1e-8)
    #             score_map = base_score_map * norm_dist_selected

    #         best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
    #         selected_points.append([int(best_y), int(best_x)])

    #     return [[x, y] for y, x in selected_points]

    # def select_n_prompt_points(
    #     self, 
    #     target_mask: np.ndarray, 
    #     opponent_mask: Optional[np.ndarray] = None, 
    #     num_points: int = 1,
    #     min_area_ratio: float = 0.001  # 全体解像度に対する最小領域面積の割合 (0.1%)
    # ) -> np.ndarray:
    #     """
    #     粗いマスクからノイズを除去し、主要領域の中心付近から分散したプロンプト点 [x, y] を抽出する。

    #     Parameters:
    #         target_mask (np.ndarray): 抽出対象の二値マスク (H, W)
    #         opponent_mask (np.ndarray, optional): 対立領域の二値マスク (H, W)
    #         num_points (int): 抽出するポイント数
    #         min_area_ratio (float): ノイズとして除元する最小面積の割合

    #     Returns:
    #         np.ndarray: 選択された座標のリスト [[x1, y1], [x2, y2], ...] (shape: N x 2)
    #     """
    #     if target_mask is None or not np.any(target_mask) or num_points <= 0:
    #         return np.empty((0, 2), dtype=np.int32)

    #     # 1. 前処理（二値化）
    #     binary_mask = (target_mask > 0).astype(np.uint8)
    #     h, w = binary_mask.shape
    #     total_pixels = h * w

    #     # 2. 連結成分解析によるノイズ（小さな孤立領域）の除去
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #         binary_mask, connectivity=8
    #     )

    #     min_area = max(1, int(total_pixels * min_area_ratio))
    #     cleaned_mask = np.zeros_like(binary_mask)
    #     area_weights = np.zeros_like(binary_mask, dtype=np.float32)
    #     max_area = 0

    #     for label in range(1, num_labels):  # 0は背景
    #         area = stats[label, cv2.CC_STAT_AREA]
    #         if area >= min_area:
    #             cleaned_mask[labels == label] = 1
    #             area_weights[labels == label] = area
    #             if area > max_area:
    #                 max_area = area

    #     # 万が一すべての領域がノイズ除去で消えた場合は、最も大きい1つの領域を復元
    #     if not np.any(cleaned_mask) and num_labels > 1:
    #         max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #         cleaned_mask[labels == max_label] = 1
    #         area_weights[labels == max_label] = stats[max_label, cv2.CC_STAT_AREA]
    #         max_area = stats[max_label, cv2.CC_STAT_AREA]

    #     if not np.any(cleaned_mask):
    #         return np.empty((0, 2), dtype=np.int32)

    #     # 面積スコアの正規化 (0 ~ 1)
    #     norm_area_map = area_weights / (max_area + 1e-8)

    #     # 3. 距離変換（境界からの距離算出）
    #     dist_target = distance_transform_edt(cleaned_mask)
    #     norm_dist_target = dist_target / (np.max(dist_target) + 1e-8)

    #     # 4. ベーススコアマップの算出 (中心度 × 領域の大きさ)
    #     base_score_map = norm_dist_target * np.sqrt(norm_area_map)

    #     # 対立領域からの距離を考慮する場合
    #     if opponent_mask is not None and np.any(opponent_mask):
    #         dist_opponent = distance_transform_edt(1 - (opponent_mask > 0).astype(np.uint8))
    #         norm_opponent = dist_opponent / (np.max(dist_opponent) + 1e-8)
    #         base_score_map *= norm_opponent

    #     # 対象領域外はスコアを負の値にして無効化
    #     base_score_map[cleaned_mask == 0] = -1.0

    #     # 5. ポイントの順次選出（複数点の場合、既出点から離れた点を選ぶ）
    #     selected_points = []
    #     grid_y, grid_x = np.ogrid[:h, :w]

    #     for i in range(num_points):
    #         if i == 0:
    #             score_map = base_score_map.copy()
    #         else:
    #             # 既存選択点からの最小ユークリッド距離
    #             dist_to_selected = np.ones((h, w), dtype=np.float32) * np.inf
    #             for py, px in selected_points:
    #                 d = np.sqrt((grid_y - py) ** 2 + (grid_x - px) ** 2)
    #                 dist_to_selected = np.minimum(dist_to_selected, d)

    #             norm_dist_selected = dist_to_selected / (np.max(dist_to_selected) + 1e-8)
    #             score_map = base_score_map * norm_dist_selected

    #         best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)

    #         # 候補となる有効なピクセルがない場合は終了
    #         if score_map[best_y, best_x] < 0:
    #             break

    #         selected_points.append([int(best_y), int(best_x)])

    #     # [x, y] (列, 行) の順で N x 2 の numpy 配列を返す
    #     return [[x, y] for y, x in selected_points]

    def select_prompt_points(
        self, 
        fg_mask: np.ndarray, 
        num_fg_points: int = 5, 
        num_bg_points: int = 5,
        dilate_kernel_size: int = 31  # 麺や周囲をカバーするための膨張サイズ
    ) -> Tuple[np.ndarray, np.ndarray]:
        """粗い前景マスクから、最適な前景点(緑)と近傍背景点(赤)を選出する。

        Args:
            fg_mask (np.ndarray): 粗い前景マスク (H, W), uint8 (0 or 255) または bool
            num_fg_points (int): 前景ポイント数
            num_bg_points (int): 背景ポイント数
            dilate_kernel_size (int): 近傍背景を定義するための膨張カーネルサイズ (奇数)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (fg_points, bg_points) 各 shape: (N, 2) [x, y]
        """
        binary_fg = (fg_mask > 0).astype(np.uint8)
        h, w = binary_fg.shape
        grid_y, grid_x = np.ogrid[:h, :w]

        # ==========================================
        # 1. 前景ポイント（緑点）の選出
        # ==========================================
        fg_points = []
        if np.any(binary_fg) and num_fg_points > 0:
            dist_fg = distance_transform_edt(binary_fg)
            base_fg_score = dist_fg / (np.max(dist_fg) + 1e-8)
            
            for i in range(num_fg_points):
                if i == 0:
                    score_map = base_fg_score.copy()
                else:
                    dist_to_selected = np.ones((h, w), dtype=np.float32) * np.inf
                    for py, px in fg_points:
                        d = np.sqrt((grid_y - py)**2 + (grid_x - px)**2)
                        dist_to_selected = np.minimum(dist_to_selected, d)
                    score_map = base_fg_score * (dist_to_selected / (np.max(dist_to_selected) + 1e-8))

                best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
                if score_map[best_y, best_x] <= 0:
                    break
                fg_points.append((best_y, best_x))

        # ==========================================
        # 2. 背景ポイント（赤点）の選出（近傍エリアに限定）
        # ==========================================
        bg_points = []
        if np.any(binary_fg) and num_bg_points > 0:
            # モルフォロジー膨張で「対象の周囲領域」を作成
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
            dilated_fg = cv2.dilate(binary_fg, kernel, iterations=1)
            
            # 「近傍背景」＝ 膨張領域 から 元の前景 を引いたリング状領域
            near_bg_mask = (dilated_fg == 1) & (binary_fg == 0)

            # 万が一近傍エリアが狭すぎる場合は画像全体をバックアップとする
            if not np.any(near_bg_mask):
                near_bg_mask = (binary_fg == 0)

            # 近傍背景領域内での中心（境界から遠い場所）を特定
            dist_bg = distance_transform_edt(near_bg_mask)
            base_bg_score = dist_bg / (np.max(dist_bg) + 1e-8)
            base_bg_score[near_bg_mask == 0] = -1.0  # 候補領域外は除外

            for i in range(num_bg_points):
                if i == 0:
                    score_map = base_bg_score.copy()
                else:
                    dist_to_selected = np.ones((h, w), dtype=np.float32) * np.inf
                    for py, px in bg_points:
                        d = np.sqrt((grid_y - py)**2 + (grid_x - px)**2)
                        dist_to_selected = np.minimum(dist_to_selected, d)
                    
                    # 既出点から離れた位置を優先
                    norm_dist = dist_to_selected / (np.max(dist_to_selected) + 1e-8)
                    score_map = base_bg_score * norm_dist

                best_y, best_x = np.unravel_index(np.argmax(score_map), score_map.shape)
                if score_map[best_y, best_x] < 0:
                    break
                bg_points.append((best_y, best_x))

        # [x, y] 形式に変換して返却
        fg_res = np.array([[x, y] for y, x in fg_points], dtype=np.int32) if fg_points else np.empty((0, 2), dtype=np.int32)
        bg_res = np.array([[x, y] for y, x in bg_points], dtype=np.int32) if bg_points else np.empty((0, 2), dtype=np.int32)

        return fg_res, bg_res

    def convert_coordinate(self, point: int, img_size: int, feat_size: int) -> float:
        return point * img_size / feat_size

    def transform_coords(self, pts: np.ndarray, width: int, height: int, feature_size: int) -> List[List[float]]:
        return [
            [
                self.convert_coordinate(x + 0.5, width, feature_size),
                self.convert_coordinate(y + 0.5, height, feature_size)
            ]
            for x, y in pts
        ]

    # =========================================================================
    # 4. EdgeTAM 推論バックエンド共通インターフェース (OpenVINO対応化用)
    # =========================================================================

    def _predict_edgetam_encoder(self, image_raw: Image.Image) -> Optional[Tuple]:
        """EdgeTAM Encoder の実行（PyTorch / Custom / 将来のOpenVINOの違いをカプセル化）"""
        return self.edgetam.encode(
            preprocess_image(
                image_raw, 
                self.edgetam.target_size, 
                self.edgetam.mean, self.edgetam.std
            )
        )

    def _predict_edgetam_decoder(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Optional[Tuple],
        height: int,
        width: int,
        mask_input: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """EdgeTAM Decoder の実行（PyTorch / Custom / 将来のOpenVINOの違いをカプセル化）"""
        if mask_input is None:
            mask_logits_tensor = np.array([0], dtype=np.float32)
        else:
            mask_logits_tensor = mask_input

        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self.edgetam.preprocess(
            (height, width),
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
            mask_logits=mask_logits_tensor,
        )

        high_res_feats0, high_res_feats1, image_embed = feats
        low_res_masks, iou_predictions = self.edgetam.decode(
            high_res_feats0, high_res_feats1, image_embed,
            point_coords=point_coords_norm,
            point_labels=point_labels_norm,
            mask_input=mask_input_tensor,
            has_mask=has_mask,
        )

        masks, scores, logits = self.edgetam.postprocess(
            low_res_masks, iou_predictions, (height // 8, width // 8)
        )

        return masks, scores, logits

    # =========================================================================
    # 5. 各パイプライン・サブタスク処理
    # =========================================================================

    def _feature_extraction(
        self, 
        image_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with Timer("特徴量抽出処理時間"):
            image_features, patch_features, cls2patch_attn, fused_patch_features = (
                self.extractor.extract_image_features(image_tensor)
            )
        return image_features, patch_features, fused_patch_features

    def _image_classification(
        self, 
        image_features: np.ndarray
    ):
        # クラス分類
        text_probs = 100.0 * image_features @ self.fgbg_text_features.T
        text_probs = self._softmax(text_probs, axis=-1)

    def _generate_foreground_map(
        self,
        patch_features: np.ndarray, 
        fused_patch_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """CLIP特徴量とKMeansによるベース前景マップの抽出"""
        # パッチレベルの特徴量を取得
        patch_tokens = fused_patch_features[0, 1:, :]

        cluster_labels_list = []
        fg_prompts_num = len(self.foreground_prompts)
        foreground_map = np.zeros((self.feature_size * self.feature_size,), dtype=np.float32)

        for k_clusters in self.k_clusters:
            cluster_labels = self.fit_predict_kmeans(
                patch_tokens, k=k_clusters, niter=self.niter
            )
            cluster_labels_list.append(cluster_labels)

            with Timer("クラスタごとのCLSトークン処理時間"):
                label_token_list: List[np.ndarray] = self.extractor.extract_cluster_tokens(
                    patch_features, cluster_labels
                )

            for label_idx, label_features in enumerate(label_token_list):
                label2text_probs = self._softmax(
                    100.0 * label_features.reshape(1, -1) @ self.fgbg_text_features.T, 
                    axis=-1
                )
                foreground_confidence = np.sum(label2text_probs[0, :fg_prompts_num])
                foreground_map[cluster_labels == label_idx] += foreground_confidence

        foreground_map = foreground_map / len(self.k_clusters)
        foreground_map = foreground_map.reshape(self.feature_size, self.feature_size)
        foreground_mask = np.where(foreground_map > self.foreground_threshold, 1, 0)

        return foreground_map, foreground_mask, cluster_labels_list

    def _run_edgetam_refinement(
        self,
        foreground_mask: np.ndarray,
        image_raw: Image.Image,
        width: int,
        height: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[int]]:
        """EdgeTAM による初期プロンプト設定および繰り返しマスク精緻化"""
        fg_mask = (foreground_mask == 1).astype(np.uint8)
        bg_mask = (foreground_mask == 0).astype(np.uint8)

        # --- Step 1: 初期の最良点（前景5点 + 背景5点）でベースマスクを作成 ---
        # init_fg_pt = self.select_n_prompt_points(fg_mask, bg_mask, num_points=self.init_n_points)
        # init_bg_pt = self.select_n_prompt_points(bg_mask, fg_mask, num_points=self.init_n_points)
        init_fg_pt, init_bg_pt = self.select_prompt_points(fg_mask, self.init_n_points, self.init_n_points, 9)

        point_coords = (
            self.transform_coords(init_fg_pt, width, height, self.feature_size) +
            self.transform_coords(init_bg_pt, width, height, self.feature_size)
        )
        point_labels = [1] * self.init_n_points + [0] * self.init_n_points

        with Timer("edgetam特徴量抽出"):
            feats = self._predict_edgetam_encoder(image_raw)

        with Timer("edgetam1回目マスク推定"):
            masks, scores, logits = self._predict_edgetam_decoder(
                point_coords=point_coords,
                point_labels=point_labels,
                feats=feats,
                height=height,
                width=width
            )

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]

        # --- Step 2: 誤差（差分領域）から動的にポイントを追加して精緻化 ---
        current_mask_list = [current_mask]
        h_gt, w_gt = fg_mask.shape[:2]

        for step in range(self.max_refine_steps):
            current_mask_resized = cv2.resize(
                current_mask, 
                (w_gt, h_gt), 
                interpolation=cv2.INTER_NEAREST
            )

            fg_fn_mask: np.ndarray = (fg_mask == 1) & (current_mask_resized == 0)
            new_pt = self.select_n_prompt_points(fg_fn_mask.astype(np.uint8), bg_mask, num_points=1)

            if len(new_pt) == 0:
                break

            new_coord = self.transform_coords(new_pt, width, height, self.feature_size)[0]
            point_coords.append(new_coord)
            point_labels.append(1)

            with Timer(f"edgetam{step+2}回目マスク推定"):
                masks, scores, logits = self._predict_edgetam_decoder(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    feats=feats,
                    height=height,
                    width=width,
                    mask_input=low_res_logits
                )

            best_idx = np.argmax(scores)
            current_mask = masks[best_idx].astype(np.uint8)
            low_res_logits = logits[best_idx:best_idx + 1]

            current_mask_list.append(current_mask)

        # マスク画像を元画像サイズにリサイズ
        resized_mask: np.ndarray = cv2.resize(
            current_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_mask, current_mask_list, point_coords, point_labels

    def _save_results(
        self,
        image_path: Path, 
        image_raw: Image.Image,
        width: int,
        height: int,
        current_mask: np.ndarray,
        current_mask_list: List[np.ndarray],
        foreground_map: np.ndarray,
        foreground_mask: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        cluster_labels_list: List[np.ndarray]
    ) -> None:
        """処理結果の画像・マスク・オーバーレイの書き出し"""
        save_dir = Path(self.output_dir) / image_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Saving results to {save_dir} ---")

        # マスク画像を保存
        for i, mask in enumerate(current_mask_list):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{i+1}.png"), mask_img)

        # レイヤー結合した特徴マップでのクラスタリング結果
        for k_clusters, cluster_labels in zip(self.k_clusters, cluster_labels_list):
            fused_cluster_labels_2d = cluster_labels.reshape(self.feature_size, self.feature_size)
            fused_cluster_rgb = self.colorize_clusters_PIL(fused_cluster_labels_2d)
            fused_cluster_rgb = fused_cluster_rgb.resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            fused_cluster_rgb.save(save_dir / f"layer_fused_k{k_clusters}.png")

        # 前景度合いマップを保存
        foreground_map_2d = (
            (foreground_map * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        map_image = Image.fromarray(foreground_map_2d).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        map_image.save(save_dir / "foreground_map.png")

        # 初期前景マスク画像を保存
        foreground_mask_2d = (
            (foreground_mask * 255)
            .astype(np.uint8)
            .reshape(self.feature_size, self.feature_size)
        )
        mask_image = Image.fromarray(foreground_mask_2d).resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        mask_image.save(save_dir / "foreground_mask.png")

        # オーバーレイ画像保存 (CLIP前景プロンプト点付き)
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask_image)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = mask_cv
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        
        if np.sum(foreground_mask == 1) > 0:
            for point_label, point_coord in zip(point_labels, point_coords):
                color = (0, 255, 0) if point_label == 1 else (0, 0, 255)
                cv2.circle(
                    overlay, 
                    (int(point_coord[0]), int(point_coord[1])), 
                    20, color, -1
                )
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)

        # オーバーレイ画像（EdgeTAM 最終マスク版）
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = (current_mask * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay_edgetam.png"), overlay)

        print("Save completed.")

    # =========================================================================
    # 6. メイン実行エントリーポイント
    # =========================================================================

    def run(self, image_path: Union[str, Path]) -> None:
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if image_path.exists():
            print(f"====== 画像パス: {image_path.name} ======")
        else:
            print(f"====== 存在しません: {image_path.name} ======")
            return

        # 画像読み込み
        image_raw = load_image(image_path)
        height, width = image_raw.shape[:2]

        # 計測開始
        start = time.time()

        # 画像前処理
        image_tensor = preprocess_image(image_raw, self.extractor.target_size, self.extractor.mean, self.extractor.std)

        # 特徴量抽出
        image_features, patch_features, fused_patch_features = self._feature_extraction(image_tensor)

        # 画像分類
        self._image_classification(image_features)

        # CLIP + KMeans による初期前景マップの推定
        foreground_map, foreground_mask, cluster_labels_list = self._generate_foreground_map(patch_features, fused_patch_features)

        # 前景が存在する場合は EdgeTAM による精緻化
        point_coords: List[List[float]] = []
        point_labels: List[int] = []

        if np.sum(foreground_mask == 1) > 0:
            current_mask, current_mask_list, point_coords, point_labels = self._run_edgetam_refinement(
                foreground_mask=foreground_mask,
                image_raw=image_raw,
                width=width,
                height=height
            )
        else:
            current_mask = np.zeros((height, width), dtype=np.uint8)
            current_mask_list = []

        elapsed_time = time.time() - start
        print(f"Inference Time: {elapsed_time * 1000:.2f} ms")

        # 処理結果の保存
        self._save_results(
            image_path=image_path,
            image_raw=image_raw,
            width=width,
            height=height,
            current_mask=current_mask,
            current_mask_list=current_mask_list,
            foreground_map=foreground_map,
            foreground_mask=foreground_mask,
            point_coords=point_coords,
            point_labels=point_labels,
            cluster_labels_list=cluster_labels_list
        )