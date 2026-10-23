"""結果の保存・可視化（セグメンテーション結果には影響しない）。"""

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from core.common.result import SegmentationResult

# セマンティック可視化の色 (RGB)。0番は背景で常にシアン
SEMANTIC_COLORS = np.array([
    [0, 255, 255],
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [255, 128, 0], [128, 0, 255], [255, 255, 255],
    [128, 64, 0], [255, 128, 192], [0, 128, 0], [128, 128, 128],
], dtype=np.uint8)
# 前景オーバーレイ（粗い前景マスク・EdgeTAM最終マスク）の赤の不透明度
FOREGROUND_OVERLAY_ALPHA = 0.3
# 前景オーバーレイに描くプロンプト点の半径(px)と色(BGR)。正例=緑、負例=赤
POINT_RADIUS = 20
POINT_COLOR_POSITIVE = (0, 255, 0)
POINT_COLOR_NEGATIVE = (0, 0, 255)
# セマンティック結果（重畳 + 凡例）のmatplotlib図
RESULT_FIGSIZE = (10, 6)
RESULT_DPI = 150
LEGEND_FONTSIZE = 9


def clean_prompt(prompt: str) -> str:
    """"a photo of a " などの固有表現と末尾のピリオドを除いたクラス名にする。"""
    return re.sub(r"^\s*a photo of (?:an?\s+)?", "", prompt, flags=re.IGNORECASE).rstrip(" .")


def semantic_class_names(all_prompts: List[str], prompt_indices: List[int]) -> List[str]:
    """クラス0=background、クラス1~=前景プロンプト名のリストを返す。"""
    return ["background"] + [clean_prompt(all_prompts[i]) for i in prompt_indices]


def semantic_colors(num_classes: int) -> np.ndarray:
    """クラス数分の色 (RGB, uint8)。足りなければ繰り返し、0番は常にシアン。"""
    fg = SEMANTIC_COLORS[1:]
    return np.vstack([SEMANTIC_COLORS[:1], fg[np.arange(num_classes - 1) % len(fg)]])[:num_classes]


def save_debug_image(debug_dir: Path, name: str, data: np.ndarray, width: int, height: int) -> None:
    """デバッグ用の中間画像を debug_dir に元画像サイズで保存する。"""
    debug_dir.mkdir(parents=True, exist_ok=True)
    image = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(debug_dir / f"{name}.png"), (image * 255).astype(np.uint8))


class ResultSaver:
    """処理結果の画像・マスク・オーバーレイの書き出し。"""

    def __init__(self, config: dict, feature_size: int):
        self.feature_size = feature_size
        self.k_clusters = config["k_clusters"]
        self.color_palette = config["color_palette"]
        self.overlay_alpha = config["overlay_alpha"]
        self.show_result = config["show_result"]

    def save(
        self,
        save_dir: Path,
        image_raw: np.ndarray,
        result: SegmentationResult,
        class_names: List[str],
    ) -> None:
        height, width = image_raw.shape[:2]
        size = (width, height)  # PIL の resize に渡す (幅, 高さ)

        # 最終結果
        self._save_semantic_result(save_dir, image_raw, result.semantic_map, class_names)
        self._save_class_masks(save_dir, result.semantic_map, class_names)
        self._save_foreground_overlay_final(save_dir, image_raw, result.semantic_map)

        # 粗い推定の中間結果
        self._save_cluster_maps(save_dir, size, result)
        mask_image = self._save_foreground_maps(save_dir, size, result)
        self._save_foreground_overlay(save_dir, image_raw, mask_image, result)

        print(f"Saving results to {save_dir}")

    # -------------------------------------------------------------------------
    # 最終結果
    # -------------------------------------------------------------------------

    def _save_semantic_result(
        self,
        save_dir: Path,
        image_raw: np.ndarray,
        semantic_map: np.ndarray,
        class_names: List[str],
    ) -> None:
        """セマンティックマップの保存と、元画像への重畳＋凡例の表示。"""
        colors = semantic_colors(len(class_names))
        color_image = colors[semantic_map]

        # クラスIDそのものを保存（パレット付きPNG）
        index_image = Image.fromarray(semantic_map, mode="P")
        index_image.putpalette(colors.flatten().tolist())
        index_image.save(save_dir / "semantic_map.png")

        # 元画像に色を重畳
        overlay = cv2.addWeighted(
            image_raw, 1.0 - self.overlay_alpha, color_image, self.overlay_alpha, 0
        )
        cv2.imwrite(str(save_dir / "semantic_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 重畳画像 + 色とプロンプトの凡例
        import matplotlib
        if not self.show_result:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        ratios = np.bincount(semantic_map.ravel(), minlength=len(class_names)) / semantic_map.size
        handles = [
            Patch(facecolor=colors[i] / 255.0, edgecolor="black",
                  label=f"{i}: {name} ({ratios[i] * 100:.1f}%)")
            for i, name in enumerate(class_names)
        ]
        fig, ax = plt.subplots(figsize=RESULT_FIGSIZE)
        ax.imshow(overlay)
        ax.axis("off")
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=LEGEND_FONTSIZE)
        fig.savefig(save_dir / "semantic_result.png", bbox_inches="tight", dpi=RESULT_DPI)
        if self.show_result:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _save_class_masks(save_dir: Path, semantic_map: np.ndarray, class_names: List[str]) -> None:
        """クラスごとの最終マスクを保存。"""
        for class_id in range(1, len(class_names)):
            mask_img = ((semantic_map == class_id) * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"foreground_mask_n{class_id}.png"), mask_img)

    @staticmethod
    def _save_foreground_overlay_final(save_dir: Path, image_raw: np.ndarray, semantic_map: np.ndarray) -> None:
        """最終マスク（EdgeTAM精緻化後）の前景を赤で重畳した画像を保存。"""
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = ((semantic_map > 0) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_cv, 1.0 - FOREGROUND_OVERLAY_ALPHA, colored_mask, FOREGROUND_OVERLAY_ALPHA, 0)
        cv2.imwrite(str(save_dir / "foreground_overlay_edgetam.png"), overlay)

    # -------------------------------------------------------------------------
    # 粗い推定の中間結果
    # -------------------------------------------------------------------------

    def _colorize_labels(self, labels_2d: np.ndarray, size: Tuple[int, int]) -> Image.Image:
        """ラベルマップをパレットで着色し、元画像サイズへ拡大する。"""
        image = Image.fromarray(labels_2d.astype(np.uint8))
        image.putpalette(self.color_palette)
        return image.resize(size, resample=Image.Resampling.NEAREST)

    def _save_cluster_maps(self, save_dir: Path, size: Tuple[int, int], result: SegmentationResult) -> None:
        """KMeansのクラスタ、整形前後の粗いセマンティックラベルを着色して保存。"""
        grid = (self.feature_size, self.feature_size)
        self._colorize_labels(result.cluster_labels.reshape(grid), size).save(
            save_dir / f"layer_fused_k{self.k_clusters}.png"
        )
        self._colorize_labels(result.semantic_labels.reshape(grid), size).save(
            save_dir / "coarse_semantic_segmentation.png"
        )
        self._colorize_labels(result.raw_semantic_labels.reshape(grid), size).save(
            save_dir / "coarse_semantic_segmentation_raw.png"
        )

    def _save_foreground_maps(self, save_dir: Path, size: Tuple[int, int], result: SegmentationResult) -> Image.Image:
        """前景度合いマップと粗い前景マスクを保存し、マスク画像を返す。"""
        grid = (self.feature_size, self.feature_size)
        foreground_map_2d = (result.foreground_map * 255).astype(np.uint8).reshape(grid)
        Image.fromarray(foreground_map_2d).resize(size, resample=Image.Resampling.BILINEAR).save(
            save_dir / "foreground_map.png"
        )

        foreground_mask_2d = (result.foreground_mask * 255).astype(np.uint8).reshape(grid)
        mask_image = Image.fromarray(foreground_mask_2d).resize(size, resample=Image.Resampling.NEAREST)
        mask_image.save(save_dir / "foreground_mask.png")
        return mask_image

    @staticmethod
    def _save_foreground_overlay(
        save_dir: Path, image_raw: np.ndarray, mask_image: Image.Image, result: SegmentationResult
    ) -> None:
        """粗い前景マスクを赤で重畳し、EdgeTAMに与えたプロンプト点を描いた画像を保存。"""
        image_cv = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)
        colored_mask = np.zeros_like(image_cv)
        colored_mask[:, :, 2] = np.array(mask_image)
        overlay = cv2.addWeighted(image_cv, 1.0 - FOREGROUND_OVERLAY_ALPHA, colored_mask, FOREGROUND_OVERLAY_ALPHA, 0)

        if np.sum(result.foreground_mask == 1) > 0:
            for point_label, point_coord in zip(result.point_labels, result.point_coords):
                color = POINT_COLOR_POSITIVE if point_label == 1 else POINT_COLOR_NEGATIVE
                cv2.circle(overlay, (int(point_coord[0]), int(point_coord[1])), POINT_RADIUS, color, -1)
        cv2.imwrite(str(save_dir / "foreground_overlay.png"), overlay)
