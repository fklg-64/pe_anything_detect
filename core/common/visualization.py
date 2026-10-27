"""結果の保存・可視化（セグメンテーション結果には影響しない）。"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from core.common.result import ClusterView, SegmentationResult

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
# 精緻化の中間画像を入れる、保存先の下のフォルダ名
DEBUG_DIR_NAME = "debug"


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
        all_prompts: List[str],
    ) -> None:
        """all_prompts: 前景プロンプト -> 背景プロンプトの順に並べた全プロンプト（プロンプト番号はこの並びの添字）。"""
        class_names = semantic_class_names(all_prompts, result.prompt_indices)
        height, width = image_raw.shape[:2]
        size = (width, height)  # PIL の resize に渡す (幅, 高さ)

        # 最終結果
        self._save_semantic_result(save_dir, image_raw, result.semantic_map, class_names)
        self._save_class_masks(save_dir, result.semantic_map, class_names)
        self._save_foreground_overlay_final(save_dir, image_raw, result.semantic_map)

        # 粗い推定の中間結果
        self._save_cluster_maps(save_dir, size, result)
        self._save_cluster_prompts(save_dir, result, all_prompts)
        mask_image = self._save_foreground_maps(save_dir, size, result)
        self._save_foreground_overlay(save_dir, image_raw, mask_image, result)

        # 精緻化の中間結果
        self._save_debug_images(save_dir / DEBUG_DIR_NAME, size, result.debug_images)

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
        """KMeansのクラスタ（EdgeTAM併用時は伝播前のPEも）、整形前後の粗いセマンティックラベルを着色して保存。"""
        grid = result.foreground_map.shape  # パッチ格子（アスペクト比を維持するモードでは正方形とは限らない）
        for image_name, view in self._cluster_views(result):
            self._colorize_labels(view.labels.reshape(grid), size).save(save_dir / image_name)
        self._colorize_labels(result.semantic_labels.reshape(grid), size).save(
            save_dir / "coarse_semantic_segmentation.png"
        )
        self._colorize_labels(result.raw_semantic_labels.reshape(grid), size).save(
            save_dir / "coarse_semantic_segmentation_raw.png"
        )

    def _cluster_views(self, result: SegmentationResult) -> List[Tuple[str, ClusterView]]:
        """保存するクラスタ分けを (画像ファイル名, クラスタ分け) で返す。EdgeTAM 併用時は、伝播前の PE、伝播後の順。"""
        views = []
        if result.pe_clusters is not None:
            views.append((f"layer_fused_pe_k{self.k_clusters}.png", result.pe_clusters))
        views.append((f"layer_fused_k{self.k_clusters}.png", ClusterView(result.cluster_labels, result.cluster_index_list)))
        return views

    def _save_cluster_prompts(self, save_dir: Path, result: SegmentationResult, all_prompts: List[str]) -> None:
        """クラスタごとの分類結果（対応するプロンプトの文）をテキストで保存。クラスタの画像と同じクラスタ番号。"""
        lines: List[str] = []
        for image_name, view in self._cluster_views(result):
            lines.append(f"[{image_name}]")
            counts = np.bincount(view.labels, minlength=len(view.index_list))
            for cluster_id, prompt_index in enumerate(view.index_list):
                lines.append(f"クラスタ{cluster_id} ({counts[cluster_id]}パッチ): {all_prompts[prompt_index]}")
            lines.append("")
        (save_dir / "cluster_prompts.txt").write_text("\n".join(lines), encoding="utf-8")

    def _save_foreground_maps(self, save_dir: Path, size: Tuple[int, int], result: SegmentationResult) -> Image.Image:
        """前景度合いマップと粗い前景マスクを保存し、マスク画像を返す。"""
        grid = result.foreground_map.shape
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

    # -------------------------------------------------------------------------
    # 精緻化の中間結果
    # -------------------------------------------------------------------------

    @staticmethod
    def _save_debug_images(debug_dir: Path, size: Tuple[int, int], debug_images: Dict[str, np.ndarray]) -> None:
        """精緻化の中間画像（0〜1 の2次元配列）を元画像サイズへ拡大して debug_dir に保存する。"""
        if not debug_images:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, data in debug_images.items():
            image = np.clip(np.asarray(data, dtype=np.float32), 0.0, 1.0)
            image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(debug_dir / f"{name}.png"), (image * 255).astype(np.uint8))


# 一覧HTMLに並べる画像（列見出し, ファイル名）。元画像は保存先ではなく入力画像のパスを参照する
GALLERY_IMAGES = [
    ("foreground_map", "foreground_map.png"),
    ("coarse_semantic_segmentation", "coarse_semantic_segmentation.png"),
    ("foreground_overlay", "foreground_overlay.png"),
    ("semantic_result", "semantic_result.png"),
]
GALLERY_FILE_NAME = "index.html"
# 元画像を保存先へコピーするときのファイル名（拡張子は元のまま）。output_dir ごと移動しても参照が切れないようにする
ORIGINAL_STEM = "original"
# 一覧HTMLの列の最小幅(px)。列は画面幅に合わせて等分され、画像は列幅に合わせて拡縮する（足りなければ横スクロール）
GALLERY_MIN_COLUMN_WIDTH = 280
# 一覧HTMLの1ページあたりの表示枚数の選択肢（先頭が既定）
GALLERY_PAGE_SIZES = [50, 100, 200]


def write_gallery_html(output_dir: Path, elapsed_ms: Dict[Path, float]) -> Path:
    """処理した画像ごとに 元画像と各結果画像を横に並べた一覧HTMLを output_dir 直下に書く。

    elapsed_ms: 画像ごとの保存先フォルダ -> 処理時間(ms)（元画像は save_dir/original.* にコピー済みのものを使う）。
    画像は埋め込まず相対パスで参照する（枚数が多くてもHTMLが小さく、画像を差し替えても反映される）。
    各画像の下にそのパスも表示する。
    表示は1ページ GALLERY_PAGE_SIZES から選んだ枚数ずつで、ページを変えると前のページの <img> を
    DOM ごと捨てて読み込み直す（大きな画像を全部メモリに載せ続けない）。
    """
    import json
    from os.path import relpath

    output_dir = Path(output_dir)

    rows = []
    for save_dir, ms in elapsed_ms.items():
        save_dir = Path(save_dir)
        original = next(iter(sorted(save_dir.glob(f"{ORIGINAL_STEM}.*"))), save_dir / ORIGINAL_STEM)
        images = [("original", original)] + [(label, save_dir / file) for label, file in GALLERY_IMAGES]
        rows.append({
            "name": save_dir.name,
            "ms": round(ms, 1),
            "images": [
                {"label": label, "path": Path(relpath(path.resolve(), output_dir.resolve())).as_posix(),
                 "exists": path.exists()}
                for label, path in images
            ],
        })

    # </script> などでデータがHTMLを壊さないよう "<" をエスケープする
    data = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    html = (GALLERY_TEMPLATE
            .replace("__DATA__", data)
            .replace("__PAGE_SIZES__", json.dumps(GALLERY_PAGE_SIZES))
            .replace("__COLUMNS__", str(len(GALLERY_IMAGES) + 1))
            .replace("__MIN_COLUMN_WIDTH__", str(GALLERY_MIN_COLUMN_WIDTH)))
    out_path = output_dir / GALLERY_FILE_NAME
    out_path.write_text(html, encoding="utf-8")
    return out_path


GALLERY_TEMPLATE = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>セグメンテーション結果一覧</title>
<style>
body { font-family: sans-serif; font-size: 18px; margin: 16px; }
button, select { font-size: 18px; }
.nav { margin: 8px 0; display: flex; gap: 12px; align-items: center; }
#wrap { overflow-x: auto; }
.grid { display: grid; grid-template-columns: repeat(__COLUMNS__, minmax(__MIN_COLUMN_WIDTH__px, 1fr)); gap: 8px; align-items: start; min-width: calc(__COLUMNS__ * __MIN_COLUMN_WIDTH__px); }
.head { font-weight: bold; font-size: 20px; padding: 6px 0; border-bottom: 2px solid #333; }
h2 { font-size: 22px; margin: 20px 0 6px; }
h2 .ms { font-weight: normal; color: #06c; margin-left: 12px; }
figure { margin: 0; min-width: 0; }
img { width: 100%; height: auto; display: block; }
code { font-size: 14px; color: #555; word-break: break-all; display: block; }
.missing { padding: 40px 0; text-align: center; color: #999; }
</style></head><body>
<h1>セグメンテーション結果一覧</h1>
<div class="nav" id="nav-top"></div>
<div id="wrap"><div class="grid head-row" id="heads"></div><div id="list"></div></div>
<div class="nav" id="nav-bottom"></div>
<script>
const ROWS = __DATA__;
const PAGE_SIZES = __PAGE_SIZES__;
let pageSize = PAGE_SIZES[0], page = 0;

function el(tag, props, children) {
  const e = Object.assign(document.createElement(tag), props || {});
  (children || []).forEach(c => e.append(c));
  return e;
}

function render() {
  document.getElementById("heads").replaceChildren(
    ...(ROWS[0] ? ROWS[0].images : []).map(im => el("div", {className: "head", textContent: im.label})));
  const pages = Math.max(1, Math.ceil(ROWS.length / pageSize));
  page = Math.min(page, pages - 1);
  const list = document.getElementById("list");
  // 前のページの img は src を空にしてから捨て、デコード済みの画像をブラウザに解放させる
  list.querySelectorAll("img").forEach(img => { img.removeAttribute("src"); });
  list.replaceChildren();
  ROWS.slice(page * pageSize, (page + 1) * pageSize).forEach(row => {
    const figs = row.images.map(im => el("figure", {}, [
      im.exists ? el("img", {src: im.path, loading: "lazy", decoding: "async", alt: im.label})
                : el("div", {className: "missing", textContent: "(なし)"}),
      el("code", {textContent: im.path}),
    ]));
    list.append(el("section", {}, [
      el("h2", {textContent: row.name}, [el("span", {className: "ms", textContent: `${row.ms.toFixed(1)} ms`})]),
      el("div", {className: "grid"}, figs)]));
  });
  ["nav-top", "nav-bottom"].forEach(id => {
    const prev = el("button", {textContent: "前へ", disabled: page === 0, onclick: () => go(page - 1)});
    const next = el("button", {textContent: "次へ", disabled: page >= pages - 1, onclick: () => go(page + 1)});
    const sel = el("select", {onchange: e => { pageSize = Number(e.target.value); page = 0; render(); }},
      PAGE_SIZES.map(n => el("option", {value: n, textContent: n + " 枚/ページ", selected: n === pageSize})));
    document.getElementById(id).replaceChildren(prev,
      el("span", {textContent: `${page + 1} / ${pages} ページ（全 ${ROWS.length} 枚）`}), next, sel);
  });
}

function go(p) { page = p; render(); window.scrollTo(0, 0); }
render();
</script></body></html>
"""
