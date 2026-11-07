"""説明資料（docs/coarse_segmentation.md §④）用の画像を作る: 前景のクラスの付け直しの、なし / ありを並べる。
入力・粗いラベル（なし / あり）・最終結果（なし / あり）を、画像ごとに 1 行にする。

    python tools/docs/make_relabel_doc_image.py [--images "PATH" ...] [--out docs/images/relabel_noodle.jpg] [--model-dir DIR]

「なし」は CoarseSegmenter.relabeler.relabel（ForegroundRelabeler）を何もしない関数に差し替えて再現する（= 粗い判定のクラスタが付けたラベルのまま）。OpenVINO 版（configs/ov_config.yaml）で実行する。
④ だけの効果を見るため、精緻化の後の名前付け（SemanticRefiner.refine の patch_features。docs/detailed_segmentation.md「7. 最終の名前付け」）は、なし・ありの両方で外す。
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.openvino.runtime_utils import make_compile_config  # noqa: F401  (import openvino より前に /opt/intel/openvino を外す)
from core.common.params import SEMANTIC_COLORS
from core.common.segmentation_runner import SegmentationRunner
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.utils import load_config, load_image
from tools.docs.make_doc_images import DISPLAY, overlay, save_panels

DATASET = Path("/home/llm/datasets/01_classification_sample/noodle")
# 部品の名前 -> 色（カップ麺で見分けやすいように固定する。他のクラスは SEMANTIC_COLORS から）
NAME_COLORS = {"noodle": (255, 60, 60), "package": (60, 200, 60), "white plastic lid": (60, 60, 255), "round white food container": (255, 210, 0)}


def short(prompt: str) -> str:
    return prompt[len("a photo of a "):-1] if prompt.startswith("a photo of a ") else prompt


def color_of(name: str, extra: dict) -> tuple:
    if name in NAME_COLORS:
        return NAME_COLORS[name]
    return extra.setdefault(name, SEMANTIC_COLORS[5 + len(extra) % (len(SEMANTIC_COLORS) - 5)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=[str(DATASET / "noodle (2).png")])
    parser.add_argument("--out", default="docs/images/relabel_noodle.jpg")
    parser.add_argument("--model-dir", default=None, help="OpenVINO モデルのフォルダ。既定は設定の model_dir")
    args = parser.parse_args()

    config = load_config()
    if args.model_dir:
        config["model_dir"] = args.model_dir
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    coarse = runner.coarse
    relabeler = coarse.relabeler
    relabel = relabeler.relabel
    refine = runner.refiner.refine

    def refine_without_naming(*a, **k):  # 最後の名前付けを外す
        k.pop("patch_features", None)
        return refine(*a, **k)

    runner.refiner.refine = refine_without_naming
    prompts = [short(p) for p in runner.foreground_prompts + runner.background_prompts]
    extra: dict = {}

    panels = []
    for path in args.images:
        image = load_image(Path(path))
        base = cv2.resize(image, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)
        panels.append((f"{Path(path).stem}  入力", base))
        for label, enabled in (("付け直しなし", False), ("付け直しあり", True)):
            relabeler.relabel = relabel if enabled else (lambda prompt_index, *a, **k: prompt_index)
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.segment(image)
            grid = result.foreground_mask.shape
            patch_prompt = np.asarray(result.cluster_index_list)[result.cluster_labels].reshape(grid)
            coarse_rgb = np.zeros((*grid, 3), np.uint8)
            for p in np.unique(patch_prompt[result.foreground_mask > 0]):
                coarse_rgb[(patch_prompt == p) & (result.foreground_mask > 0)] = color_of(prompts[p], extra)
            coarse_panel = cv2.resize(coarse_rgb, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
            coarse_panel = np.where(coarse_panel.any(-1, keepdims=True), (0.35 * base + 0.65 * coarse_panel).astype(np.uint8), base)
            panels.append((f"粗いラベル  {label}", coarse_panel))
            final = base.copy()
            semantic = cv2.resize(result.semantic_map, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
            for k, p in enumerate(result.prompt_indices, start=1):
                final = overlay(final, (semantic == k), color_of(prompts[p], extra), 0.55)
            panels.append((f"最終結果  {label}", final))
    relabeler.relabel = relabel
    # 列: 入力 / 粗い（なし）/ 最終（なし）/ 粗い（あり）/ 最終（あり）の順に並んでいる
    save_panels(Path(args.out), panels, ncols=5, size=2.4)


if __name__ == "__main__":
    main()
