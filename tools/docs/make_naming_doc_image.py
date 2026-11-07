"""説明資料（docs/detailed_segmentation.md「7. 最終の名前付け」）用の画像を作る: 精緻化の後の名前付けの、なし / ありを並べる。
入力・最終結果（なし / あり）を、画像ごとに 1 行にする。パネルの見出しにクラスの名前を並べる。

    python tools/docs/make_naming_doc_image.py --images PATH ... [--out docs/images/naming_types.jpg]

「なし」は SemanticRefiner.refine に patch_features を渡さずに再現する（= 粗い段階のクラスの名前のまま）。OpenVINO 版（configs/ov_config.yaml）で実行する。
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


def short(prompt: str) -> str:
    for head in ("a photo of a ", "a photo of an ", "a photo of "):
        if prompt.startswith(head):
            return prompt[len(head):-1]
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out", default="docs/images/naming_types.jpg")
    args = parser.parse_args()

    config = load_config()
    runner = SegmentationRunner(OVCLIPFeatureExtractor(config), OVEdgeTamRunner(config), config)
    refine = runner.refiner.refine
    prompts = [short(p) for p in runner.foreground_prompts + runner.background_prompts]
    colors: dict = {}

    def color_of(name: str) -> tuple:
        return colors.setdefault(name, SEMANTIC_COLORS[1 + len(colors) % (len(SEMANTIC_COLORS) - 1)])

    def without_naming(*a, **k):
        k.pop("patch_features", None)
        return refine(*a, **k)

    panels = []
    for path in args.images:
        image = load_image(Path(path))
        base = cv2.resize(image, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA)
        panels.append((f"{Path(path).stem}  入力", base))
        for label, enabled in (("名前付けなし", False), ("名前付けあり", True)):
            runner.refiner.refine = refine if enabled else without_naming
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.segment(image)
            final = base.copy()
            semantic = cv2.resize(result.semantic_map, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
            names = []
            for k, p in enumerate(result.prompt_indices, start=1):
                final = overlay(final, (semantic == k), color_of(prompts[p]), 0.6)
                names.append(prompts[p])
            panels.append((f"{label}\n" + "・".join(names), final))
    runner.refiner.refine = refine
    save_panels(Path(args.out), panels, ncols=3, size=3.0)


if __name__ == "__main__":
    main()
