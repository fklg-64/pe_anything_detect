#!/usr/bin/env python
"""C++ 版と Python 版（core/openvino + core/common）の結果を、同じ画像・同じ設定で比べる。

  1. Python 版 (SegmentationRunner + OpenVINO backend, configs/ov_config.yaml) で各画像を処理して結果を保持
  2. C++ 版 (pe_segmentation_sample --dump-npy) で同じ画像を処理して配列を書き出す
  3. 最終のセマンティックマップと、粗い推定の中間結果（前景確信度・クラスタ・ラベル）を比べて表にする

使い方（リポジトリのルートで）:
  python cppsrc/scripts/compare_with_python.py [--images-file LIST.txt | 画像 ...] [--bin cppsrc/build/bin/pe_segmentation_sample]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if "--system-openvino" in sys.argv:
    # core/openvino/runtime_utils.py は import すると /opt/intel/openvino を sys.path から外すので、その前にシステム版を読み込んでおく
    sys.path.insert(0, "/opt/intel/openvino/python")
    import openvino as ov  # noqa: E402,F401

import numpy as np  # noqa: E402


def make_system_openvino_extractor_class():
    """C++ 版と同じ OpenVINO（システム版 /opt/intel/openvino）で Python 版を動かすための OVCLIPFeatureExtractor。

    Python 版は pip 版 openvino + openvino_tokenizers の拡張を使うが、拡張は pip 版の openvino にしか読み込めない。
    ここでは openvino をシステム版にそろえ、トークナイザだけ HuggingFace のトークナイザ（変換元。pe_tokenizer.xml と同じ出力）で代用する。
    OpenVINO のバージョン差による数値の揺れ（~1e-6）と、C++ 実装の差を切り分けるための確認用。
    """
    import numpy as np
    from core.openvino.feature_extractor import OVCLIPFeatureExtractor
    from core.openvino.runtime_utils import make_compile_config, model_file_suffix
    from transformers import AutoTokenizer

    class SystemOpenVINOExtractor(OVCLIPFeatureExtractor):
        def load_ov_model(self):
            core = ov.Core()
            cfg = make_compile_config(self.num_threads)
            d = Path(self.model_dir)
            suffix = model_file_suffix(self.use_int8)
            visual = core.read_model(d / f"pe_visual_model{suffix}")
            for result in visual.get_results():
                if result.output(0).get_names() & {"cls2patch_attention"}:
                    visual.remove_result(result)
            self.patch_size = int(visual.rt_info["patch_size"].astype(int))
            self.feature_size = self.target_size // self.patch_size
            self.compiled_visual_model = core.compile_model(visual, self.device, cfg)
            self.compiled_text_model = core.compile_model(core.read_model(d / f"pe_text_model{suffix}"), self.device, cfg)
            self.compiled_patch_logits_model = core.compile_model(core.read_model(d / f"pe_patch_logits_model{suffix}"), self.device, cfg)
            self.hf_tokenizer = AutoTokenizer.from_pretrained("timm/PE-Core-T-16-384")
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self.hf_tokenizer.pad_token_id = 0

        def extract_text_features(self, prompts, normalize=True):
            ids = self.hf_tokenizer(prompts, padding="max_length", max_length=32, truncation=True, return_tensors="np")["input_ids"]
            out = self.compiled_text_model([ids.astype(np.int64)])[self.compiled_text_model.output(0)]
            return out / np.linalg.norm(out, axis=-1, keepdims=True) if normalize else out

    return SystemOpenVINOExtractor


def run_python(images, config_path, system_openvino=False):
    from core.common.segmentation_runner import SegmentationRunner
    from core.openvino.edgetam_runner import OVEdgeTamRunner
    from core.openvino.feature_extractor import OVCLIPFeatureExtractor
    from core.utils import load_config, load_image

    config = load_config(config_path)
    extractor_class = make_system_openvino_extractor_class() if system_openvino else OVCLIPFeatureExtractor
    runner = SegmentationRunner(extractor_class(config), OVEdgeTamRunner(config), config)
    results = []
    for path in images:
        r = runner.segment(load_image(path))
        results.append(r)
    return results


def run_cpp(binary, images, out_dir, extra_args):
    cmd = [str(binary), "--no-save", "--dump-npy", str(out_dir), *extra_args, *images]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise SystemExit("C++ 版の実行に失敗しました")
    return proc.stdout


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*")
    parser.add_argument("--images-file", help="画像のパスを 1 行に 1 つ書いたファイル")
    parser.add_argument("--bin", default=str(ROOT / "cppsrc/build/bin/pe_segmentation_sample"))
    parser.add_argument("--config", default=str(ROOT / "configs/ov_config.yaml"))
    parser.add_argument("--keep", help="C++ の出力 (.npy) をこのディレクトリに残す")
    parser.add_argument("--system-openvino", action="store_true",
                        help="Python 版も C++ 版と同じシステム版 OpenVINO で動かす（トークナイザは HF で代用）。OpenVINO のバージョン差による揺れを除いて比べる")
    parser.add_argument("--cpp-arg", action="append", default=[], help="C++ のサンプルに渡す追加の引数")
    args = parser.parse_args()

    images = list(args.images)
    if args.images_file:
        images += [l.strip() for l in Path(args.images_file).read_text().splitlines() if l.strip()]
    if not images:
        parser.error("画像を指定してください")

    print(f"Python 版を実行（{len(images)} 枚）...")
    py = run_python(images, args.config, args.system_openvino)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(args.keep or tmp)
        print("C++ 版を実行...")
        run_cpp(args.bin, images, out_dir, args.cpp_arg)

        header = f"{'画像':<34}{'最終 一致画素':>13}{'前景IoU':>9}{'クラス':>8}{'確信度 最大差':>14}{'クラスタ':>9}{'粗ラベル':>9}{'点 最大差':>10}"
        print("\n" + header)
        rows = []
        for i, (path, r) in enumerate(zip(images, py)):
            d = out_dir / f"{i:02d}"
            cs = np.load(d / "semantic_map.npy")
            same_pixels = float((cs == r.semantic_map).mean())
            fg_iou = iou(cs > 0, r.semantic_map > 0)
            cp = np.load(d / "prompt_indices.npy").tolist()
            classes = "同じ" if cp == list(r.prompt_indices) else f"{len(cp)}/{len(r.prompt_indices)}"
            fmap_diff = float(np.abs(np.load(d / "foreground_map.npy") - r.foreground_map).max())
            same_clusters = bool(np.array_equal(np.load(d / "cluster_labels.npy"), r.cluster_labels))
            same_labels = bool(np.array_equal(np.load(d / "semantic_labels.npy"), r.semantic_labels)
                               and np.array_equal(np.load(d / "foreground_mask.npy"), r.foreground_mask))
            pc = np.load(d / "point_coords.npy")
            point_diff = float(np.abs(pc - np.asarray(r.point_coords, dtype=np.float32)).max()) if pc.shape == np.asarray(r.point_coords).shape else float("nan")
            name = Path(path).parent.name + "/" + Path(path).name
            print(f"{name[-33:]:<34}{same_pixels * 100:>12.3f}%{fg_iou:>9.4f}{classes:>8}{fmap_diff:>14.2e}{str(same_clusters):>9}{str(same_labels):>9}{point_diff:>10.2e}")
            rows.append((same_pixels, fg_iou))
        arr = np.array(rows)
        print(f"\n最終マップの一致画素: 平均 {arr[:, 0].mean() * 100:.3f}% / 最小 {arr[:, 0].min() * 100:.3f}%   前景 IoU: 平均 {arr[:, 1].mean():.4f} / 最小 {arr[:, 1].min():.4f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
