#!/usr/bin/env python
"""トークナイザの確認用データを作る: OpenVINO の pe_tokenizer.xml（Python 版が使っているもの）の出力を正解にして、
cppsrc/tests/data/tokenizer_cases.json に書く。C++ の ClipTokenizer（tests/test_tokenizer.cpp）がこれと一致することを確認する。

使い方: python cppsrc/scripts/gen_tokenizer_cases.py [--model-dir weights/openvino_models_l6]
"""
import argparse
import json
import random
import sys
from pathlib import Path

# pip 版 openvino と衝突しないよう、システム版を sys.path から外す（convert_openvino.py と同じ）
sys.path = [p for p in sys.path if not p.startswith("/opt/intel/openvino")]

import openvino as ov  # noqa: E402
import openvino_tokenizers  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def prompt_texts():
    """リポジトリの prompts/ の yaml にある文（コメントを除く）。"""
    import yaml

    texts = []
    for path in sorted((ROOT / "prompts").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("custom", "base", "color", "shape"):
            texts += [t for t in data.get(key, []) if isinstance(t, str)]
    return texts


def random_texts(n, seed):
    rng = random.Random(seed)
    words = ["a", "photo", "of", "metal", "screw", "nut", "plastic", "part", "don't", "it's", "we're", "they've", "I'm", "you'll", "he'd",
             "3.14", "x2y", "A/B", "(test)", "end.", "foo-bar", "NASA", "İstanbul", "naïve", "café", "Ünïcödé", "ΑΒΓ", "Привет", "日本語", "カメラ",
             "100%", "$5", "a_b", "50/50", "...", "!?", "'quoted'", '"double"', "tab\tsep", "new\nline", "emoji😀", "x" * 40, "ß", "ÿ"]
    texts = []
    for _ in range(n):
        k = rng.randint(1, 14)
        sep = rng.choice([" ", " ", " ", "  ", "\t", ""])
        texts.append(sep.join(rng.choice(words) for _ in range(k)))
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"()[]{}-_/\\|@#$%^&*+=~`<>"
    for _ in range(n):
        texts.append("".join(rng.choice(chars) for _ in range(rng.randint(0, 60))))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(ROOT / "weights/openvino_models_l6"))
    parser.add_argument("--out", default=str(ROOT / "cppsrc/tests/data/tokenizer_cases.json"))
    args = parser.parse_args()

    core = ov.Core()
    core.add_extension(openvino_tokenizers._ext_path)
    tokenizer = core.compile_model(core.read_model(Path(args.model_dir) / "pe_tokenizer.xml"), "CPU")

    texts = prompt_texts() + random_texts(300, 0)
    texts += ["", " ", "a", "A PHOTO OF A SCREW.", " ".join(["a"] * 40), "word " * 50]
    cases = []
    for text in texts:
        ids = tokenizer([text])[0][0].tolist()
        cases.append({"text": text, "ids": ids})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(cases)} 件を書きました: {args.out}")


if __name__ == "__main__":
    main()
