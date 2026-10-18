"""YOLOE（テキストプロンプトのゼロショット検出・セグメンテーション）を、画像とプロンプトを指定して実行する（CLI）。

develop.py と同じ画像で、現行のパイプライン（CLIP + KMeans + EdgeTAM）と比べる用途。
GUI で試すときは app.py を使う（README.md 参照）。

    # 1枚を、プロンプトを指定して実行
    python tools/yoloe/yoloe.py --source "/home/llm/datasets/02_object_detection/coin/001_crop.png" --names coin

    # develop.py の image_paths 全部を、画像ごとのプロンプト（JSON）で実行
    python tools/yoloe/yoloe.py --develop --names-json tools/yoloe/prompts/develop_prompts.json --conf 0.05

出力（--output-dir、既定は outputs/yoloe）:
    <画像名>/overlay.png       マスクとラベルを重ねた画像
    <画像名>/semantic_map.png  クラスID画像（0=背景、1..N=プロンプトの順。パレット付きPNG）
    <画像名>/instance_map.png  インスタンスID画像（0=背景、1..N=検出の順）
    <画像名>/detections.json   検出ごとのクラス名・確信度・ボックス
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from yoloe_core import (
    DEFAULT_CHECKPOINT, REPO_ROOT, YoloeSegmenter, build_maps, develop_image_paths,
    load_prompts, prompts_for_image, save_outputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="*", default=[], help="画像のパス（複数可）")
    parser.add_argument("--develop", action="store_true", help="develop.py の image_paths も対象にする")
    parser.add_argument("--names", nargs="+", default=["object"], help="プロンプト（クラス名）。英語の名詞句")
    parser.add_argument("--names-json", default=None, help="画像ごとのプロンプトを書いた JSON（キー: 画像名(拡張子なし)、default）")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="確信度の下限")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS の IoU")
    parser.add_argument("--imgsz", type=int, default=640, help="推論の画像サイズ")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "yoloe"))
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.source] + ([Path(p).resolve() for p in develop_image_paths()] if args.develop else [])
    if not paths:
        raise SystemExit("--source か --develop で画像を指定してください")
    prompts = load_prompts(Path(args.names_json).resolve()) if args.names_json else {}

    segmenter = YoloeSegmenter(args.checkpoint, args.device)
    times = []
    for path in paths:
        if not path.exists():
            print(f"存在しません: {path}")
            continue
        names = prompts_for_image(prompts, path.stem, args.names)
        image = np.array(Image.open(path).convert("RGB"))
        result, ms = segmenter.predict(image, names, args.conf, args.iou, args.imgsz)
        instance_map, semantic_map, detections = build_maps(result, *image.shape[:2])
        save_outputs(Path(args.output_dir).resolve() / path.stem, image, instance_map, semantic_map, detections, names)
        times.append(ms)
        summary = ", ".join(f"{d['class']}({d['confidence']:.2f})" for d in detections[:8]) or "検出なし"
        print(f"{path.name:<40} {ms:6.1f}ms  検出{len(detections):3d}  {summary}")
    if times:
        print(f"平均 {np.mean(times):.1f}ms（最初の1枚は初期化を含むので除くと {np.mean(times[1:]) if len(times) > 1 else times[0]:.1f}ms）"
              f"  出力: {args.output_dir}")


if __name__ == "__main__":
    main()
