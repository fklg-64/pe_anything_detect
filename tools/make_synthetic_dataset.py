"""物体の切り抜きを背景に貼り付けて、正解付きの評価データセットを作る。

生成AIで作った「部品（単体の物体）」と「背景」を、個数・向き・大きさ・影を変えて合成する。
貼り付けた位置がそのまま正解になるので、マスク・有無・個数の正解を自動で出力できる。

入力:
    --parts DIR        部品。次のどちらかの形式（混在可）
                         - RGBA の PNG（アルファが物体領域）
                         - <name>.png と <name>_mask.png（マスクは0/255）
    --backgrounds DIR  背景の画像。サブディレクトリ名を背景の種類（plain / textured / pattern など）として
                       メタデータに記録する（直下に置いた場合は "default"）
出力 (--out):
    images/0000.png    合成画像
    masks/0000.png     インスタンスID画像（uint8。0=背景、1..N=物体）
    meta.jsonl         1行1画像: 個数、背景の種類・ファイル名、使った部品、物体と背景の輝度差、接触の有無

    python tools/make_synthetic_dataset.py --parts ./assets/parts --backgrounds ./assets/backgrounds \
        --out ./datasets/synthetic_v1 --num 200 --count 1 5 --seed 0
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_parts(parts_dir: Path):
    """部品を (名前, RGB, マスク) のリストで返す。"""
    parts = []
    for path in sorted(parts_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem.endswith("_mask"):
            continue
        image = Image.open(path)
        mask_path = path.with_name(f"{path.stem}_mask.png")
        if mask_path.exists():
            rgb = np.array(image.convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L")) > 127
        elif image.mode == "RGBA":
            array = np.array(image)
            rgb, mask = array[..., :3], array[..., 3] > 127
        else:
            continue
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        parts.append((path.stem, rgb[box], mask[box]))
    return parts


def load_backgrounds(background_dir: Path):
    """背景を (種類, ファイル名, RGB) のリストで返す。"""
    backgrounds = []
    for path in sorted(background_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            kind = path.parent.name if path.parent != background_dir else "default"
            backgrounds.append((kind, path.name, np.array(Image.open(path).convert("RGB"))))
    return backgrounds


def fit_background(image: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """背景を、短辺が size になるよう拡大縮小し、ランダムに size×size で切り出す。"""
    height, width = image.shape[:2]
    scale = size / min(height, width)
    image = cv2.resize(image, (max(size, round(width * scale)), max(size, round(height * scale))))
    y = rng.integers(0, image.shape[0] - size + 1)
    x = rng.integers(0, image.shape[1] - size + 1)
    return image[y:y + size, x:x + size].copy()


def transform_part(rgb, mask, target_size: int, rng: np.random.Generator):
    """部品を、長辺 target_size に拡大縮小し、ランダムに回転・反転する。"""
    if rng.random() < 0.5:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    scale = target_size / max(rgb.shape[:2])
    rgb = cv2.resize(np.ascontiguousarray(rgb), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8) * 255, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    # 回転しても欠けないよう、対角線の長さまで余白を付ける
    diag = int(np.ceil(np.hypot(*rgb.shape[:2])))
    pad_y, pad_x = (diag - rgb.shape[0]) // 2, (diag - rgb.shape[1]) // 2
    rgb = cv2.copyMakeBorder(rgb, pad_y, diag - rgb.shape[0] - pad_y, pad_x, diag - rgb.shape[1] - pad_x, cv2.BORDER_REPLICATE)
    mask = cv2.copyMakeBorder(mask, pad_y, diag - mask.shape[0] - pad_y, pad_x, diag - mask.shape[1] - pad_x, cv2.BORDER_CONSTANT)
    matrix = cv2.getRotationMatrix2D((diag / 2, diag / 2), rng.uniform(0, 360), 1.0)
    rgb = cv2.warpAffine(rgb, matrix, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, matrix, (diag, diag), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(mask > 127)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return rgb[box], mask[box]


def composite(canvas, instance_map, rgb, alpha, top, left, instance_id, shadow: bool):
    """部品を貼り付ける。影は、マスクを少しずらしてぼかしたものを暗く乗せる。"""
    h, w = alpha.shape
    region = (slice(top, top + h), slice(left, left + w))
    soft = cv2.GaussianBlur(alpha, (0, 0), 0.8).astype(np.float32)[..., None] / 255.0
    if shadow:
        offset = max(2, round(0.05 * max(h, w)))
        shadow_alpha = np.zeros((h + offset * 2, w + offset * 2), np.float32)
        shadow_alpha[offset:offset + h, offset:offset + w] = alpha / 255.0
        shadow_alpha = cv2.GaussianBlur(shadow_alpha, (0, 0), offset)
        sh, sw = canvas.shape[:2]
        y0, x0 = max(top - offset // 2, 0), max(left - offset // 2, 0)
        y1, x1 = min(y0 + shadow_alpha.shape[0], sh), min(x0 + shadow_alpha.shape[1], sw)
        patch = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = (patch * (1 - 0.4 * shadow_alpha[:y1 - y0, :x1 - x0, None])).astype(np.uint8)
    canvas[region] = (canvas[region] * (1 - soft) + rgb * soft).astype(np.uint8)
    instance_map[region][alpha > 127] = instance_id


def make_sample(parts, background, size, count_range, scale_range, overlap_prob, shadow, rng):
    kind, bg_name, bg_image = background
    canvas = fit_background(bg_image, size, rng)
    instance_map = np.zeros((size, size), np.uint8)
    used = []
    target = int(rng.integers(count_range[0], count_range[1] + 1))
    touching = False
    for _ in range(target):
        name, rgb, mask = parts[rng.integers(len(parts))]
        allow_overlap = rng.random() < overlap_prob
        for _try in range(30):
            part_rgb, part_alpha = transform_part(rgb, mask, int(size * rng.uniform(*scale_range)), rng)
            h, w = part_alpha.shape
            if h >= size or w >= size:
                continue
            top, left = int(rng.integers(0, size - h)), int(rng.integers(0, size - w))
            footprint = instance_map[top:top + h, left:left + w][part_alpha > 127]
            if allow_overlap or not footprint.any():
                break
        else:
            continue  # 置けなかった物体は数えない（個数は実際に置いた数）
        if footprint.any():
            touching = True
        composite(canvas, instance_map, part_rgb, part_alpha, top, left, len(used) + 1, shadow)
        used.append(name)
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inside = instance_map > 0
    contrast = float(abs(gray[inside].mean() - gray[~inside].mean())) if inside.any() and (~inside).any() else 0.0
    return canvas, instance_map, dict(count=len(used), bg_kind=kind, bg=bg_name, parts=used,
                                      contrast=round(contrast, 1), overlap=touching)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True)
    parser.add_argument("--backgrounds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--count", type=int, nargs=2, default=[1, 5], metavar=("MIN", "MAX"), help="1枚あたりの物体数の範囲")
    parser.add_argument("--size", type=int, default=1024, help="出力画像の1辺(px)")
    parser.add_argument("--scale", type=float, nargs=2, default=[0.12, 0.3], metavar=("MIN", "MAX"), help="物体の長辺 / 画像の1辺")
    parser.add_argument("--overlap-prob", type=float, default=0.0, help="物体どうしを重ねて置く確率")
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    parts, backgrounds = load_parts(Path(args.parts)), load_backgrounds(Path(args.backgrounds))
    if not parts or not backgrounds:
        raise SystemExit(f"部品({len(parts)})または背景({len(backgrounds)})が見つかりません")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    with open(out / "meta.jsonl", "w", encoding="utf-8") as meta_file:
        for i in range(args.num):
            image, instance_map, meta = make_sample(
                parts, backgrounds[rng.integers(len(backgrounds))], args.size, args.count, args.scale,
                args.overlap_prob, not args.no_shadow, rng,
            )
            Image.fromarray(image).save(out / "images" / f"{i:04d}.png")
            Image.fromarray(instance_map).save(out / "masks" / f"{i:04d}.png")
            meta_file.write(json.dumps({"id": f"{i:04d}", **meta}, ensure_ascii=False) + "\n")
    print(f"{args.num}枚を {out} に保存しました（部品 {len(parts)} 種、背景 {len(backgrounds)} 枚）")
