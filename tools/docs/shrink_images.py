"""docs/images/ の画像を、git に載せられる大きさに縮める（幅の上限 + JPEG）。

    python tools/docs/shrink_images.py [--dir docs/images] [--docs docs] [--max-width 1200] [--quality 80] [--keep]

PNG を JPEG（幅が上限を超えるときは縮小）に変換し、docs/*.md の参照（images/xxx.png）を .jpg に書き換える。
変換後の PNG は削除する（--keep で残す）。何度実行しても同じ結果になる（JPEG は対象外）。
make_doc_images.py も同じ設定（`save_jpeg`）で最初から JPEG で保存するので、通常は実行しなくてよい。
"""
import argparse
import re
from pathlib import Path
from typing import Union

from PIL import Image

MAX_WIDTH = 1200   # 資料で文字が読める幅。これを超える画像は縮小する
JPEG_QUALITY = 80  # 80 で、図の文字・線が劣化して見えない（85 だと約 1.2 倍の容量）


def save_jpeg(image: Image.Image, dest: Union[str, Path], max_width: int = MAX_WIDTH, quality: int = JPEG_QUALITY) -> int:
    """画像を幅の上限まで縮めて JPEG で保存する。保存したバイト数を返す。"""
    image = image.convert("RGB")
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.LANCZOS)
    dest = Path(dest)
    image.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)
    return dest.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="docs/images")
    parser.add_argument("--docs", default="docs", help="参照を書き換える Markdown のフォルダ（直下の *.md）")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH)
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--keep", action="store_true", help="変換後も PNG を残す")
    args = parser.parse_args()

    before = after = 0
    converted = []
    for png in sorted(Path(args.dir).glob("*.png")):
        before += png.stat().st_size
        with Image.open(png) as image:
            after += save_jpeg(image, png.with_suffix(".jpg"), args.max_width, args.quality)
        converted.append(png.name)
        if not args.keep:
            png.unlink()
    print(f"{len(converted)} 枚を変換: {before / 1e6:.2f} MB -> {after / 1e6:.2f} MB")

    # Markdown の参照を .jpg に書き換える
    names = {re.escape(Path(n).stem) for n in converted}
    if names:
        pattern = re.compile(r"(images/(?:%s))\.png" % "|".join(sorted(names)))
        for md in sorted(Path(args.docs).glob("*.md")):
            text = md.read_text(encoding="utf-8")
            new = pattern.sub(r"\1.jpg", text)
            if new != text:
                md.write_text(new, encoding="utf-8")
                print("参照を更新:", md)


if __name__ == "__main__":
    main()
