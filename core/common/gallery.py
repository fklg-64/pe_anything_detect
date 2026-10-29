"""処理した画像の結果を横に並べて見る、一覧HTML（output_dir/index.html）の生成。"""

import json
from os.path import relpath
from pathlib import Path
from typing import Dict

# HTML/JS のひな形（__DATA__ などの印を write_gallery_html が埋める）
GALLERY_TEMPLATE = (Path(__file__).parent / "gallery_template.html").read_text(encoding="utf-8")

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
