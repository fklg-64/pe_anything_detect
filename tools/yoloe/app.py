"""YOLOE のテキストプロンプト検出を、ブラウザで試す Web GUI（FastAPI）。

    python tools/yoloe/app.py                 # http://127.0.0.1:8000 を開く
    python tools/yoloe/app.py --port 8080
    python tools/yoloe/app.py --host 0.0.0.0  # Docker・別マシンのブラウザから開く場合（画像のパス指定が外から使える点に注意）

流れ: モデル読み込み -> 画像読み込み -> プロンプト設定（JSON を読み込み、変更すると書き換える）-> 検査実行
      -> 画像上に結果を重ねて表示（ホイールで拡大縮小、ドラッグで移動）

画面は web/index.html（単一ファイル。ビルド不要）。ここはそのAPIサーバー。
"""
import argparse
import io
import threading
import time
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from yoloe_core import (
    DEFAULT_CHECKPOINT, DEFAULT_PROMPTS_FILE, IMAGE_SUFFIXES, REPO_ROOT, WEIGHTS_DIR, YOLOE_DIR,
    YoloeSegmenter, build_maps, class_color, develop_image_paths, extract_detections,
    list_checkpoints, load_prompts, save_outputs, save_prompts,
)

WEB_DIR = YOLOE_DIR / "web"
EXPORT_DIR = REPO_ROOT / "outputs" / "yoloe_gui"
MAX_IMAGES = 16          # メモリに保持する画像の数（古いものから捨てる）


# =============================================================================
# サーバーの状態
# =============================================================================

class Session:
    """モデルと、読み込み済みの画像・直近の推論結果。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.segmenter: Optional[YoloeSegmenter] = None
        self.model = {"state": "idle", "checkpoint": "", "device": "cpu", "load_ms": None, "message": ""}
        self.images: "OrderedDict[str, dict]" = OrderedDict()
        self.last: Dict[str, dict] = {}        # 画像ID -> 直近の推論結果（保存用）
        self._counter = 0

    def load_model(self, checkpoint: str, device: str) -> None:
        """別スレッドで読み込む（画面にスピナーを出すため）。"""
        self.model = {"state": "loading", "checkpoint": checkpoint, "device": device, "load_ms": None, "message": ""}

        def work():
            start = time.perf_counter()
            try:
                segmenter = YoloeSegmenter(checkpoint, device)
                segmenter.set_names(["object"])      # テキストエンコーダも読み込んでおく（初回の推論を速くする）
                with self.lock:
                    self.segmenter = segmenter
                self.model = {"state": "ready", "checkpoint": checkpoint, "device": device,
                              "load_ms": round((time.perf_counter() - start) * 1000), "message": ""}
            except Exception as error:      # 読み込みの失敗は画面に出す
                self.model = {"state": "error", "checkpoint": checkpoint, "device": device, "load_ms": None,
                              "message": f"{type(error).__name__}: {error}"}

        threading.Thread(target=work, daemon=True).start()

    def add_image(self, name: str, data: bytes) -> dict:
        """画像を登録する。EXIF の向きを反映したうえで、ブラウザに返す用のバイト列も作る。"""
        try:
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        except Exception:
            raise HTTPException(400, f"画像として読み込めません: {name}")
        rgb = np.array(pil.convert("RGB"))
        buffer = io.BytesIO()
        # 写真(JPEG)と、大きな画像（400万画素超）は JPEG で返す。小さな PNG はそのまま PNG（無劣化）で返す
        if Path(name).suffix.lower() in (".jpg", ".jpeg") or rgb.shape[0] * rgb.shape[1] > 4_000_000:
            Image.fromarray(rgb).save(buffer, "JPEG", quality=95)
            mime = "image/jpeg"
        else:
            Image.fromarray(rgb).save(buffer, "PNG", compress_level=1)
            mime = "image/png"
        with self.lock:
            self._counter += 1
            image_id = f"img{self._counter}"
            self.images[image_id] = {"name": name, "stem": Path(name).stem, "rgb": rgb, "bytes": buffer.getvalue(), "mime": mime}
            while len(self.images) > MAX_IMAGES:
                old, _ = self.images.popitem(last=False)
                self.last.pop(old, None)
        return {"id": image_id, "name": name, "stem": Path(name).stem, "width": rgb.shape[1], "height": rgb.shape[0]}

    def image(self, image_id: str) -> dict:
        with self.lock:
            if image_id not in self.images:
                raise HTTPException(404, "画像が見つかりません。読み込み直してください")
            return self.images[image_id]


session = Session()
app = FastAPI(title="YOLOE GUI")


# =============================================================================
# API
# =============================================================================

class ModelRequest(BaseModel):
    checkpoint: str
    device: str = "cpu"


class PathRequest(BaseModel):
    path: str


class PromptSaveRequest(BaseModel):
    file: str
    key: str
    names: List[str]


class PredictRequest(BaseModel):
    image_id: str
    names: List[str]
    conf: float = 0.1
    iou: float = 0.7
    imgsz: int = 640


class ExportRequest(BaseModel):
    image_id: str


def checkpoint_path(name: str) -> str:
    """weights/ 内のファイル名、または絶対パスのチェックポイントを解決する。"""
    path = Path(name)
    path = path if path.is_absolute() else WEIGHTS_DIR / path
    if path.suffix != ".pt" or not path.exists():
        raise HTTPException(400, f"チェックポイントが見つかりません: {name}")
    return str(path)


def resolve_prompts_file(file: str) -> Path:
    path = Path(file)
    path = path if path.is_absolute() else REPO_ROOT / path
    if path.suffix != ".json":
        raise HTTPException(400, "プロンプトのファイルは .json を指定してください")
    return path


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config")
def get_config():
    develop = [{"name": Path(p).name, "path": p} for p in develop_image_paths() if Path(p).exists()]
    return {
        "checkpoints": [p.name for p in list_checkpoints()],
        "default_checkpoint": session.model["checkpoint"] and Path(session.model["checkpoint"]).name or DEFAULT_CHECKPOINT.name,
        "device": session.model["device"],
        "prompts_file": str(DEFAULT_PROMPTS_FILE.relative_to(REPO_ROOT)),
        "develop_images": develop,
        "model": session.model,
    }


@app.get("/api/model")
def get_model():
    return session.model


@app.post("/api/model")
def post_model(request: ModelRequest):
    session.load_model(checkpoint_path(request.checkpoint), request.device)
    return session.model


@app.post("/api/image/upload")
async def upload_image(file: UploadFile = File(...)):
    return session.add_image(file.filename or "upload.png", await file.read())


@app.post("/api/image/path")
def open_image(request: PathRequest):
    path = Path(request.path).expanduser()
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        raise HTTPException(400, f"画像ファイルが見つかりません: {request.path}")
    return session.add_image(path.name, path.read_bytes())


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    image = session.image(image_id)
    return Response(image["bytes"], media_type=image["mime"], headers={"Cache-Control": "no-store"})


@app.get("/api/prompts")
def get_prompts(file: str):
    path = resolve_prompts_file(file)
    return {"file": str(path), "exists": path.exists(), "data": load_prompts(path)}


@app.put("/api/prompts")
def put_prompts(request: PromptSaveRequest):
    path = resolve_prompts_file(request.file)
    if not request.key.strip():
        raise HTTPException(400, "保存先のキーが空です")
    save_prompts(path, request.key.strip(), [n.strip() for n in request.names])
    return {"ok": True, "file": str(path)}


@app.post("/api/predict")
def predict(request: PredictRequest):
    if session.model["state"] != "ready" or session.segmenter is None:
        raise HTTPException(409, "モデルが読み込まれていません")
    names = [n.strip() for n in request.names if n.strip()]
    if not names:
        raise HTTPException(400, "プロンプトを1つ以上入力してください")
    image = session.image(request.image_id)
    rgb = image["rgb"]
    result, ms = session.segmenter.predict(rgb, names, request.conf, request.iou, request.imgsz)
    height, width = rgb.shape[:2]
    detections = extract_detections(result, height, width)
    with session.lock:
        session.last[request.image_id] = {"result": result, "names": names}
    return {
        "time_ms": round(ms, 1), "width": width, "height": height,
        "names": names, "colors": ["#%02x%02x%02x" % class_color(i) for i in range(len(names))],
        "detections": detections,
    }


@app.post("/api/export")
def export(request: ExportRequest):
    """直近の推論結果を outputs/yoloe_gui/<画像名>/ に保存する（overlay / semantic_map / instance_map / detections.json）。"""
    image = session.image(request.image_id)
    with session.lock:
        last = session.last.get(request.image_id)
    if last is None:
        raise HTTPException(409, "この画像はまだ検査していません")
    rgb = image["rgb"]
    instance_map, semantic_map, detections = build_maps(last["result"], *rgb.shape[:2])
    out_dir = EXPORT_DIR / image["stem"]
    save_outputs(out_dir, rgb, instance_map, semantic_map, detections, last["names"])
    return {"dir": str(out_dir.relative_to(REPO_ROOT))}


# =============================================================================
# 起動
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="起動時に読み込むモデル")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print("注意: 外部から接続できる設定です。画像のパス指定 API で、このマシン上の画像を読み出せます。")
    if Path(args.checkpoint).exists():
        session.load_model(args.checkpoint, args.device)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"YOLOE GUI: {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
