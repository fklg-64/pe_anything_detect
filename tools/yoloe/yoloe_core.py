"""YOLOE（テキストプロンプトのゼロショット検出・セグメンテーション）の共通処理。

CLI（yoloe.py）と Web GUI（app.py）の両方から使う。
"""
import ast
import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOE

YOLOE_DIR = Path(__file__).resolve().parent
REPO_ROOT = YOLOE_DIR.parents[1]
# 重み（YOLOE のチェックポイントと、テキストエンコーダ mobileclip_blt.ts）の置き場所。git には入れない
WEIGHTS_DIR = YOLOE_DIR / "weights"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "yoloe-v8s-seg.pt"
DEFAULT_PROMPTS_FILE = YOLOE_DIR / "prompts" / "develop_prompts.json"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# クラスごとの色 (RGB)。先頭のプロンプトから順に使う（背景の色は含まない）
COLORS = np.array([
    [255, 59, 48], [52, 199, 89], [10, 132, 255], [255, 204, 0], [191, 90, 242], [90, 200, 250],
    [255, 149, 0], [255, 105, 180], [160, 82, 45], [48, 209, 88], [100, 210, 255], [142, 142, 147],
], dtype=np.uint8)


def class_color(index: int) -> tuple:
    return tuple(int(c) for c in COLORS[index % len(COLORS)])


# =============================================================================
# 画像・プロンプトの入出力
# =============================================================================

def develop_image_paths() -> List[str]:
    """develop.py の image_paths のうち、コメントアウトされていないものを返す。"""
    tree = ast.parse((REPO_ROOT / "develop.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "image_paths":
            return [ast.literal_eval(e) for e in node.value.elts]
    return []


def list_checkpoints() -> List[Path]:
    """weights/ にある YOLOE のチェックポイント（*.pt）。"""
    return sorted(WEIGHTS_DIR.glob("*.pt"))


def load_prompts(path: Path) -> dict:
    """{画像名(拡張子なし): [プロンプト, ...], "default": [...]} の JSON を読む。無ければ空。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_prompts(path: Path, key: str, names: Sequence[str]) -> None:
    """key のプロンプトだけを書き換えて保存する（他のキーはそのまま。途中で壊れないよう一時ファイル経由）。"""
    data = load_prompts(path)
    data[key] = [n for n in names if n]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def prompts_for_image(prompts: dict, stem: str, fallback: Sequence[str]) -> List[str]:
    return list(prompts.get(stem) or prompts.get("default") or fallback)


# =============================================================================
# 推論
# =============================================================================

class YoloeSegmenter:
    """プロンプトを切り替えながら、YOLOE でセグメンテーションする。"""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.checkpoint = str(checkpoint)
        self.device = device
        self.names: Optional[List[str]] = None
        self._lock = threading.RLock()
        with self._in_weights_dir():
            self.model = YOLOE(self.checkpoint)
            self.model.to(device)

    @contextlib.contextmanager
    def _in_weights_dir(self):
        """テキストエンコーダ(mobileclip_blt.ts)は、カレントディレクトリから探し、無ければダウンロードする。
        リポジトリを汚さないよう、weights/ に移動して実行する。"""
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        previous = os.getcwd()
        os.chdir(WEIGHTS_DIR)
        try:
            yield
        finally:
            os.chdir(previous)

    def set_names(self, names: Sequence[str]) -> None:
        """プロンプトが前回と同じなら、テキストの埋め込みは作り直さない。"""
        names = list(names)
        with self._lock:
            if names != self.names:
                with self._in_weights_dir():
                    self.model.set_classes(names, self.model.get_text_pe(names))
                self.names = names

    def predict(self, image: np.ndarray, names: Sequence[str], conf: float, iou: float, imgsz: int):
        """RGB 画像を推論する。(結果, 推論時間[ms]) を返す。"""
        with self._lock:
            self.set_names(names)
            start = time.perf_counter()
            result = self.model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            return result, (time.perf_counter() - start) * 1000


# =============================================================================
# 結果の整形
# =============================================================================

def extract_detections(result, height: int, width: int, polygon_epsilon: float = 0.7) -> List[dict]:
    """検出を確信度の高い順に返す。各検出に、クラス・確信度・ボックス・輪郭（穴を含む）を持たせる。

    輪郭は元画像の座標。rings は外周と穴の輪郭のリストで、塗りつぶしは even-odd で行う。
    """
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    masks = None if result.masks is None else result.masks.data.cpu().numpy() > 0.5
    for rank, i in enumerate(np.argsort(-confs)):
        rings = []
        if masks is not None:
            mh, mw = masks[i].shape
            contours, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                contour = cv2.approxPolyDP(contour, polygon_epsilon, True).reshape(-1, 2).astype(np.float32)
                contour[:, 0] *= width / mw
                contour[:, 1] *= height / mh
                rings.append(np.round(contour, 1).tolist())
        detections.append({
            "id": rank + 1,
            "class_index": int(classes[i]),
            "class": result.names[classes[i]],
            "confidence": round(float(confs[i]), 4),
            "box": [round(float(v), 1) for v in boxes[i]],
            "rings": rings,
        })
    return detections


def build_maps(result, height: int, width: int):
    """検出結果から、(インスタンスID画像, クラスID画像, 検出のリスト) を作る。

    マスクは確信度の低い順に塗るので、重なった画素は確信度の高い検出が残る。
    クラスIDは 1..N（プロンプトの順）、背景は 0。
    """
    instance_map = np.zeros((height, width), np.uint8)
    semantic_map = np.zeros((height, width), np.uint8)
    detections = []
    if result.masks is None:
        return instance_map, semantic_map, detections

    masks = result.masks.data.cpu().numpy() > 0.5
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    for rank, i in enumerate(np.argsort(confs), start=1):       # 低 -> 高
        mask = cv2.resize(masks[i].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        instance_map[mask] = min(rank, 255)
        semantic_map[mask] = classes[i] + 1
    for i in np.argsort(-confs):
        detections.append({"class": result.names[classes[i]], "confidence": round(float(confs[i]), 4),
                           "box_xyxy": [round(float(v), 1) for v in boxes[i]]})
    return instance_map, semantic_map, detections


def draw_overlay(image: np.ndarray, semantic_map: np.ndarray, detections: list, names: list) -> np.ndarray:
    """クラスごとに色を重ね、検出ごとにボックスとラベルを描く。"""
    colors = COLORS[(semantic_map.astype(int) - 1) % len(COLORS)]
    overlay = np.where((semantic_map > 0)[..., None], (0.6 * image + 0.4 * colors).astype(np.uint8), image)
    scale = max(image.shape[:2]) / 1000.0
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box_xyxy"])
        color = class_color(names.index(det["class"]))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, max(1, round(2 * scale)))
        cv2.putText(overlay, f"{det['class']} {det['confidence']:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, max(1, round(2 * scale)))
    return overlay


def save_outputs(out_dir: Path, image, instance_map, semantic_map, detections, names):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_overlay(image, semantic_map, detections, names)).save(out_dir / "overlay.png")
    index_image = Image.fromarray(semantic_map, mode="P")
    palette = np.vstack([[0, 255, 255], COLORS[np.arange(len(names)) % len(COLORS)]])  # 背景はシアン
    index_image.putpalette(palette.flatten().tolist())
    index_image.save(out_dir / "semantic_map.png")
    Image.fromarray(instance_map).save(out_dir / "instance_map.png")
    (out_dir / "detections.json").write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")
