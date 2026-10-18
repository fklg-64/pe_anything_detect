import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from typing import Optional, Tuple, List, Union

from core.utils import load_config, load_image, preprocess_image
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.torch.edgetam_runner import TorchEdgeTamBackend

class SAMInteractiveApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("SAM Prompt Segmentation GUI")
        self.geometry("1100x750")

        # ---------------------------------------------------------
        # 1. 内部データ・プロンプト状態の保持
        # ---------------------------------------------------------
        self.cv_image = None  # 元画像 (BGR / numpy array)
        self.points = []  # [[x1, y1], [x2, y2], ...] (元画像座標系)
        self.point_labels = (
            []
        )  # [1, 0, ...] (1: Positive/Foreground, 0: Negative/Background)
        self.latest_logits = None  # 前回の推論結果 logits (入力用)
        self.latest_mask = None  # 前回の推論結果 Mask (HxW bool/uint8)

        # ズーム・パン（移動）の状態変数
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0

        # ---------------------------------------------------------
        # 2. UI レイアウト構成
        # ---------------------------------------------------------
        # 左側: コントロールパネル
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_load = ctk.CTkButton(
            self.sidebar,
            text="Load Image",
            command=self.load_image,
            corner_radius=0,
            font=("Sans", 13, "bold"),
        )
        self.btn_load.pack(padx=10, pady=15)

        self.btn_reset = ctk.CTkButton(
            self.sidebar,
            text="Reset Prompts",
            command=self.reset_prompts,
            fg_color="#D32F2F",
            hover_color="#9A0007",
            corner_radius=0,
            font=("Sans", 13, "bold"),
        )
        self.btn_reset.pack(padx=10, pady=10)

        # 説明文
        self.info_label = ctk.CTkLabel(
            self.sidebar,
            text="[ How to use ]\n"
            "- Left Click : Positive Point (+)\n"
            "- Right Click: Negative Point (-)\n"
            "- Scroll Wheel: Zoom In / Out\n"
            "- Wheel Drag: Pan Image\n"
            "- Wheel Double Click: Reset View\n\n"
            "[ Stored prompts ]",
            justify="left",
            anchor="w",
        )
        self.info_label.pack(padx=10, pady=(20, 5), anchor="w")

        self.lbl_status = ctk.CTkLabel(
            self.sidebar,
            text="points: 0\nLogits: None",
            justify="left",
            anchor="w",
            text_color="gray",
        )
        self.lbl_status.pack(padx=10, pady=5, anchor="w")

        # 右側: 画像描画エリア (Canvas)
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(
            side="right", expand=True, fill="both", padx=10, pady=10
        )

        self.canvas = tk.Canvas(
            self.canvas_frame, bg="#1E1E1E", highlightthickness=0
        )
        self.canvas.pack(expand=True, fill="both")

        # Canvasイベントバインド
        self.canvas.bind("<Button-1>", self.on_left_click)  # 正ポイント
        self.canvas.bind(
            "<Button-3>", self.on_right_click
        )  # 負ポイント (Macは <Button-2>)
        self.canvas.bind("<Configure>", self.on_resize)

        # ズーム・パンイベントバインド（ホイール操作）
        self.canvas.bind("<MouseWheel>", self.on_zoom)  # Windows / macOS
        self.canvas.bind("<Button-4>", self.on_zoom)  # Linux Wheel Up
        self.canvas.bind("<Button-5>", self.on_zoom)  # Linux Wheel Down

        self.canvas.bind(
            "<ButtonPress-2>", self.on_pan_start
        )  # ホイール長押し開始
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)  # ホイールドラッグ中
        self.canvas.bind(
            "<Double-Button-2>", lambda e: self.reset_view()
        )  # ホイールダブルクリック

        self.tk_img = None

        self.load_model()

    # モデルの読み込み
    def load_model(self):
        mode: str = "openvino"
        if mode == "openvino":
            config = load_config(f"./configs/ov_config.yaml")
            self.edgetam = OVEdgeTamRunner(config)
        elif mode == "pytorch":
            config = load_config(f"./configs/torch_config.yaml")
            self.edgetam = TorchEdgeTamBackend(config)

    def _predict_edgetam_encoder(self, image_raw: np.ndarray) -> Optional[Tuple]:
        return self.edgetam.encode(
            preprocess_image(
                image_raw,
                self.edgetam.target_size,
                self.edgetam.mean, self.edgetam.std
            )
        )

    def _predict_edgetam_decoder(
        self,
        point_coords: List[List[float]],
        point_labels: List[int],
        feats: Optional[Tuple],
        height: int,
        width: int,
        mask_input: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mask_input is None:
            mask_logits_tensor = np.array([0], dtype=np.float32)
        else:
            mask_logits_tensor = mask_input

        mask_input_tensor, point_coords_norm, point_labels_norm, has_mask = self.edgetam.preprocess(
            (height, width),
            point_coords=np.array(point_coords, dtype=np.float32),
            point_labels=np.array(point_labels, dtype=np.int32),
            mask_logits=mask_logits_tensor,
        )

        high_res_feats0, high_res_feats1, image_embed = feats
        low_res_masks, iou_predictions = self.edgetam.decode(
            high_res_feats0, high_res_feats1, image_embed,
            point_coords=point_coords_norm,
            point_labels=point_labels_norm,
            mask_input=mask_input_tensor,
            has_mask=has_mask,
        )

        masks, scores, logits = self.edgetam.postprocess(
            low_res_masks, iou_predictions, (height // 8, width // 8)
        )

        return masks, scores, logits


    # =========================================================
    # 3. 推論処理部（SAM組み込み対象の関数）
    # =========================================================
    def run_sam_inference(
        self,
        image: np.ndarray,
        feats: List[np.ndarray], 
        points: List[List[float]],
        point_labels: List[int],
        logits: np.ndarray | None,
    ):
        """SAM等のモデル推論を実行する処理部分。"""
        if len(points) == 0:
            return None, None

        height, width = image.shape[:2]

        masks, scores, logits = self._predict_edgetam_decoder(points, point_labels, feats, height, width, logits)

        best_idx = np.argmax(scores)
        current_mask = masks[best_idx].astype(np.uint8)
        low_res_logits = logits[best_idx:best_idx + 1]

        resized_masks: np.ndarray = cv2.resize(
            current_mask,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_masks, low_res_logits

    # =========================================================
    # 4. イベントハンドラ & GUI制御
    # =========================================================
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.cv_image = cv2.imread(file_path)
        self.reset_prompts()

        # 特徴量抽出
        cv_image_rgb: np.ndarray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.feats: List[np.ndarray] = self._predict_edgetam_encoder(cv_image_rgb)

    def reset_prompts(self):
        """ポイントや過去のlogits・表示位置を全リセット"""
        self.points.clear()
        self.point_labels.clear()
        self.latest_logits = None
        self.latest_mask = None
        self.reset_view()
        self.update_status()

    def reset_view(self):
        """拡大率と表示位置を初期位置にリセット"""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.redraw()

    def on_left_click(self, event):
        """Positive Point (+) の追加"""
        self.add_point(event.x, event.y, label=1)

    def on_right_click(self, event):
        """Negative Point (-) の追加"""
        self.add_point(event.x, event.y, label=0)

    def add_point(self, canvas_x, canvas_y, label):
        if self.cv_image is None:
            return

        img_x, img_y = self.canvas_to_img_coords(canvas_x, canvas_y)
        h, w = self.cv_image.shape[:2]

        if not (0 <= img_x < w and 0 <= img_y < h):
            return

        self.points.append([img_x, img_y])
        self.point_labels.append(label)

        self.latest_mask, self.latest_logits = self.run_sam_inference(
            image=self.cv_image,
            feats=self.feats, 
            points=self.points,
            point_labels=self.point_labels,
            logits=self.latest_logits,
        )

        self.update_status()
        self.redraw()

    # --- ズーム & パン処理 ---
    def on_zoom(self, event):
        if self.cv_image is None:
            return

        # ズーム倍率計算
        if event.num == 4 or event.delta > 0:
            factor = 1.15
        elif event.num == 5 or event.delta < 0:
            factor = 1 / 1.15
        else:
            return

        new_zoom = self.zoom_factor * factor
        if new_zoom < 0.2 or new_zoom > 30.0:  # ズーム範囲制限
            return

        # マウスカーソル位置を中心に拡大縮小する計算
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.cv_image.shape[:2]
        base_scale = min(canvas_w / img_w, canvas_h / img_h)

        old_scale = base_scale * self.zoom_factor
        old_offset_x = (canvas_w - img_w * old_scale) / 2 + self.pan_offset_x
        old_offset_y = (canvas_h - img_h * old_scale) / 2 + self.pan_offset_y

        img_x = (event.x - old_offset_x) / old_scale
        img_y = (event.y - old_offset_y) / old_scale

        self.zoom_factor = new_zoom
        new_scale = base_scale * self.zoom_factor

        new_offset_x = event.x - img_x * new_scale
        new_offset_y = event.y - img_y * new_scale

        self.pan_offset_x = new_offset_x - (canvas_w - img_w * new_scale) / 2
        self.pan_offset_y = new_offset_y - (canvas_h - img_h * new_scale) / 2

        self.redraw()

    def on_pan_start(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_pan_drag(self, event):
        if self.cv_image is None:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.redraw()

    def update_status(self):
        logits_str = (
            f"Shape {self.latest_logits.shape}"
            if self.latest_logits is not None
            else "None"
        )
        pos_cnt = self.point_labels.count(1)
        neg_cnt = self.point_labels.count(0)
        text = f"Pos (+): {pos_cnt} / Neg (-): {neg_cnt}\nLogits: {logits_str}"
        self.lbl_status.configure(text=text)

    # =========================================================
    # 5. 描画・座標変換ロジック
    # =========================================================
    def redraw(self):
        if self.cv_image is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        # 1. 表示用の画像作成（Mask重畳表示処理）
        display_cv = self.cv_image.copy()

        if self.latest_mask is not None:
            mask_color = np.array([0, 255, 0], dtype=np.uint8)  # マスク色 (BGR: 緑)
            alpha = 0.45
            mask_bool = self.latest_mask.astype(bool)

            overlay = display_cv.copy()
            overlay[mask_bool] = mask_color
            cv2.addWeighted(
                overlay, alpha, display_cv, 1 - alpha, 0, display_cv
            )

            # マスク境界（輪郭線）
            mask_uint8 = mask_bool.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display_cv, contours, -1, (0, 255, 0), 2)

        # 2. ズーム・パンを反映した描画計算
        img_h, img_w = display_cv.shape[:2]
        base_scale = min(canvas_w / img_w, canvas_h / img_h)
        scale = base_scale * self.zoom_factor

        offset_x = (canvas_w - img_w * scale) / 2 + self.pan_offset_x
        offset_y = (canvas_h - img_h * scale) / 2 + self.pan_offset_y

        # キャンバス枠外のクロップ処理（パフォーマンス維持用）
        src_x1 = max(0, int(-offset_x / scale)) if offset_x < 0 else 0
        src_y1 = max(0, int(-offset_y / scale)) if offset_y < 0 else 0
        src_x2 = (
            min(img_w, int((canvas_w - offset_x) / scale))
            if (canvas_w - offset_x) / scale <= img_w
            else img_w
        )
        src_y2 = (
            min(img_h, int((canvas_h - offset_y) / scale))
            if (canvas_h - offset_y) / scale <= img_h
            else img_h
        )

        dst_x1 = int(src_x1 * scale + offset_x)
        dst_y1 = int(src_y1 * scale + offset_y)
        dst_w = int((src_x2 - src_x1) * scale)
        dst_h = int((src_y2 - src_y1) * scale)

        self.canvas.delete("all")

        if dst_w > 0 and dst_h > 0 and (src_x2 > src_x1) and (src_y2 > src_y1):
            crop_img = display_cv[src_y1:src_y2, src_x1:src_x2]
            resized = cv2.resize(
                crop_img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR
            )
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))
            self.canvas.create_image(
                dst_x1, dst_y1, anchor="nw", image=self.tk_img
            )

        # 3. プロンプトポイントの描画（拡大表示に応じて位置を自動計算）
        r = 5
        for pt, label in zip(self.points, self.point_labels):
            cx = int(pt[0] * scale + offset_x)
            cy = int(pt[1] * scale + offset_y)
            color = "#00FF00" if label == 1 else "#FF0000"  # Pos: 緑, Neg: 赤
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r, fill=color, outline="white", width=2
            )

    def canvas_to_img_coords(self, cx, cy):
        """Canvas座標から元画像座標へ逆変換（ズーム・パン対応）"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.cv_image.shape[:2]

        base_scale = min(canvas_w / img_w, canvas_h / img_h)
        scale = base_scale * self.zoom_factor

        offset_x = (canvas_w - img_w * scale) / 2 + self.pan_offset_x
        offset_y = (canvas_h - img_h * scale) / 2 + self.pan_offset_y

        img_x = int((cx - offset_x) / scale)
        img_y = int((cy - offset_y) / scale)
        return img_x, img_y

    def on_resize(self, event):
        self.redraw()


if __name__ == "__main__":
    app = SAMInteractiveApp()
    app.mainloop()