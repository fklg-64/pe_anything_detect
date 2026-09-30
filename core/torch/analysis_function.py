import io
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.chart import LineChart, Reference
from openpyxl.drawing.image import Image as DrawImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import seaborn as sns
import torch

def export_complete_vit_analysis_to_excel(
    layer_features, H, W, output_filename="vit_layer_analysis_complete.xlsx"
):
    """バイオリンプロット画像、Excel純正折れ線グラフ、サマリー表、正方形2Dマップを1つの.xlsxに出力"""
    num_layers = len(layer_features)
    num_patches = H * W

    # --- 1. ノルム計算とパーセンタイル算出 ---
    all_norms = []  # [12, HW+1] (CLS含む)
    patch_norms_list = []  # [12, HW] (CLS除外)

    percentiles = {
        "P50": [],
        "P90": [],
        "P99": [],
        "P99.9": [],
        "Max": [],
    }

    for feat in layer_features:
        norms = torch.norm(feat[0], p=2, dim=-1).cpu().numpy()
        all_norms.append(norms)

        p_norms = norms[1:]  # CLS除外
        patch_norms_list.append(p_norms)

        percentiles["P50"].append(float(np.percentile(p_norms, 50)))
        percentiles["P90"].append(float(np.percentile(p_norms, 90)))
        percentiles["P99"].append(float(np.percentile(p_norms, 99)))
        percentiles["P99.9"].append(float(np.percentile(p_norms, 99.9)))
        percentiles["Max"].append(float(np.max(p_norms)))

    all_norms = np.array(all_norms)

    # --- 2. Matplotlibでバイオリンプロットを作成しメモリ上に保存 ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.violinplot(
        data=patch_norms_list,
        inner=None,
        color="#e0e0e0",
        linewidth=1,
        cut=0,
        ax=ax,
    )

    x = np.arange(num_layers)
    ax.plot(x, percentiles["P50"], marker="o", color="#1f77b4", label="P50")
    ax.plot(x, percentiles["P90"], marker="s", color="#2ca02c", label="P90")
    ax.plot(x, percentiles["P99"], marker="^", color="#ff7f0e", label="P99")
    ax.plot(x, percentiles["P99.9"], marker="d", color="#d62728", label="P99.9")
    ax.plot(
        x,
        percentiles["Max"],
        marker="x",
        color="#8c564b",
        linestyle="--",
        label="Max",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}" for i in range(num_layers)])
    ax.set_title(
        "Layer-wise Patch Norm Distribution (Violin + Percentiles)",
        fontweight="bold",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Patch Norm")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    # 画像をファイルに保存せずバイト列として取得
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close(fig)
    img_buf.seek(0)

    # --- 3. OpenPyXL Workbookの初期化 ---
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    font_small = Font(name="Segoe UI", size=8)
    font_bold_small = Font(name="Segoe UI", size=8, bold=True)
    font_header = Font(name="Segoe UI", size=9, bold=True, color="FFFFFF")
    fill_header = PatternFill(
        start_color="366092", end_color="366092", fill_type="solid"
    )
    fill_cls = PatternFill(
        start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"
    )
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    color_scale = ColorScaleRule(
        start_type="min",
        start_color="63BE7B",
        mid_type="percentile",
        mid_value=50,
        mid_color="FFEB84",
        end_type="max",
        end_color="F8696B",
    )

    # ==========================================
    # シート1: 分布・グラフシート (プロット1)
    # ==========================================
    ws_dist = wb.create_sheet(title="Norm_Distribution")
    ws_dist.views.sheetView[0].showGridLines = True

    # バイオリンプロット画像を挿入 (セル A1)
    img = DrawImage(img_buf)
    img.width = 650
    img.height = 325
    ws_dist.add_image(img, "A1")

    # パーセンタイル集計表の書き込み (セル A18 から)
    start_row_pct = 18
    pct_headers = ["Layer"] + list(percentiles.keys())
    for c_idx, h in enumerate(pct_headers, 1):
        cell = ws_dist.cell(row=start_row_pct, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center")

    for l_idx in range(num_layers):
        r = start_row_pct + 1 + l_idx
        ws_dist.cell(row=r, column=1, value=f"L{l_idx+1}")
        for c_idx, key in enumerate(percentiles.keys(), 2):
            val_cell = ws_dist.cell(
                row=r, column=c_idx, value=percentiles[key][l_idx]
            )
            val_cell.number_format = "0.00"
            val_cell.border = thin_border
            val_cell.font = font_small

    # Excel純正の折れ線グラフを追加 (集計表から動的生成)
    chart = LineChart()
    chart.title = "Percentile Spikes Across Layers (Excel Native Chart)"
    chart.style = 13
    chart.y_axis.title = "Patch Norm"
    chart.x_axis.title = "Layer"

    data_ref = Reference(
        ws_dist,
        min_col=2,
        min_row=start_row_pct,
        max_col=6,
        max_row=start_row_pct + num_layers,
    )
    cats_ref = Reference(
        ws_dist,
        min_col=1,
        min_row=start_row_pct + 1,
        max_row=start_row_pct + num_layers,
    )

    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)
    chart.width = 14
    chart.height = 8

    # 折れ線グラフを H1 に挿入
    ws_dist.add_chart(chart, "H1")

    # ==========================================
    # シート2: 全層サマリー
    # ==========================================
    ws_summary = wb.create_sheet(title="Norm_Summary")
    ws_summary.views.sheetView[0].showGridLines = True

    headers = ["Index", "Type"] + [f"Layer_{i+1}" for i in range(num_layers)]
    ws_summary.append(headers)

    for col_num in range(1, len(headers) + 1):
        cell = ws_summary.cell(row=1, column=col_num)
        cell.font = font_header
        cell.fill = fill_header

    for p_idx in range(num_patches + 1):
        row_num = p_idx + 2
        token_type = "CLS" if p_idx == 0 else f"Patch_{p_idx-1}"
        row_data = [p_idx, token_type] + [
            float(all_norms[l, p_idx]) for l in range(num_layers)
        ]
        ws_summary.append(row_data)

        if p_idx == 0:
            for c in range(1, len(row_data) + 1):
                cell = ws_summary.cell(row=row_num, column=c)
                cell.fill = fill_cls
                cell.font = font_bold_small

    for r in range(2, num_patches + 3):
        for c in range(3, num_layers + 3):
            cell = ws_summary.cell(row=r, column=c)
            cell.font = font_small
            cell.number_format = "0.00"
            cell.border = thin_border

    for col in range(3, num_layers + 3):
        col_letter = openpyxl.utils.get_column_letter(col)
        ws_summary.conditional_formatting.add(
            f"{col_letter}3:{col_letter}{num_patches+2}", color_scale
        )

    # ==========================================
    # シート3: 各層 2D Map (正方形・小さめセル)
    # ==========================================
    SQUARE_ROW_HEIGHT = 20.0
    SQUARE_COL_WIDTH = 5.2

    for l in range(num_layers):
        ws_2d = wb.create_sheet(title=f"L{l+1}_2D_Map")
        ws_2d.views.sheetView[0].showGridLines = True

        ws_2d.cell(
            row=1, column=1, value=f"L{l+1} CLS Norm:"
        ).font = font_bold_small
        cls_cell = ws_2d.cell(row=1, column=3, value=float(all_norms[l, 0]))
        cls_cell.font = font_bold_small
        cls_cell.fill = fill_cls
        cls_cell.number_format = "0.00"

        patch_norms_2d = all_norms[l, 1:].reshape(H, W)
        start_row = 3

        for r in range(H):
            current_row = start_row + r
            ws_2d.row_dimensions[current_row].height = SQUARE_ROW_HEIGHT

            for c in range(W):
                col_letter = openpyxl.utils.get_column_letter(c + 1)
                ws_2d.column_dimensions[col_letter].width = SQUARE_COL_WIDTH

                cell = ws_2d.cell(
                    row=current_row,
                    column=c + 1,
                    value=float(patch_norms_2d[r, c]),
                )
                cell.font = font_small
                cell.number_format = "0.00"
                cell.border = thin_border
                cell.alignment = Alignment(
                    horizontal="center", vertical="center"
                )

        end_col_letter = openpyxl.utils.get_column_letter(W)
        ws_2d.conditional_formatting.add(
            f"A{start_row}:{end_col_letter}{start_row + H - 1}", color_scale
        )

    wb.save(output_filename)
    print(f"全解析データ集約Excelを出力しました: {output_filename}")
