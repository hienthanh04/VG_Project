"""
NHÁNH A — BƯỚC 3: BÁO CÁO TỔNG HỢP PHONG CÁCH
================================================
Gom kết quả từ A1 (màu sắc) và A2 (texture) thành 1 figure tổng hợp
để dùng trong báo cáo NCKH.

Output:
    style_report/vangogh_style_report.jpg  — figure báo cáo (có thể chèn thẳng vào Word)
    style_report/style_score_table.csv    — bảng điểm phong cách tổng hợp

Cách dùng:
    python A3_style_report.py \
        --color_dir   ./color_analysis \
        --texture_dir ./texture_analysis \
        --output_dir  ./style_report
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from pathlib import Path


def load_img(path):
    """Đọc ảnh để chèn vào subplot."""
    if not os.path.exists(path):
        return None
    return mpimg.imread(path)


def load_stats(csv_path, group_col="group"):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def build_style_score(color_df, texture_df):
    """
    Tính Style Score tổng hợp cho mỗi nhóm ảnh.
    Score được tính từ độ tương đồng với nhóm original về:
      - Saturation (màu bão hòa)
      - Yellow/Blue ratio (màu đặc trưng Van Gogh)
      - Gabor energy (texture nét cọ)
      - LBP entropy (độ phong phú texture)
    """
    records = []

    groups = set()
    if color_df is not None:   groups |= set(color_df["group"].unique())
    if texture_df is not None: groups |= set(texture_df["group"].unique())

    ref_color   = color_df[color_df["group"] == "original"]   if color_df   is not None else None
    ref_texture = texture_df[texture_df["group"] == "original"] if texture_df is not None else None

    def norm_diff(ref_series, target_series):
        """Trả về 1 - |norm_diff| → 1.0 = giống hệt."""
        ref_mean = ref_series.mean()
        tgt_mean = target_series.mean()
        diff = abs(ref_mean - tgt_mean) / (abs(ref_mean) + 1e-8)
        return max(0.0, 1.0 - diff)

    for grp in sorted(groups):
        row = {"group": grp}
        scores = []

        if ref_color is not None and color_df is not None:
            tgt = color_df[color_df["group"] == grp]
            if len(tgt) > 0:
                s1 = norm_diff(ref_color["s_mean"],        tgt["s_mean"])
                s2 = norm_diff(ref_color["yellow_ratio"],  tgt["yellow_ratio"])
                s3 = norm_diff(ref_color["blue_ratio"],    tgt["blue_ratio"])
                row.update({"color_saturation_score": round(s1,3),
                             "yellow_ratio_score":    round(s2,3),
                             "blue_ratio_score":      round(s3,3)})
                scores += [s1, s2, s3]

        if ref_texture is not None and texture_df is not None:
            tgt = texture_df[texture_df["group"] == grp]
            if len(tgt) > 0:
                s4 = norm_diff(ref_texture["gabor_total_energy"], tgt["gabor_total_energy"])
                s5 = norm_diff(ref_texture["lbp_entropy"],        tgt["lbp_entropy"])
                row.update({"gabor_energy_score": round(s4,3),
                             "lbp_entropy_score": round(s5,3)})
                scores += [s4, s5]

        row["style_score_overall"] = round(float(np.mean(scores)), 3) if scores else 0.0
        records.append(row)

    return pd.DataFrame(records)


def build_report(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Đọc dữ liệu ──
    color_df   = load_stats(os.path.join(args.color_dir,   "color_stats.csv"))
    texture_df = load_stats(os.path.join(args.texture_dir, "texture_stats.csv"))
    score_df   = build_style_score(color_df, texture_df)

    score_path = os.path.join(args.output_dir, "style_score_table.csv")
    score_df.to_csv(score_path, index=False)
    print("\nBảng Style Score:")
    print(score_df[["group", "style_score_overall"]].to_string(index=False))

    # ── Ảnh từ các bước trước ──
    hsv_img      = load_img(os.path.join(args.color_dir,   "hsv_histograms.jpg"))
    palette_img  = load_img(os.path.join(args.color_dir,   "color_palette.jpg"))
    gabor_img    = load_img(os.path.join(args.texture_dir, "gabor_orientation.jpg"))
    lbp_img      = load_img(os.path.join(args.texture_dir, "lbp_histogram.jpg"))
    heatmap_img  = load_img(os.path.join(args.texture_dir, "gabor_heatmap_sample.jpg"))

    # ── Layout báo cáo ──
    fig = plt.figure(figsize=(16, 18))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.40, wspace=0.25,
                            top=0.93, bottom=0.04)

    fig.suptitle(
        "Phân tích Phong Cách Hội Họa Van Gogh\n"
        "Màu sắc đặc trưng · Texture · Brushstroke",
        fontsize=14, fontweight="500", y=0.97
    )

    def show_img(ax, img, title):
        if img is not None:
            ax.imshow(img, aspect="auto")
        else:
            ax.text(0.5, 0.5, f"[{title}\nchưa có file]",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888")
            ax.set_facecolor("#f5f5f5")
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    show_img(fig.add_subplot(gs[0, 0]), hsv_img,
             "Phân bố màu HSV — So sánh 3 nhóm")
    show_img(fig.add_subplot(gs[0, 1]), palette_img,
             "Palette màu đặc trưng Van Gogh (K-Means)")
    show_img(fig.add_subplot(gs[1, 0]), gabor_img,
             "Hướng nét cọ (Gabor Energy) — Polar & Bar")
    show_img(fig.add_subplot(gs[1, 1]), lbp_img,
             "Micro-texture (LBP Histogram) — So sánh 3 nhóm")

    if heatmap_img is not None:
        ax_hm = fig.add_subplot(gs[2, :])
        show_img(ax_hm, heatmap_img, "Heatmap Gabor — Vùng nét cọ được phát hiện trên tranh Van Gogh")
    else:
        ax_hm = fig.add_subplot(gs[2, :])
        ax_hm.axis("off")

    # ── Bảng Style Score ──
    ax_score = fig.add_subplot(gs[3, :])
    ax_score.axis("off")
    ax_score.set_title("Style Score tổng hợp — Mức độ học phong cách Van Gogh", fontsize=10, pad=8)

    display_cols = ["group", "style_score_overall"]
    extra_cols   = [c for c in ["color_saturation_score","yellow_ratio_score",
                                 "blue_ratio_score","gabor_energy_score","lbp_entropy_score"]
                    if c in score_df.columns]
    display_cols += extra_cols

    score_display = score_df[display_cols].copy()
    score_display.columns = [c.replace("_score","").replace("_"," ").title()
                              for c in display_cols]

    table = ax_score.table(
        cellText  = score_display.values,
        colLabels = score_display.columns,
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Header row màu xanh
    for j in range(len(score_display.columns)):
        cell = table[0, j]
        cell.set_facecolor("#1D9E75")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight ô Overall cao nhất (trừ original)
    if "style_score_overall" in score_df.columns:
        max_grp = (score_df[score_df["group"] != "original"]
                   .sort_values("style_score_overall", ascending=False)
                   .iloc[0]["group"] if len(score_df[score_df["group"] != "original"]) > 0 else None)
        for i, row in enumerate(score_df.itertuples(), start=1):
            bg = "#E1F5EE" if row.group == "original" else \
                 "#FFF9E6" if row.group == max_grp else "white"
            for j in range(len(score_display.columns)):
                table[i, j].set_facecolor(bg)

    # Ghi chú cuối
    fig.text(0.5, 0.015,
             "Style Score = 1.0 là giống hệt tranh gốc Van Gogh · "
             "'restored' cần đạt gần 'original' hơn 'degraded' để chứng minh model đã học phong cách",
             ha="center", fontsize=8, color="#555", style="italic")

    out_path = os.path.join(args.output_dir, "vangogh_style_report.jpg")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nBáo cáo tổng hợp lưu tại: {out_path}")
    print("→ Chèn file này trực tiếp vào Word/báo cáo NCKH.")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Báo cáo tổng hợp phong cách Van Gogh")
    parser.add_argument("--color_dir",   default="./color_analysis")
    parser.add_argument("--texture_dir", default="./texture_analysis")
    parser.add_argument("--output_dir",  default="./style_report")
    args = parser.parse_args()
    build_report(args)

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
# python A3_style_report.py \
#     --color_dir   ./color_analysis \
#     --texture_dir ./texture_analysis \
#     --output_dir  ./style_report
