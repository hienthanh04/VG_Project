"""
NHÁNH A — BƯỚC 1: PHÂN TÍCH MÀU SẮC ĐẶC TRƯNG VAN GOGH
=========================================================
Phân tích phân bố màu HSV của:
  1. Tranh Van Gogh gốc (ground truth)
  2. Ảnh suy giảm (input)
  3. Ảnh phục hồi (output model)

Output:
  - color_analysis/hsv_histograms.jpg   : biểu đồ histogram HSV 3 nhóm
  - color_analysis/color_palette.jpg    : palette màu đặc trưng Van Gogh
  - color_analysis/color_stats.csv      : số liệu chi tiết từng ảnh
  - color_analysis/summary.txt          : nhận xét tổng hợp

Cách dùng:
    python A1_color_analysis.py \
        --original_dir  ./dataset_restore/test_originals \
        --degraded_dir  ./dataset_restore/test \
        --restored_dir  ./eval_results/restored \
        --output_dir    ./color_analysis

    # Hoặc chỉ phân tích tranh gốc trước:
    python A1_color_analysis.py --original_dir ./vangogh_color --output_dir ./color_analysis
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.cluster import KMeans

# ── Hàm tiện ích ─────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_images(folder):
    """Đọc tất cả ảnh trong folder, trả về list numpy arrays (RGB)."""
    paths = sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS])
    imgs  = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append((p.stem, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    print(f"  Đọc được {len(imgs)} ảnh từ: {folder}")
    return imgs


def rgb_to_hsv_flat(img_rgb):
    """Chuyển ảnh RGB → HSV và trả về mảng phẳng (N, 3)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    return hsv  # H: 0-180, S: 0-255, V: 0-255

# ── 1. Histogram HSV tổng hợp ─────────────────────────────────────────────────

def compute_avg_histogram(images, channel, bins, value_range):
    """
    Tính histogram trung bình của một kênh HSV trên toàn bộ tập ảnh.
    Mỗi ảnh được chuẩn hóa trước khi lấy trung bình → so sánh công bằng.
    """
    histograms = []
    for _, img in images:
        hsv_flat = rgb_to_hsv_flat(img)
        hist, edges = np.histogram(hsv_flat[:, channel], bins=bins, range=value_range, density=True)
        histograms.append(hist)
    avg = np.mean(histograms, axis=0)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, avg


def plot_hsv_comparison(orig_imgs, deg_imgs, rest_imgs, output_dir):
    """
    Vẽ biểu đồ so sánh histogram H, S, V cho 3 nhóm ảnh.
    Layout: 3 hàng (H, S, V) × 1 cột, mỗi hàng có 3 đường.
    """
    configs = [
        # (channel_idx, tên kênh, số bins, range, nhãn trục X)
        (0, "Hue (màu sắc)",      36, (0, 180),  "Giá trị H (0=đỏ, 60=vàng, 120=xanh lá, 150=xanh dương)"),
        (1, "Saturation (độ bão hòa)", 32, (0, 255), "Giá trị S"),
        (2, "Value (độ sáng)",    32, (0, 255),  "Giá trị V"),
    ]

    groups = []
    if orig_imgs:  groups.append(("Tranh gốc Van Gogh", orig_imgs,  "#1D9E75", "-"))
    if deg_imgs:   groups.append(("Input (degraded)",   deg_imgs,   "#888780", "--"))
    if rest_imgs:  groups.append(("Output (restored)",  rest_imgs,  "#D85A30", ":"))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle("Phân tích phân bố màu HSV — So sánh 3 nhóm ảnh", fontsize=13, y=0.98)

    for ax, (ch_idx, ch_name, bins, rng, xlabel) in zip(axes, configs):
        for label, imgs, color, ls in groups:
            centers, avg_hist = compute_avg_histogram(imgs, ch_idx, bins, rng)
            ax.plot(centers, avg_hist, label=label, color=color, linestyle=ls, linewidth=1.8)

        # Vùng màu đặc trưng Van Gogh trên kênh H
        if ch_idx == 0:
            ax.axvspan(20, 35,  alpha=0.10, color="#EF9F27", label="Vàng (đặc trưng VG)")
            ax.axvspan(100, 130, alpha=0.10, color="#378ADD", label="Xanh dương (đặc trưng VG)")

        ax.set_title(ch_name, fontsize=11, fontweight="500")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Mật độ xác suất", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, "hsv_histograms.jpg")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu: {out_path}")


# ── 2. Color palette đặc trưng Van Gogh ──────────────────────────────────────

def extract_dominant_colors(images, n_colors=8):
    """
    Dùng K-Means clustering trên không gian màu để tìm
    n_colors màu đặc trưng nhất trong bộ tranh Van Gogh.
    """
    print("  Đang trích xuất palette màu (K-Means)...")
    # Lấy mẫu để tăng tốc
    all_pixels = []
    for _, img in images:
        resized = cv2.resize(img, (64, 64))
        all_pixels.append(resized.reshape(-1, 3))
    pixels = np.vstack(all_pixels).astype(np.float32)

    # Subsample nếu quá nhiều điểm
    if len(pixels) > 50000:
        idx = np.random.choice(len(pixels), 50000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Sắp xếp theo tần suất (cluster size)
    counts = np.bincount(kmeans.labels_)
    order  = np.argsort(-counts)
    return centers[order], counts[order] / counts.sum()


def plot_color_palette(images, output_dir, n_colors=8):
    """Vẽ và lưu bảng màu đặc trưng của bộ tranh Van Gogh."""
    colors, freqs = extract_dominant_colors(images, n_colors)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_title(f"Palette {n_colors} màu đặc trưng nhất — Tranh Van Gogh", fontsize=11)

    for i, (color, freq) in enumerate(zip(colors, freqs)):
        rect = plt.Rectangle([i, 0], 0.95, 1.0, color=color/255)
        ax.add_patch(rect)
        r, g, b = color
        hex_str = f"#{r:02X}{g:02X}{b:02X}"
        text_color = "white" if (0.299*r + 0.587*g + 0.114*b) < 128 else "black"
        ax.text(i + 0.475, 0.5, hex_str,   ha="center", va="center", fontsize=7,
                color=text_color, fontweight="bold")
        ax.text(i + 0.475, 0.1, f"{freq*100:.1f}%", ha="center", va="bottom",
                fontsize=7, color=text_color)

    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "color_palette.jpg")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu: {out_path}")
    return colors, freqs

# ── 3. Số liệu chi tiết từng ảnh ─────────────────────────────────────────────

def compute_color_stats_single(img_rgb):
    """
    Tính các đặc trưng màu định lượng cho một ảnh:
    - Mean/std của H, S, V
    - Tỉ lệ pixel thuộc vùng màu vàng và xanh dương (đặc trưng Van Gogh)
    - Độ bão hòa màu trung bình (Van Gogh nổi tiếng dùng màu bão hòa cao)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Vùng vàng: H ∈ [20, 35], S > 80
    yellow_mask = (h >= 20) & (h <= 35) & (s > 80)
    # Vùng xanh dương: H ∈ [100, 130], S > 60
    blue_mask   = (h >= 100) & (h <= 130) & (s > 60)
    total_px    = h.size

    return {
        "h_mean": round(float(h.mean()), 3),
        "h_std":  round(float(h.std()),  3),
        "s_mean": round(float(s.mean()), 3),
        "s_std":  round(float(s.std()),  3),
        "v_mean": round(float(v.mean()), 3),
        "v_std":  round(float(v.std()),  3),
        "yellow_ratio": round(float(yellow_mask.sum()) / total_px, 4),
        "blue_ratio":   round(float(blue_mask.sum())   / total_px, 4),
        "saturation_high_ratio": round(float((s > 150).sum()) / total_px, 4),
    }


def compute_color_stats_group(images, label):
    records = []
    for name, img in images:
        stats = compute_color_stats_single(img)
        stats["image"] = name
        stats["group"] = label
        records.append(stats)
    return records


# ── 4. Đo độ tương đồng histogram (Bhattacharyya distance) ───────────────────

def histogram_similarity(imgs_a, imgs_b, channel, bins, value_range):
    """
    Tính khoảng cách Bhattacharyya trung bình giữa histograms của 2 nhóm ảnh.
    Giá trị càng thấp = 2 nhóm càng giống nhau về phân bố màu.
    """
    _, hist_a = compute_avg_histogram(imgs_a, channel, bins, value_range)
    _, hist_b = compute_avg_histogram(imgs_b, channel, bins, value_range)

    # Bhattacharyya: -ln(sum(sqrt(p*q)))
    bc = np.sum(np.sqrt(hist_a * hist_b + 1e-10))
    bd = -np.log(bc + 1e-10)
    return round(float(bd), 4)


# ── Pipeline chính ────────────────────────────────────────────────────────────

def analyze(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1] Đang đọc ảnh...")
    orig_imgs = load_images(args.original_dir) if args.original_dir else []
    deg_imgs  = load_images(args.degraded_dir) if args.degraded_dir else []
    rest_imgs = load_images(args.restored_dir) if args.restored_dir else []

    if not orig_imgs:
        raise ValueError("Cần ít nhất --original_dir để phân tích!")

    print("\n[2] Vẽ histogram HSV so sánh...")
    plot_hsv_comparison(orig_imgs, deg_imgs, rest_imgs, args.output_dir)

    print("\n[3] Trích xuất color palette...")
    colors, freqs = plot_color_palette(orig_imgs, args.output_dir, n_colors=8)

    print("\n[4] Tính số liệu chi tiết từng ảnh...")
    all_records = []
    if orig_imgs:  all_records += compute_color_stats_group(orig_imgs, "original")
    if deg_imgs:   all_records += compute_color_stats_group(deg_imgs,  "degraded")
    if rest_imgs:  all_records += compute_color_stats_group(rest_imgs, "restored")

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(args.output_dir, "color_stats.csv"), index=False)

    print("\n[5] Tính độ tương đồng histogram (Bhattacharyya)...")
    summary_lines = ["PHÂN TÍCH MÀU SẮC ĐẶC TRƯNG VAN GOGH\n" + "="*50 + "\n"]

    # Thống kê trung bình theo nhóm
    summary_lines.append("Thống kê trung bình theo nhóm:")
    for grp in df["group"].unique():
        sub = df[df["group"] == grp]
        summary_lines.append(f"\n  [{grp.upper()}]")
        summary_lines.append(f"  Saturation trung bình : {sub['s_mean'].mean():.1f}/255")
        summary_lines.append(f"  Tỉ lệ pixel vàng      : {sub['yellow_ratio'].mean()*100:.2f}%")
        summary_lines.append(f"  Tỉ lệ pixel xanh dương: {sub['blue_ratio'].mean()*100:.2f}%")
        summary_lines.append(f"  Tỉ lệ màu bão hòa cao : {sub['saturation_high_ratio'].mean()*100:.2f}%")

    # So sánh độ tương đồng với tranh gốc
    if rest_imgs and orig_imgs:
        summary_lines.append("\n\nĐộ tương đồng histogram với tranh gốc (Bhattacharyya — thấp hơn = giống hơn):")
        for ch_idx, ch_name, bins, rng in [(0,"Hue",36,(0,180)),(1,"Saturation",32,(0,255)),(2,"Value",32,(0,255))]:
            if deg_imgs:
                bd_deg  = histogram_similarity(orig_imgs, deg_imgs,  ch_idx, bins, rng)
                summary_lines.append(f"  {ch_name:12s}: degraded vs original = {bd_deg:.4f}")
            bd_rest = histogram_similarity(orig_imgs, rest_imgs, ch_idx, bins, rng)
            summary_lines.append(f"  {ch_name:12s}: restored vs original = {bd_rest:.4f}  ← model đã học phân bố màu")

    # Màu đặc trưng
    summary_lines.append("\n\nTop 5 màu đặc trưng nhất (K-Means):")
    for i, (color, freq) in enumerate(zip(colors[:5], freqs[:5])):
        r, g, b = color
        summary_lines.append(f"  #{r:02X}{g:02X}{b:02X}  (RGB: {r},{g},{b}) — {freq*100:.1f}% pixel")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nHoàn thành! Kết quả tại: {args.output_dir}/")
    print("  hsv_histograms.jpg  — so sánh phân bố màu 3 nhóm")
    print("  color_palette.jpg   — palette màu đặc trưng")
    print("  color_stats.csv     — số liệu từng ảnh")
    print("  summary.txt         — nhận xét tổng hợp")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích màu sắc đặc trưng Van Gogh")
    parser.add_argument("--original_dir", default="./vangogh_color",
                        help="Thư mục tranh Van Gogh màu gốc")
    parser.add_argument("--degraded_dir", default=None,
                        help="Thư mục ảnh suy giảm (input model) — tùy chọn")
    parser.add_argument("--restored_dir", default=None,
                        help="Thư mục ảnh phục hồi (output model) — tùy chọn")
    parser.add_argument("--output_dir",   default="./color_analysis")
    args = parser.parse_args()
    analyze(args)

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
# Chỉ phân tích tranh gốc (làm trước):
#   python A1_color_analysis.py --original_dir ./vangogh_color
#
# So sánh cả 3 nhóm:
#   python A1_color_analysis.py \
#       --original_dir ./vangogh_color \
#       --degraded_dir ./dataset_restore/test_inputs \
#       --restored_dir ./eval_results/restored
