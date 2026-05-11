"""
NHÁNH A — BƯỚC 2: PHÂN TÍCH TEXTURE & BRUSHSTROKE
====================================================
Dùng 2 phương pháp để đo đặc trưng nét cọ Van Gogh:

  1. Gabor Filter — đo hướng và tần số của nét cọ
     Van Gogh nổi tiếng với nét cọ có hướng rõ ràng (xoáy, chéo, ngang)
     Gabor ở nhiều góc (0°–150°) đo energy theo từng hướng → phân bố hướng cọ

  2. LBP (Local Binary Pattern) — đo micro-texture bề mặt
     LBP so sánh từng pixel với 8 láng giềng → mã nhị phân → histogram
     Van Gogh có texture thô, phân bố LBP rất khác ảnh photo thông thường

Output:
  - texture_analysis/gabor_orientation.jpg  : biểu đồ energy theo hướng
  - texture_analysis/lbp_histogram.jpg      : histogram LBP so sánh 3 nhóm
  - texture_analysis/gabor_heatmap_sample.jpg: heatmap Gabor trên ảnh mẫu
  - texture_analysis/texture_stats.csv      : số liệu chi tiết
  - texture_analysis/summary.txt            : nhận xét

Cách dùng:
    python A2_texture_analysis.py \
        --original_dir ./vangogh_color \
        --restored_dir ./eval_results/restored \
        --output_dir   ./texture_analysis
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.feature import local_binary_pattern

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Hàm tiện ích ─────────────────────────────────────────────────────────────

def load_gray_images(folder, max_size=256):
    """Đọc ảnh, chuyển grayscale, resize về max_size để đồng nhất."""
    paths = sorted([p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS])
    imgs  = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (max_size, max_size))
            imgs.append((p.stem, img.astype(np.float32) / 255.0))
    print(f"  Đọc được {len(imgs)} ảnh từ: {folder}")
    return imgs


# ── 1. Gabor Filter — phân tích hướng nét cọ ─────────────────────────────────

def build_gabor_kernels(orientations, frequency=0.15, sigma=3.0, size=31):
    """
    Tạo bộ Gabor kernels ở nhiều hướng khác nhau.
    frequency: tần số của sóng (liên quan kích thước nét cọ)
    sigma    : độ rộng của Gaussian envelope
    """
    kernels = []
    for theta_deg in orientations:
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel(
            ksize=(size, size),
            sigma=sigma,
            theta=theta,
            lambd=1.0 / frequency,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        kernel /= kernel.sum() + 1e-8
        kernels.append(kernel)
    return kernels


def compute_gabor_energy(img_gray, kernels):
    """
    Áp dụng từng Gabor kernel, trả về vector năng lượng (1 giá trị / hướng).
    Energy = mean của magnitude response → đo mức độ nét cọ theo hướng đó.
    """
    energies = []
    for k in kernels:
        response  = cv2.filter2D(img_gray, cv2.CV_32F, k)
        energy    = float(np.mean(np.abs(response)))
        energies.append(energy)
    return np.array(energies)


def analyze_gabor_group(images, kernels, orientations):
    """Tính Gabor energy trung bình và std cho cả nhóm ảnh."""
    all_energies = []
    for _, img in images:
        e = compute_gabor_energy(img, kernels)
        all_energies.append(e)
    arr = np.array(all_energies)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_gabor_orientation(groups_data, orientations, output_dir):
    """
    Vẽ polar chart + bar chart năng lượng Gabor theo hướng cho 3 nhóm.
    Van Gogh đặc trưng: energy phân bố không đều, có peak rõ ở một số hướng.
    """
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Phân tích hướng nét cọ (Gabor Energy) — Đặc trưng Van Gogh", fontsize=12)

    # Bar chart truyền thống (dễ đọc hơn)
    ax_bar = fig.add_subplot(1, 2, 1)
    x = np.arange(len(orientations))
    width = 0.25
    colors_map = {
        "Tranh gốc Van Gogh": "#1D9E75",
        "Input (degraded)":   "#888780",
        "Output (restored)":  "#D85A30",
    }
    offsets = [-width, 0, width]
    for i, (label, mean_e, std_e) in enumerate(groups_data):
        norm_e = mean_e / (mean_e.max() + 1e-8)  # normalize để so sánh
        ax_bar.bar(x + offsets[i], norm_e, width, label=label,
                   color=colors_map.get(label, "#888"), alpha=0.8,
                   yerr=std_e / (mean_e.max()+1e-8), capsize=2, error_kw={"linewidth":0.5})

    ax_bar.set_xlabel("Hướng nét cọ (độ)")
    ax_bar.set_ylabel("Năng lượng Gabor (chuẩn hóa)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{o}°" for o in orientations], fontsize=8)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.spines[["top","right"]].set_visible(False)

    # Polar chart — trực quan hơn cho hướng
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")
    thetas = np.deg2rad(orientations + [orientations[0]])  # đóng vòng
    for label, mean_e, std_e in groups_data:
        norm_e = mean_e / (mean_e.max() + 1e-8)
        vals = np.append(norm_e, norm_e[0])
        ax_pol.plot(thetas, vals, label=label, color=colors_map.get(label, "#888"),
                    linewidth=1.8)
        ax_pol.fill(thetas, vals, alpha=0.08, color=colors_map.get(label, "#888"))
    ax_pol.set_title("Polar — phân bố hướng nét cọ", fontsize=9, pad=15)
    ax_pol.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gabor_orientation.jpg")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu: {out_path}")


# ── 2. Heatmap Gabor trên ảnh mẫu ────────────────────────────────────────────

def save_gabor_heatmap_sample(images, kernels, orientations, output_dir, n_samples=3):
    """
    Với n ảnh mẫu, vẽ heatmap response của 3 hướng Gabor đặc trưng.
    Giúp cô/người đọc thấy trực quan nét cọ được phát hiện ở đâu.
    """
    selected = images[:n_samples]
    n_show   = 3  # chỉ show 3 hướng đặc trưng nhất
    step     = max(1, len(orientations) // n_show)
    show_idx = list(range(0, len(orientations), step))[:n_show]

    fig, axes = plt.subplots(len(selected), n_show + 1,
                             figsize=(4 * (n_show + 1), 3.5 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    fig.suptitle("Heatmap Gabor — Van Gogh nét cọ theo hướng", fontsize=11)

    for row, (name, img) in enumerate(selected):
        axes[row][0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(f"Ảnh gốc\n{name[:20]}", fontsize=8)
        axes[row][0].axis("off")

        for col, ki in enumerate(show_idx, start=1):
            resp = cv2.filter2D(img, cv2.CV_32F, kernels[ki])
            axes[row][col].imshow(np.abs(resp), cmap="hot", vmin=0)
            axes[row][col].set_title(f"Hướng {orientations[ki]}°", fontsize=8)
            axes[row][col].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gabor_heatmap_sample.jpg")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu: {out_path}")


# ── 3. LBP — phân tích micro-texture ─────────────────────────────────────────

LBP_RADIUS    = 3
LBP_N_POINTS  = 8 * LBP_RADIUS
LBP_N_BINS    = LBP_N_POINTS + 2  # uniform LBP


def compute_lbp_histogram(img_gray_float):
    """
    Tính histogram LBP uniform cho một ảnh grayscale.
    Uniform LBP: chỉ đếm pattern có ≤2 lần chuyển 0↔1 → robust với rotation.
    """
    img_uint8 = (img_gray_float * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS,
                           range=(0, LBP_N_BINS), density=True)
    return hist


def compute_avg_lbp(images):
    hists = [compute_lbp_histogram(img) for _, img in images]
    return np.mean(hists, axis=0), np.std(hists, axis=0)


def plot_lbp_histograms(groups_data, output_dir):
    """
    So sánh LBP histogram của 3 nhóm.
    Nếu model học đúng texture Van Gogh, LBP của restored phải gần với original.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_title("So sánh LBP Texture Histogram — Đặc trưng nét cọ Van Gogh", fontsize=11)

    colors_map = {
        "Tranh gốc Van Gogh": "#1D9E75",
        "Input (degraded)":   "#888780",
        "Output (restored)":  "#D85A30",
    }
    ls_map = {
        "Tranh gốc Van Gogh": "-",
        "Input (degraded)":   "--",
        "Output (restored)":  ":",
    }
    x = np.arange(LBP_N_BINS)
    for label, mean_h, std_h in groups_data:
        c  = colors_map.get(label, "#888")
        ls = ls_map.get(label, "-")
        ax.plot(x, mean_h, label=label, color=c, linestyle=ls, linewidth=1.8)
        ax.fill_between(x, mean_h - std_h, mean_h + std_h, alpha=0.1, color=c)

    ax.set_xlabel("LBP Pattern index (Uniform LBP, R=3)", fontsize=9)
    ax.set_ylabel("Mật độ xác suất", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)

    ax.text(0.99, 0.97,
            "Đường 'Tranh gốc' và 'Restored' gần nhau → model đã học được texture",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#555", style="italic")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "lbp_histogram.jpg")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu: {out_path}")


# ── 4. Số liệu texture định lượng ────────────────────────────────────────────

def compute_texture_stats_single(img_gray_float, kernels):
    """
    Tổng hợp các số liệu texture cho 1 ảnh:
    - Gabor total energy (tổng năng lượng nét cọ)
    - Gabor dominant orientation (hướng cọ chính)
    - LBP entropy (đo độ phong phú texture)
    - Contrast (độ tương phản cục bộ)
    """
    energies  = compute_gabor_energy(img_gray_float, kernels)
    lbp_hist  = compute_lbp_histogram(img_gray_float)

    # Entropy của LBP distribution
    lbp_hist_norm = lbp_hist + 1e-10
    lbp_entropy   = -float(np.sum(lbp_hist_norm * np.log2(lbp_hist_norm + 1e-10)))

    # Độ tương phản cục bộ (std của gradient magnitude)
    gx = cv2.Sobel(img_gray_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    return {
        "gabor_total_energy":     round(float(energies.sum()), 5),
        "gabor_dominant_angle":   round(float(energies.argmax() * 30), 1),
        "gabor_energy_spread":    round(float(energies.std()), 5),
        "lbp_entropy":            round(lbp_entropy, 4),
        "gradient_mean":          round(float(grad_mag.mean()), 5),
        "gradient_std":           round(float(grad_mag.std()), 5),
    }


# ── Pipeline chính ────────────────────────────────────────────────────────────

def analyze(args):
    os.makedirs(args.output_dir, exist_ok=True)

    ORIENTATIONS = list(range(0, 180, 30))  # 0°, 30°, 60°, 90°, 120°, 150°
    kernels = build_gabor_kernels(ORIENTATIONS)

    print("\n[1] Đang đọc ảnh (grayscale)...")
    orig_imgs = load_gray_images(args.original_dir) if args.original_dir else []
    deg_imgs  = load_gray_images(args.degraded_dir) if args.degraded_dir else []
    rest_imgs = load_gray_images(args.restored_dir) if args.restored_dir else []

    if not orig_imgs:
        raise ValueError("Cần ít nhất --original_dir!")

    print("\n[2] Tính Gabor energy theo hướng...")
    gabor_groups = []
    if orig_imgs:
        m, s = analyze_gabor_group(orig_imgs, kernels, ORIENTATIONS)
        gabor_groups.append(("Tranh gốc Van Gogh", m, s))
    if deg_imgs:
        m, s = analyze_gabor_group(deg_imgs, kernels, ORIENTATIONS)
        gabor_groups.append(("Input (degraded)", m, s))
    if rest_imgs:
        m, s = analyze_gabor_group(rest_imgs, kernels, ORIENTATIONS)
        gabor_groups.append(("Output (restored)", m, s))

    plot_gabor_orientation(gabor_groups, ORIENTATIONS, args.output_dir)

    print("\n[3] Vẽ heatmap Gabor trên ảnh mẫu...")
    save_gabor_heatmap_sample(orig_imgs, kernels, ORIENTATIONS, args.output_dir)

    print("\n[4] Tính LBP histogram...")
    lbp_groups = []
    if orig_imgs:
        m, s = compute_avg_lbp(orig_imgs)
        lbp_groups.append(("Tranh gốc Van Gogh", m, s))
    if deg_imgs:
        m, s = compute_avg_lbp(deg_imgs)
        lbp_groups.append(("Input (degraded)", m, s))
    if rest_imgs:
        m, s = compute_avg_lbp(rest_imgs)
        lbp_groups.append(("Output (restored)", m, s))

    plot_lbp_histograms(lbp_groups, args.output_dir)

    print("\n[5] Tính số liệu texture chi tiết từng ảnh...")
    records = []
    for grp_label, imgs in [("original", orig_imgs), ("degraded", deg_imgs), ("restored", rest_imgs)]:
        for name, img in imgs:
            stats = compute_texture_stats_single(img, kernels)
            stats["image"] = name
            stats["group"] = grp_label
            records.append(stats)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, "texture_stats.csv"), index=False)

    # Tóm tắt
    print("\n" + "="*50)
    print("KẾT QUẢ PHÂN TÍCH TEXTURE")
    print("="*50)
    summary_lines = ["PHÂN TÍCH TEXTURE & BRUSHSTROKE — VAN GOGH\n" + "="*50 + "\n"]
    for grp in df["group"].unique():
        sub = df[df["group"] == grp]
        print(f"\n  [{grp.upper()}]")
        summary_lines.append(f"\n[{grp.upper()}]")
        for col in ["gabor_total_energy", "lbp_entropy", "gradient_mean"]:
            line = f"  {col:28s}: {sub[col].mean():.5f} ± {sub[col].std():.5f}"
            print(line); summary_lines.append(line)

    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nHoàn thành! Kết quả tại: {args.output_dir}/")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích texture & brushstroke Van Gogh")
    parser.add_argument("--original_dir", default="./vangogh_color")
    parser.add_argument("--degraded_dir", default=None)
    parser.add_argument("--restored_dir", default=None)
    parser.add_argument("--output_dir",   default="./texture_analysis")
    args = parser.parse_args()
    analyze(args)

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
# Chỉ tranh gốc:
#   python A2_texture_analysis.py --original_dir ./vangogh_color
#
# So sánh đầy đủ:
#   python A2_texture_analysis.py \
#       --original_dir ./vangogh_color \
#       --degraded_dir ./dataset_restore/test_inputs \
#       --restored_dir ./eval_results/restored
