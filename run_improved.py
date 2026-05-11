"""
CHẠY CONFIG MỚI + CHỌN ẢNH SO SÁNH
=====================================
Việc 1: Tạo dataset mới (params nhẹ hơn) + train 150 epoch
Việc 2: Chọn ảnh đại diện từ config cũ để giữ lại so sánh

Cách dùng:
    # Chạy cả 2 việc:
    python run_improved.py --input_dir ./vangogh_color

    # Chỉ tạo dataset + train (không chọn ảnh cũ):
    python run_improved.py --input_dir ./vangogh_color --skip_select

    # Chỉ chọn ảnh từ config cũ (đã train rồi):
    python run_improved.py --select_only
"""

import os
import subprocess
import argparse
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: Tạo dataset mới với params nhẹ hơn
# ─────────────────────────────────────────────────────────────────────────────

def patch_degradation_params():
    """
    Thêm 2 hàm degradation với params nhẹ hơn vào step1_patch_blur.py.
    Nếu file đã có rồi thì bỏ qua.
    """
    patch_code = '''

# ── Config mới — params nhẹ hơn ─────────────────────────────────────────────

def degrade_gray_blur_noise_light(img):
    """gray_blur_noise nhẹ: blur_ksize=9, noise_sigma=10"""
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    noise = np.random.normal(0, 10, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_blur_noise_light(img):
    """blur_noise nhẹ: blur_ksize=9, noise_sigma=10 (còn màu)"""
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    noise = np.random.normal(0, 10, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Thêm vào map
DEGRADATION_MAP_FULL["gray_blur_noise_light"] = degrade_gray_blur_noise_light
DEGRADATION_MAP_FULL["blur_noise_light"]      = degrade_blur_noise_light
'''
    patch_path = "step1_patch_blur.py"
    if not os.path.exists(patch_path):
        print(f"[ERROR] Không tìm thấy {patch_path}")
        return False

    with open(patch_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "gray_blur_noise_light" in content:
        print("  [SKIP] Patch đã có trong file rồi.")
        return True

    with open(patch_path, "a", encoding="utf-8") as f:
        f.write(patch_code)
    print(f"  Đã thêm 2 hàm mới vào {patch_path}")
    return True


def run_cmd(cmd, desc):
    print(f"\n{'─'*56}")
    print(f"▶ {desc}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print('─'*56)
    subprocess.run([str(c) for c in cmd], check=True)


def train_new_config(args):
    """Train với params nhẹ hơn và 150 epochs."""

    for deg_type, run_name in [
        ("gray_blur_noise_light", "run_gray_bn_light"),
        ("blur_noise_light",      "run_blur_n_light"),
    ]:
        data_dir = f"./datasets/{run_name}"
        out_dir  = f"./outputs/{run_name}"
        eval_dir = f"./evals/{run_name}"

        print(f"\n{'='*56}")
        print(f"  Config: {deg_type} | epochs={args.epochs}")
        print(f"{'='*56}")

        # Bước 1: Dataset
        run_cmd([
            "python", "step1_patch_blur.py",
            "--input_dir",  args.input_dir,
            "--output_dir", data_dir,
            "--deg_type",   deg_type,
            "--train_ratio","0.8",
            "--img_size",   "256",
            "--seed",       "42",
        ], f"Tạo dataset: {deg_type} (blur=9, noise=10)")

        # Bước 2: Train
        run_cmd([
            "python", "step2_train_pix2pix.py",
            "--data_dir",   data_dir,
            "--output_dir", out_dir,
            "--epochs",     str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--save_every", "25",
        ], f"Train pix2pix — {args.epochs} epochs")

        # Bước 3: Evaluate
        run_cmd([
            "python", "step3_evaluate.py",
            "--data_dir",   os.path.join(data_dir, "test"),
            "--model_path", f"{out_dir}/checkpoints/generator_best.pth",
            "--output_dir", eval_dir,
        ], "Đánh giá kết quả")

    print("\n✓ Xong training config mới!")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: Chọn ảnh đại diện từ config cũ để giữ lại so sánh
# ─────────────────────────────────────────────────────────────────────────────

def select_representative_images(eval_dir_old, output_dir, n_best=3, n_mid=2):
    """
    Từ kết quả cũ (ksize=13, sigma=20), chọn:
    - n_best ảnh SSIM cao nhất  → "trường hợp model làm tốt nhất"
    - n_mid  ảnh SSIM trung bình → "trường hợp điển hình"

    Sao chép ảnh so sánh [degraded|restored|original] vào thư mục riêng
    để dễ dàng chèn vào báo cáo.
    """
    csv_path = os.path.join(eval_dir_old, "metrics.csv")
    vis_dir  = os.path.join(eval_dir_old, "visuals")

    if not os.path.exists(csv_path):
        print(f"[ERROR] Không tìm thấy: {csv_path}")
        return

    df = pd.read_csv(csv_path).sort_values("ssim_restored", ascending=False)

    os.makedirs(output_dir, exist_ok=True)

    selected = {}

    # Top n_best
    best_rows = df.head(n_best)
    for _, row in best_rows.iterrows():
        selected[row["image"]] = "best"

    # Middle n_mid — lấy quanh median
    mid_idx = len(df) // 2
    mid_rows = df.iloc[mid_idx - 1 : mid_idx + n_mid]
    for _, row in mid_rows.iterrows():
        selected[row["image"]] = "mid"

    print(f"\nChọn {len(selected)} ảnh đại diện từ config cũ:")
    copied = 0
    for img_name, category in selected.items():
        # Tìm file ảnh so sánh trong visuals/
        vis_candidates = list(Path(vis_dir).glob(f"{img_name}*"))
        if not vis_candidates:
            print(f"  [SKIP] Không tìm thấy visual: {img_name}")
            continue

        src = vis_candidates[0]
        ssim_val = df[df["image"] == img_name]["ssim_restored"].values[0]
        dst_name = f"{category}_{img_name}_ssim{ssim_val:.3f}.jpg"
        dst = os.path.join(output_dir, dst_name)
        shutil.copy(src, dst)
        print(f"  [{category.upper()}] {img_name} — SSIM={ssim_val:.4f} → {dst_name}")
        copied += 1

    # Tạo summary CSV của ảnh được chọn
    sel_df = df[df["image"].isin(selected.keys())][
        ["image", "ssim_degraded", "ssim_restored", "ssim_delta",
         "psnr_degraded", "psnr_restored", "psnr_delta"]
    ].copy()
    sel_df["category"] = sel_df["image"].map(selected)
    sel_df.to_csv(os.path.join(output_dir, "selected_metrics.csv"), index=False)

    print(f"\n✓ Đã sao chép {copied} ảnh vào: {output_dir}/")
    print(f"  selected_metrics.csv — số liệu của ảnh được chọn")

    # Tạo figure ghép ảnh được chọn
    _make_selected_figure(output_dir, selected)


def _make_selected_figure(image_dir, selected):
    """Ghép tất cả ảnh được chọn thành 1 figure để chèn vào báo cáo."""
    img_files = sorted(Path(image_dir).glob("*.jpg"))
    img_files = [f for f in img_files if not f.name.startswith("figure")]

    if not img_files:
        return

    n = len(img_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("Ảnh đại diện từ config cũ (blur_ksize=13, noise_sigma=20)\n"
                 "Dùng để so sánh với config mới trong báo cáo", fontsize=11)

    for ax, img_path in zip(axes, img_files):
        img = plt.imread(str(img_path))
        ax.imshow(img)
        # Lấy tên từ filename
        parts = img_path.stem.split("_", 1)
        category = parts[0].upper() if parts else ""
        ax.set_title(f"[{category}]\n{img_path.stem[len(category)+1:20]}...",
                     fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(image_dir, "figure_selected.jpg")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Figure tổng hợp: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: Bảng so sánh config cũ vs mới
# ─────────────────────────────────────────────────────────────────────────────

def compare_old_vs_new():
    """
    So sánh kết quả config cũ (sigma=20) vs config mới (sigma=10)
    cho cả gray_blur_noise và blur_noise.
    """
    configs = [
        # (label, eval_dir)
        ("gray_blur_noise\n(ksize=13, σ=20)", "./eval_gray_blur_noise"),
        ("gray_blur_noise_light\n(ksize=9, σ=10)",  "./evals/run_gray_bn_light"),
        ("blur_noise\n(ksize=13, σ=20)",      "./eval_blur_noise"),
        ("blur_noise_light\n(ksize=9, σ=10)", "./evals/run_blur_n_light"),
    ]

    records = []
    for label, eval_dir in configs:
        csv_path = os.path.join(eval_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] Chưa có: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        records.append({
            "Config": label,
            "SSIM restored": round(df["ssim_restored"].mean(), 4),
            "PSNR restored": round(df["psnr_restored"].mean(), 2),
            "ΔEdge IoU":     round(df["edge_iou_delta"].mean(), 4),
            "ΔSSIM":         round(df["ssim_delta"].mean(), 4),
            "Ảnh tốt":       int((df["ssim_restored"] > 0.8).sum()),
            "Ảnh tb":        int(((df["ssim_restored"] >= 0.6) & (df["ssim_restored"] <= 0.8)).sum()),
            "Ảnh fail":      int((df["ssim_restored"] < 0.6).sum()),
        })

    if not records:
        print("Chưa có kết quả để so sánh.")
        return

    result_df = pd.DataFrame(records)
    os.makedirs("./comparison", exist_ok=True)
    result_df.to_csv("./comparison/old_vs_new.csv", index=False)

    print("\n" + "="*60)
    print("SO SÁNH CONFIG CŨ VS MỚI")
    print("="*60)
    print(result_df.to_string(index=False))

    # Biểu đồ
    _plot_old_vs_new(result_df)


def _plot_old_vs_new(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("So sánh config cũ (σ=20) vs mới (σ=10)\n"
                 "Xanh = gray_blur_noise, Cam = blur_noise", fontsize=11)

    colors = ["#5B7FA6", "#1D9E75", "#D8854A", "#E8C338"]
    metrics = [
        ("SSIM restored", "SSIM (restored)", "Cao hơn = tốt hơn"),
        ("ΔSSIM",         "ΔSSIM cải thiện", "Cao hơn = cải thiện nhiều hơn"),
        ("ΔEdge IoU",     "ΔEdge IoU",       "Cao hơn = phục hồi nét tốt hơn"),
    ]

    for ax, (col, title, note) in zip(axes, metrics):
        bars = ax.bar(range(len(df)), df[col],
                      color=colors[:len(df)], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(df[col].max()) * 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(
            [c.split("\n")[0] for c in df["Config"]],
            fontsize=7, rotation=15, ha="right"
        )
        ax.set_ylabel(note, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("./comparison/old_vs_new_chart.jpg", dpi=130, bbox_inches="tight")
    plt.close()
    print("\n  Biểu đồ so sánh: ./comparison/old_vs_new_chart.jpg")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",    default="./vangogh_color")
    parser.add_argument("--epochs",       default=150, type=int)
    parser.add_argument("--batch_size",   default=4,   type=int)
    parser.add_argument("--skip_select",  action="store_true",
                        help="Bỏ qua bước chọn ảnh từ config cũ")
    parser.add_argument("--select_only",  action="store_true",
                        help="Chỉ chọn ảnh từ config cũ, không train")
    parser.add_argument("--compare_only", action="store_true",
                        help="Chỉ tạo bảng so sánh cũ vs mới")
    # Thư mục eval của config cũ
    parser.add_argument("--old_gray_bn",  default="./eval_gray_blur_noise",
                        help="Thư mục eval của gray_blur_noise cũ")
    parser.add_argument("--old_blur_n",   default="./eval_blur_noise",
                        help="Thư mục eval của blur_noise cũ")
    args = parser.parse_args()

    if args.compare_only:
        compare_old_vs_new()

    elif args.select_only:
        print("\n[Bước 2] Chọn ảnh đại diện từ config cũ...")
        select_representative_images(
            args.old_gray_bn,
            "./selected_images/gray_blur_noise_heavy",
            n_best=3, n_mid=2
        )
        select_representative_images(
            args.old_blur_n,
            "./selected_images/blur_noise_heavy",
            n_best=3, n_mid=2
        )

    else:
        # Bước 1: Patch thêm hàm mới
        print("\n[Bước 1] Thêm params mới vào step1_patch_blur.py...")
        patch_degradation_params()

        # Bước 2: Chọn ảnh từ config cũ (làm trước khi train để có ngay)
        if not args.skip_select:
            print("\n[Bước 2] Chọn ảnh đại diện từ config cũ để giữ lại...")
            select_representative_images(
                args.old_gray_bn,
                "./selected_images/gray_blur_noise_heavy",
                n_best=3, n_mid=2
            )
            select_representative_images(
                args.old_blur_n,
                "./selected_images/blur_noise_heavy",
                n_best=3, n_mid=2
            )

        # Bước 3: Train config mới
        print(f"\n[Bước 3] Train config mới (blur=9, noise=10, epochs={args.epochs})...")
        train_new_config(args)

        # Bước 4: So sánh
        print("\n[Bước 4] Tổng hợp so sánh cũ vs mới...")
        compare_old_vs_new()

        print("\n" + "="*56)
        print("HOÀN THÀNH! Output:")
        print("  selected_images/   — ảnh đại diện từ config cũ")
        print("  datasets/run_*     — dataset config mới")
        print("  outputs/run_*      — checkpoints")
        print("  evals/run_*        — kết quả đánh giá")
        print("  comparison/        — bảng + biểu đồ so sánh")

# ── Ví dụ chạy ───────────────────────────────────────────────────────────────
# Chạy đầy đủ (chọn ảnh cũ + train mới + so sánh):
#   python run_improved.py --input_dir ./vangogh_color --epochs 150
#
# Chỉ chọn ảnh từ config cũ để giữ lại:
#   python run_improved.py --select_only
#
# Sau khi train xong, chỉ tạo bảng so sánh:
#   python run_improved.py --compare_only
