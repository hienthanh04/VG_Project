"""
RUN ALL — Chạy tất cả 4 runs tự động + tổng hợp bảng so sánh
=============================================================
Chạy tuần tự 4 experiments:
  Run 1: blur_only         (mờ, còn màu)
  Run 2: grayscale_noise   (mất màu + nhiễu) — đã có, skip nếu muốn
  Run 3: blur_noise        (mờ + nhiễu, còn màu)
  Run 4: gray_blur_noise   (mất màu + mờ + nhiễu) ← khó nhất

Sau đó tổng hợp kết quả thành bảng so sánh cho báo cáo.

Cách dùng:
    # Chạy tất cả 4 runs (mỗi run ~1-2 giờ tùy GPU):
    python run_all_experiments.py --input_dir ./vangogh_color --epochs 100

    # Chỉ chạy run mới (3 và 4), bỏ qua run đã có:
    python run_all_experiments.py --input_dir ./vangogh_color --skip_existing

    # Chỉ tổng hợp bảng so sánh (nếu đã chạy xong hết):
    python run_all_experiments.py --summary_only
"""

import os
import subprocess
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Cấu hình 4 runs ──────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "run_id":    "run1_blur_only",
        "deg_type":  "blur_only",
        "label":     "Run 1 — blur_only\n(mờ, còn màu)",
        "color":     "#4A90D9",
        "desc":      "Ảnh mờ, vẫn còn màu — bài toán dễ nhất",
    },
    {
        "run_id":    "run2_grayscale_noise",
        "deg_type":  "grayscale_noise",
        "label":     "Run 2 — grayscale_noise\n(mất màu + nhiễu)",
        "color":     "#888780",
        "desc":      "Mất màu + nhiễu — đã có kết quả",
    },
    {
        "run_id":    "run3_blur_noise",
        "deg_type":  "blur_noise",
        "label":     "Run 3 — blur_noise\n(mờ + nhiễu, còn màu)",
        "color":     "#E8A838",
        "desc":      "Mờ + nhiễu, còn màu — mức trung bình",
    },
    {
        "run_id":    "run4_gray_blur_noise",
        "deg_type":  "gray_blur_noise",
        "label":     "Run 4 — gray_blur_noise\n(mất màu + mờ + nhiễu)",
        "color":     "#D85A30",
        "desc":      "Bộ 3 suy giảm — khó nhất, thực tế nhất",
    },
]

# ── Hàm chạy từng bước ────────────────────────────────────────────────────────

def run_cmd(cmd, desc):
    """Chạy command và in output realtime."""
    print(f"\n{'─'*60}")
    print(f"▶ {desc}")
    print(f"  {' '.join(cmd)}")
    print('─'*60)
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_experiment(exp, args):
    run_id   = exp["run_id"]
    deg_type = exp["deg_type"]

    data_dir  = f"./datasets/{run_id}"
    out_dir   = f"./outputs/{run_id}"
    eval_dir  = f"./evals/{run_id}"
    ckpt_path = f"{out_dir}/checkpoints/generator_best.pth"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {run_id}  ({exp['desc']})")
    print(f"{'='*60}")

    # ── Bước 1: Tạo dataset ──
    if args.skip_existing and os.path.exists(os.path.join(data_dir, "train")):
        print(f"  [SKIP] Dataset đã tồn tại: {data_dir}")
    else:
        run_cmd([
            "python", "step1_patch_blur.py",
            "--input_dir",   args.input_dir,
            "--output_dir",  data_dir,
            "--deg_type",    deg_type,
            "--train_ratio", str(args.train_ratio),
            "--img_size",    str(args.img_size),
            "--seed",        str(args.seed),
        ], f"[{run_id}] Tạo dataset ({deg_type})")

    # ── Bước 2: Train pix2pix ──
    if args.skip_existing and os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint đã tồn tại: {ckpt_path}")
    else:
        run_cmd([
            "python", "step2_train_pix2pix.py",
            "--data_dir",   data_dir,
            "--output_dir", out_dir,
            "--epochs",     str(args.epochs),
            "--batch_size", str(args.batch_size),
        ], f"[{run_id}] Train pix2pix ({args.epochs} epochs)")

    # ── Bước 3: Evaluate ──
    if args.skip_existing and os.path.exists(os.path.join(eval_dir, "metrics.csv")):
        print(f"  [SKIP] Eval đã tồn tại: {eval_dir}")
    else:
        run_cmd([
            "python", "step3_evaluate.py",
            "--data_dir",   os.path.join(data_dir, "test"),
            "--model_path", ckpt_path,
            "--output_dir", eval_dir,
            "--img_size",   str(args.img_size),
        ], f"[{run_id}] Đánh giá kết quả")

    print(f"\n  ✓ Hoàn thành: {run_id}")
    return eval_dir


# ── Tổng hợp bảng so sánh ────────────────────────────────────────────────────

def build_comparison_table(experiments, output_dir="./comparison"):
    """
    Đọc metrics.csv của tất cả runs và tổng hợp thành bảng so sánh.
    Xuất: CSV + biểu đồ bar chart cho báo cáo.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for exp in experiments:
        csv_path = f"./evals/{exp['run_id']}/metrics.csv"
        if not os.path.exists(csv_path):
            print(f"  [WARNING] Chưa có kết quả: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        records.append({
            "Run":                  exp["run_id"].replace("_", " "),
            "Degradation":          exp["deg_type"],
            "Mô tả":                exp["desc"],
            # Degraded (baseline)
            "SSIM (degraded)":      round(df["ssim_degraded"].mean(),  4),
            "PSNR (degraded)":      round(df["psnr_degraded"].mean(),  2),
            "EdgeIoU (degraded)":   round(df["edge_iou_degraded"].mean(), 4),
            # Restored (model output)
            "SSIM (restored)":      round(df["ssim_restored"].mean(),  4),
            "PSNR (restored)":      round(df["psnr_restored"].mean(),  2),
            "EdgeIoU (restored)":   round(df["edge_iou_restored"].mean(), 4),
            # Cải thiện
            "ΔSSIM":                round(df["ssim_delta"].mean(),     4),
            "ΔPSNR":                round(df["psnr_delta"].mean(),     2),
            "ΔEdgeIoU":             round(df["edge_iou_delta"].mean(), 4),
            # Phân loại
            "Ảnh tốt":              int((df["ssim_restored"] > 0.8).sum()),
            "Ảnh trung bình":       int(((df["ssim_restored"] >= 0.6) & (df["ssim_restored"] <= 0.8)).sum()),
            "Ảnh fail":             int((df["ssim_restored"] < 0.6).sum()),
            "_color":               exp["color"],   # dùng cho chart, ẩn khi in
        })

    if not records:
        print("Chưa có kết quả nào để tổng hợp.")
        return

    result_df = pd.DataFrame(records)

    # Lưu CSV (bỏ cột _color)
    display_df = result_df.drop(columns=["_color"])
    csv_out = os.path.join(output_dir, "comparison_table.csv")
    display_df.to_csv(csv_out, index=False)
    print(f"\nBảng so sánh lưu tại: {csv_out}")

    # ── Vẽ biểu đồ bar chart ──
    _plot_comparison_chart(result_df, output_dir)
    _plot_improvement_chart(result_df, output_dir)

    # In ra màn hình
    print("\n" + "="*70)
    print("BẢNG SO SÁNH KẾT QUẢ 4 RUNS")
    print("="*70)
    cols_show = ["Run", "SSIM (degraded)", "SSIM (restored)", "ΔSSIM",
                 "PSNR (degraded)", "PSNR (restored)", "ΔPSNR",
                 "Ảnh tốt", "Ảnh trung bình", "Ảnh fail"]
    print(display_df[cols_show].to_string(index=False))

    return result_df


def _plot_comparison_chart(df, output_dir):
    """Bar chart so sánh SSIM và PSNR của degraded vs restored cho 4 runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("So sánh kết quả 4 runs — Degraded vs Restored", fontsize=12)

    runs   = [r.split(" ")[1] for r in df["Run"]]  # tên ngắn
    colors = df["_color"].tolist()
    x      = range(len(df))
    w      = 0.35

    for ax, metric_base, metric_rest, title in [
        (axes[0], "SSIM (degraded)",  "SSIM (restored)",  "SSIM"),
        (axes[1], "PSNR (degraded)",  "PSNR (restored)",  "PSNR (dB)"),
    ]:
        bars1 = ax.bar([i - w/2 for i in x], df[metric_base],
                       w, label="Degraded (input)", color="#CCCCCC", edgecolor="white")
        bars2 = ax.bar([i + w/2 for i in x], df[metric_rest],
                       w, label="Restored (output)", color=colors, edgecolor="white", alpha=0.85)

        # Giá trị trên bar
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(runs, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, "comparison_ssim_psnr.jpg")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Biểu đồ so sánh: {out}")


def _plot_improvement_chart(df, output_dir):
    """Bar chart cải thiện (Δ) của 4 runs — chứng minh model luôn cải thiện."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_title("Mức độ cải thiện sau phục hồi (Δ = Restored − Degraded)\n"
                 "Thanh dương = model cải thiện so với input", fontsize=11)

    runs   = [r.split(" ")[1] for r in df["Run"]]
    colors = df["_color"].tolist()
    x      = range(len(df))
    w      = 0.25

    for i, (col, label, offset) in enumerate([
        ("ΔSSIM",    "ΔSSIM",         -w),
        ("ΔPSNR",    "ΔPSNR/10",       0),   # chia 10 để cùng thang
        ("ΔEdgeIoU", "ΔEdge IoU",      w),
    ]):
        vals = df[col] / 10 if col == "ΔPSNR" else df[col]
        bars = ax.bar([xi + offset for xi in x], vals, w,
                      label=label, alpha=0.82, edgecolor="white",
                      color=[c if v >= 0 else "#FF4444"
                             for c, v in zip(colors, vals)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001 if v >= 0 else bar.get_height() - 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel("Giá trị cải thiện (ΔPSNR đã chia 10)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, "comparison_improvement.jpg")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Biểu đồ cải thiện: {out}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy tất cả 4 experiments tự động")
    parser.add_argument("--input_dir",     default="./vangogh_color")
    parser.add_argument("--epochs",        default=100, type=int)
    parser.add_argument("--batch_size",    default=4,   type=int)
    parser.add_argument("--train_ratio",   default=0.8, type=float)
    parser.add_argument("--img_size",      default=256, type=int)
    parser.add_argument("--seed",          default=42,  type=int)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Bỏ qua bước đã có kết quả (tiếp tục từ chỗ dở)")
    parser.add_argument("--summary_only",  action="store_true",
                        help="Chỉ tổng hợp bảng so sánh, không train")
    parser.add_argument("--runs",          default="1,2,3,4",
                        help="Chọn runs muốn chạy, ví dụ: 1,3,4")
    args = parser.parse_args()

    # Chọn runs muốn chạy
    run_indices = [int(r)-1 for r in args.runs.split(",")]
    selected    = [EXPERIMENTS[i] for i in run_indices if i < len(EXPERIMENTS)]

    if not args.summary_only:
        print(f"\nSẽ chạy {len(selected)} experiments:")
        for exp in selected:
            print(f"  - {exp['run_id']} ({exp['desc']})")

        for exp in selected:
            run_experiment(exp, args)

    # Tổng hợp bảng so sánh
    print("\n" + "="*60)
    print("TỔNG HỢP KẾT QUẢ")
    print("="*60)
    build_comparison_table(EXPERIMENTS, output_dir="./comparison")
    print("\nHoàn thành tất cả! Output:")
    print("  datasets/run*/     — dataset từng run")
    print("  outputs/run*/      — checkpoints + samples")
    print("  evals/run*/        — metrics + visuals")
    print("  comparison/        — bảng so sánh + biểu đồ")

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
# Chạy tất cả 4 runs:
#   python run_all_experiments.py --input_dir ./vangogh_color --epochs 100
#
# Chỉ chạy run 3 và 4 (run 1, 2 đã có):
#   python run_all_experiments.py --runs 3,4 --skip_existing
#
# Chỉ tổng hợp bảng (đã train xong hết):
#   python run_all_experiments.py --summary_only
