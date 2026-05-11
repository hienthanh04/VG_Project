"""
BƯỚC 3: ĐÁNH GIÁ KẾT QUẢ PHỤC HỒI
====================================
So sánh 3 bộ ảnh: [degraded | restored | original]
Tính SSIM, PSNR, Edge IoU cho từng cặp.
Sinh báo cáo CSV + ảnh so sánh trực quan.

Cách dùng:
    python step3_evaluate.py --data_dir   ./dataset_restore/test \
                             --model_path ./output_pix2pix/checkpoints/generator_best.pth \
                             --output_dir ./eval_results

Yêu cầu:
    pip install torch torchvision scikit-image opencv-python pandas matplotlib
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import Generator từ step2
import sys
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_pix2pix import Generator

# ── Hàm tính metric ──────────────────────────────────────────────────────────

def calc_ssim(img_a, img_b):
    """SSIM trên ảnh uint8 RGB."""
    return ssim(img_a, img_b, channel_axis=2, data_range=255)

def calc_psnr(img_a, img_b):
    """PSNR trên ảnh uint8 RGB."""
    return psnr(img_a, img_b, data_range=255)

def calc_edge_iou(img_a, img_b, threshold=50):
    """
    Edge IoU: tỉ lệ giao/hợp của vùng cạnh giữa 2 ảnh.
    Cạnh được phát hiện bằng Canny.
    """
    def get_edges(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray, threshold, threshold * 2) > 0

    edges_a = get_edges(img_a)
    edges_b = get_edges(img_b)
    intersection = np.logical_and(edges_a, edges_b).sum()
    union        = np.logical_or(edges_a, edges_b).sum()
    return float(intersection) / float(union + 1e-8)

# ── Sinh ảnh ra từ model ──────────────────────────────────────────────────────

def load_model(model_path, device):
    G = Generator().to(device)
    state = torch.load(model_path, map_location=device)
    # Hỗ trợ cả 2 dạng checkpoint
    if "G" in state:
        state = state["G"]
    G.load_state_dict(state)
    G.eval()
    return G

def infer_single(G, degraded_np, img_size, device):
    """
    Nhận ảnh numpy (H,W,3) uint8, trả về ảnh phục hồi numpy (H,W,3) uint8.
    """
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    x = transform(degraded_np).unsqueeze(0).to(device)
    with torch.no_grad():
        out = G(x).squeeze(0).cpu()
    out = (out * 0.5 + 0.5).clamp(0, 1)
    out_np = (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.resize(out_np, (degraded_np.shape[1], degraded_np.shape[0]))

# ── Vẽ ảnh so sánh 3 chiều ───────────────────────────────────────────────────

def save_comparison(degraded, restored, original, path, metrics):
    """Lưu ảnh so sánh [Degraded | Restored | Original] kèm metric."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input (degraded)", "Output (restored)", "Ground truth (original)"]
    imgs   = [degraded, restored, original]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    subtitle = (
        f"SSIM: {metrics['ssim_restored']:.4f}  |  "
        f"PSNR: {metrics['psnr_restored']:.2f} dB  |  "
        f"Edge IoU: {metrics['edge_iou_restored']:.4f}"
    )
    fig.suptitle(subtitle, fontsize=10, y=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()

# ── Pipeline đánh giá ────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    G = load_model(args.model_path, device)
    print(f"Model loaded: {args.model_path}")

    exts  = {".jpg", ".jpeg", ".png"}
    pairs = sorted([p for p in Path(args.data_dir).glob("*") if p.suffix.lower() in exts])
    if not pairs:
        raise FileNotFoundError(f"Không có ảnh test trong: {args.data_dir}")
    print(f"Đánh giá {len(pairs)} ảnh...")

    vis_dir = os.path.join(args.output_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    records = []
    for pair_path in pairs:
        pair_img = cv2.imread(str(pair_path))
        if pair_img is None:
            continue
        pair_img = cv2.cvtColor(pair_img, cv2.COLOR_BGR2RGB)

        h, w = pair_img.shape[:2]
        hw = w // 2
        degraded = pair_img[:, :hw, :]
        original = pair_img[:, hw:, :]

        # Resize về img_size để đồng nhất
        sz = args.img_size
        degraded = cv2.resize(degraded, (sz, sz))
        original = cv2.resize(original, (sz, sz))

        # Sinh ảnh phục hồi
        restored = infer_single(G, degraded, sz, device)

        # Tính metric (restored vs original)
        m_ssim_r  = calc_ssim(restored,  original)
        m_psnr_r  = calc_psnr(restored,  original)
        m_eiou_r  = calc_edge_iou(restored, original)

        # Tính metric (degraded vs original) làm baseline
        m_ssim_d  = calc_ssim(degraded,  original)
        m_psnr_d  = calc_psnr(degraded,  original)
        m_eiou_d  = calc_edge_iou(degraded, original)

        record = {
            "image":            pair_path.stem,
            "ssim_degraded":    round(m_ssim_d, 4),
            "ssim_restored":    round(m_ssim_r, 4),
            "ssim_delta":       round(m_ssim_r - m_ssim_d, 4),
            "psnr_degraded":    round(m_psnr_d, 2),
            "psnr_restored":    round(m_psnr_r, 2),
            "psnr_delta":       round(m_psnr_r - m_psnr_d, 2),
            "edge_iou_degraded": round(m_eiou_d, 4),
            "edge_iou_restored": round(m_eiou_r, 4),
            "edge_iou_delta":    round(m_eiou_r - m_eiou_d, 4),
        }
        records.append(record)

        # Lưu ảnh so sánh
        vis_path = os.path.join(vis_dir, pair_path.stem + "_compare.jpg")
        save_comparison(degraded, restored, original, vis_path, record)

    # ── Tổng hợp kết quả ──
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    # In bảng tóm tắt
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")
    print("="*60)
    summary_cols = ["ssim_degraded", "ssim_restored", "ssim_delta",
                    "psnr_degraded", "psnr_restored", "psnr_delta",
                    "edge_iou_degraded", "edge_iou_restored", "edge_iou_delta"]
    summary = df[summary_cols].agg(["mean", "std", "min", "max"]).round(4)
    print(summary.to_string())

    # Phân loại ảnh tốt / trung bình / fail (theo SSIM restored)
    df["quality"] = pd.cut(
        df["ssim_restored"],
        bins=[0, 0.6, 0.8, 1.0],
        labels=["fail", "trung bình", "tốt"]
    )
    print("\nPhân loại chất lượng ảnh phục hồi:")
    print(df["quality"].value_counts().to_string())

    # Lưu báo cáo tóm tắt
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ ĐÁNH GIÁ PIX2PIX — PHỤC HỒI TRANH VAN GOGH\n")
        f.write("="*60 + "\n")
        f.write(summary.to_string() + "\n\n")
        f.write("Phân loại chất lượng:\n")
        f.write(df["quality"].value_counts().to_string() + "\n\n")
        f.write("Top 5 ảnh tốt nhất (SSIM cao nhất):\n")
        f.write(df.nlargest(5, "ssim_restored")[["image","ssim_restored","psnr_restored"]].to_string(index=False) + "\n\n")
        f.write("Top 5 ảnh kém nhất (SSIM thấp nhất):\n")
        f.write(df.nsmallest(5, "ssim_restored")[["image","ssim_restored","psnr_restored"]].to_string(index=False) + "\n")

    # Vẽ biểu đồ so sánh SSIM degraded vs restored
    _plot_ssim_comparison(df, args.output_dir)

    print(f"\nKết quả lưu tại: {args.output_dir}/")
    print(f"  metrics.csv      — số liệu từng ảnh")
    print(f"  summary.txt      — báo cáo tóm tắt")
    print(f"  visuals/         — ảnh so sánh 3 chiều")
    print(f"  ssim_comparison.jpg — biểu đồ SSIM")


def _plot_ssim_comparison(df, output_dir):
    """Vẽ scatter plot SSIM degraded → restored để thấy cải thiện."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["ssim_degraded"], df["ssim_restored"], alpha=0.6, s=40, c="#1D9E75")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Không đổi")
    ax.set_xlabel("SSIM (degraded vs original)", fontsize=11)
    ax.set_ylabel("SSIM (restored vs original)", fontsize=11)
    ax.set_title("Cải thiện SSIM sau phục hồi (điểm trên đường chéo = tốt hơn)", fontsize=11)
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ssim_comparison.jpg"), dpi=120)
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình pix2pix")
    parser.add_argument("--data_dir",   default="./dataset_restore/test",
                        help="Thư mục ảnh test (cặp ghép trái-phải)")
    parser.add_argument("--model_path", default="./output_pix2pix/checkpoints/generator_best.pth",
                        help="Đường dẫn đến generator checkpoint")
    parser.add_argument("--output_dir", default="./eval_results",
                        help="Thư mục lưu kết quả đánh giá")
    parser.add_argument("--img_size",   default=256, type=int)
    args = parser.parse_args()

    evaluate(args)

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
# python step3_evaluate.py \
#   --data_dir   ./dataset_restore/test \
#   --model_path ./output_pix2pix/checkpoints/generator_best.pth \
#   --output_dir ./eval_results
