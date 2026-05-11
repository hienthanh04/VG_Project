"""
TRAIN ARLES PERIOD — 4 models (theo pattern Paris v2)
=======================================================
Giống train_paris_v2.py nhưng cho thời kỳ Arles.

Thay đổi so với Arles cũ (step2_train_pix2pix.py v1):
  1. InstanceNorm thay BatchNorm  → màu sắc ổn định
  2. Feature Matching Loss        → texture rõ nét
  3. Linear LR Decay nửa sau      → hội tụ tốt hơn
  4. Image Buffer (pool 50)       → D ổn định hơn
  5. lambda_l1=100 (giống Paris)  → giữ màu sắc tốt

Config đã test tốt trên Paris:
  - grayscale:       lambda_l1=100, epochs=200
  - blur:            lambda_l1=100, epochs=200
  - blur_noise:      lambda_l1=100, epochs=200
  - gray_blur_noise: lambda_l1=100, epochs=200

Cách dùng:
    # Chạy từng model (khuyến nghị để theo dõi):
    python train_arles_v2.py --deg_type grayscale
    python train_arles_v2.py --deg_type blur
    python train_arles_v2.py --deg_type blur_noise
    python train_arles_v2.py --deg_type gray_blur_noise

    # Chạy full 4 models:
    python train_arles_v2.py

    # Tùy chỉnh (nếu muốn thử):
    python train_arles_v2.py --deg_type grayscale --epochs 200 --lambda_l1 100 --batch_size 4

    # Chỉ evaluate (đã train xong):
    python train_arles_v2.py --eval_only

    # So sánh Arles cũ vs mới:
    python train_arles_v2.py --compare_only
"""

import os
import cv2
import sys
import subprocess
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Paths ─────────────────────────────────────────────────────────────────────

# Dataset gốc của Arles — thư mục có sẵn train/ và test/
DEFAULT_ARLES_DIR   = "./vangogh_arles"

# Output của lần train mới này
DEFAULT_DATASET_DIR = "./dataset_arles_v2"
DEFAULT_OUTPUT_DIR  = "./output_arles_v2"
DEFAULT_EVAL_DIR    = "./eval_arles_v2"

# Output của lần train cũ (để so sánh)
OLD_OUTPUT_DIR      = "./output_pix2pix"       # grayscale_noise cũ
OLD_EVAL_DIR        = "./eval_results"          # eval của train cũ

DEG_TYPES = ["grayscale", "blur", "blur_noise", "gray_blur_noise"]

# ── Degradation functions (giống Paris v2) ────────────────────────────────────

def degrade_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def degrade_blur(img, ksize=17):
    if ksize % 2 == 0: ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def degrade_blur_noise(img, ksize=9, sigma=5):
    if ksize % 2 == 0: ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    noise = np.random.normal(0, sigma, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def degrade_gray_blur_noise(img, ksize=7, sigma=5):
    if ksize % 2 == 0: ksize += 1
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    noise = np.random.normal(0, sigma, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

DEG_FN = {
    "grayscale":       degrade_grayscale,
    "blur":            degrade_blur,
    "blur_noise":      degrade_blur_noise,
    "gray_blur_noise": degrade_gray_blur_noise,
}

# ── Check dataset ─────────────────────────────────────────────────────────────

def check_data(arles_dir):
    print(f"\n{'='*52}")
    print(f"  Kiểm tra dataset Arles")
    print(f"  {arles_dir}")
    print(f"{'='*52}")

    ok = True
    for split in ["train", "test"]:
        d = os.path.join(arles_dir, split)
        if not os.path.exists(d):
            print(f"  [LỖI] Không tìm thấy: {d}")
            ok = False
            continue
        imgs = [p for p in Path(d).glob("*") if p.suffix.lower() in IMG_EXTS]
        print(f"  {split}: {len(imgs)} ảnh")
        if len(imgs) == 0:
            print(f"  [LỖI] Không có ảnh!")
            ok = False

    # Kiểm tra thêm dataset cũ để so sánh sau này
    old_ckpt = os.path.join(OLD_OUTPUT_DIR, "checkpoints", "generator_best.pth")
    if os.path.exists(old_ckpt):
        print(f"  [INFO] Tìm thấy checkpoint cũ: {old_ckpt}")
        print(f"  → Sau khi train xong có thể so sánh với --compare_only")
    else:
        print(f"  [INFO] Không tìm thấy checkpoint cũ tại {old_ckpt}")

    if ok:
        print(f"  [OK] Dataset sẵn sàng")
    return ok

# ── Build dataset ─────────────────────────────────────────────────────────────

def make_pair(original_img, degraded_img, size=256):
    orig = cv2.resize(original_img, (size, size))
    degr = cv2.resize(degraded_img, (size, size))
    return np.concatenate([degr, orig], axis=1)

def build_dataset_for_deg(arles_dir, output_dir, deg_type, img_size=256):
    degrade_fn = DEG_FN[deg_type]
    for split in ["train", "test"]:
        src_dir = os.path.join(arles_dir, split)
        dst_dir = os.path.join(output_dir, split)
        os.makedirs(dst_dir, exist_ok=True)
        img_paths = sorted([p for p in Path(src_dir).glob("*")
                            if p.suffix.lower() in IMG_EXTS])
        ok = skip = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None: skip += 1; continue
            degraded = degrade_fn(img)
            pair     = make_pair(img, degraded, size=img_size)
            cv2.imwrite(
                os.path.join(dst_dir, img_path.stem + "_pair.jpg"),
                pair, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            ok += 1
        print(f"    {split}: {ok} ảnh" + (f" (bỏ qua {skip})" if skip else ""))
    print(f"  [OK] Dataset [{deg_type}] → {output_dir}")

def build_all_datasets(arles_dir, dataset_base, img_size=256):
    print(f"\n[Bước 1] Tạo 4 datasets Arles v2...")
    for deg_type in DEG_TYPES:
        out_dir = os.path.join(dataset_base, deg_type)
        print(f"\n  {deg_type}")
        build_dataset_for_deg(arles_dir, out_dir, deg_type, img_size)
    print("  Xong datasets!")

# ── Train ─────────────────────────────────────────────────────────────────────

def run_cmd(cmd, desc):
    print(f"\n  ▶ {desc}")
    print(f"    {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)

def train_single(deg_type, dataset_base, output_base,
                 epochs, batch_size, lambda_l1, lambda_feat, lr):
    data_dir   = os.path.join(dataset_base, deg_type)
    output_dir = os.path.join(output_base,  deg_type)
    if not os.path.exists(os.path.join(data_dir, "train")):
        print(f"  [SKIP] Dataset chưa có: {data_dir}")
        return
    run_cmd([
        "python", "step2_train_pix2pix.py",
        "--data_dir",    data_dir,
        "--output_dir",  output_dir,
        "--epochs",      str(epochs),
        "--batch_size",  str(batch_size),
        "--lambda_l1",   str(lambda_l1),
        "--lambda_feat", str(lambda_feat),
        "--lr",          str(lr),
        "--save_every",  "10",
    ], f"Train Arles v2 [{deg_type}] — {epochs} epochs | lambda_l1={lambda_l1}")

def train_all(dataset_base, output_base, epochs, batch_size,
              lambda_l1, lambda_feat, lr):
    print(f"\n[Bước 2] Train 4 models Arles v2 ({epochs} epochs)...")
    for deg_type in DEG_TYPES:
        print(f"\n  {'─'*44}")
        print(f"  Model: {deg_type.upper()}")
        print(f"  {'─'*44}")
        train_single(deg_type, dataset_base, output_base,
                     epochs, batch_size, lambda_l1, lambda_feat, lr)
    print("\n  Xong tất cả training!")

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_single(deg_type, dataset_base, output_base, eval_base, img_size=256):
    ckpt = os.path.join(output_base, deg_type, "checkpoints", "generator_best.pth")
    test = os.path.join(dataset_base, deg_type, "test")
    evd  = os.path.join(eval_base, deg_type)
    if not os.path.exists(ckpt):
        print(f"  [SKIP] Chưa có checkpoint: {ckpt}"); return
    run_cmd([
        "python", "step3_evaluate.py",
        "--data_dir",   test,
        "--model_path", ckpt,
        "--output_dir", evd,
        "--img_size",   str(img_size),
    ], f"Evaluate Arles v2 [{deg_type}]")

def evaluate_all(dataset_base, output_base, eval_base, img_size=256):
    print(f"\n[Bước 3] Evaluate 4 models Arles v2...")
    for deg_type in DEG_TYPES:
        evaluate_single(deg_type, dataset_base, output_base, eval_base, img_size)

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(eval_base, label="Arles v2"):
    try:
        import pandas as pd
    except ImportError:
        return

    print(f"\n{'='*68}")
    print(f"  KẾT QUẢ {label.upper()}")
    print(f"{'='*68}")
    print(f"  {'Deg type':<20} {'SSIM↑':>8} {'PSNR↑':>8} {'ΔSSIM↑':>8} {'Tốt':>5} {'TB':>5} {'Fail':>5}")
    print(f"  {'─'*64}")

    for deg_type in DEG_TYPES:
        csv_path = os.path.join(eval_base, deg_type, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"  {deg_type:<20} {'—':>8} {'—':>8} {'—':>8} {'—':>5} {'—':>5} {'—':>5}")
            continue
        df   = pd.read_csv(csv_path)
        ssim = df["ssim_restored"].mean()
        psnr = df["psnr_restored"].mean()
        delt = df["ssim_delta"].mean()
        good = int((df["ssim_restored"] > 0.8).sum())
        mid  = int(((df["ssim_restored"] >= 0.6) & (df["ssim_restored"] <= 0.8)).sum())
        fail = int((df["ssim_restored"] < 0.6).sum())
        print(f"  {deg_type:<20} {ssim:>8.4f} {psnr:>8.2f} {delt:>8.4f} {good:>5} {mid:>5} {fail:>5}")
    print(f"  {'─'*64}")

# ── Compare old vs new ────────────────────────────────────────────────────────

def compare_old_new(old_eval_base, new_eval_base):
    """
    So sánh kết quả Arles cũ (v1) vs mới (v2).
    Arles cũ chỉ có grayscale_noise ở eval_results/
    → map về grayscale để so sánh trực tiếp.
    """
    try:
        import pandas as pd
    except ImportError:
        return

    print(f"\n{'='*72}")
    print(f"  SO SÁNH ARLES CŨ (v1) vs MỚI (v2)")
    print(f"{'='*72}")

    # Arles cũ: grayscale_noise ở eval_results/
    old_paths = {
        "grayscale_noise (v1)": os.path.join(old_eval_base, "metrics.csv"),
    }
    # Arles mới: 4 deg types
    new_paths = {
        dt: os.path.join(new_eval_base, dt, "metrics.csv")
        for dt in DEG_TYPES
    }

    print(f"\n  {'Config':<28} {'SSIM restored':>14} {'PSNR restored':>14} {'ΔSSIM':>8}")
    print(f"  {'─'*68}")

    # In Arles cũ
    for label, path in old_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  {label:<28} {df['ssim_restored'].mean():>14.4f} "
                  f"{df['psnr_restored'].mean():>14.2f} "
                  f"{df['ssim_delta'].mean():>8.4f}  ← cũ")

    print(f"  {'─'*68}")

    # In Arles mới
    for dt, path in new_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  {dt:<28} {df['ssim_restored'].mean():>14.4f} "
                  f"{df['psnr_restored'].mean():>14.2f} "
                  f"{df['ssim_delta'].mean():>8.4f}  ← v2")

    print(f"  {'─'*68}")
    print(f"\n  → Checkpoint mới tại: {new_eval_base.replace('eval', 'output')}/[deg_type]/checkpoints/")
    print(f"  → Cập nhật app.py để dùng checkpoint mới cho Arles!")

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train lại Arles theo pattern Paris v2")

    parser.add_argument("--arles_dir",    default=DEFAULT_ARLES_DIR,
                        help="Thư mục gốc Arles (có train/ và test/ bên trong)")
    parser.add_argument("--dataset_base", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_base",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_base",    default=DEFAULT_EVAL_DIR)

    # Config train — giống Paris v2 đã test tốt
    parser.add_argument("--epochs",       default=200,  type=int)
    parser.add_argument("--batch_size",   default=4,    type=int)
    parser.add_argument("--lambda_l1",    default=100,  type=int,
                        help="100 giữ màu tốt (đã test trên Paris)")
    parser.add_argument("--lambda_feat",  default=10,   type=int,
                        help="Feature Matching Loss (0=tắt)")
    parser.add_argument("--lr",           default=2e-4, type=float)
    parser.add_argument("--img_size",     default=256,  type=int)

    # Chạy từng phần
    parser.add_argument("--deg_type",     default=None, choices=DEG_TYPES,
                        help="Chỉ chạy 1 deg type cụ thể")
    parser.add_argument("--dataset_only", action="store_true")
    parser.add_argument("--train_only",   action="store_true")
    parser.add_argument("--eval_only",    action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--compare_only", action="store_true",
                        help="So sánh kết quả cũ vs mới")

    args = parser.parse_args()

    # ── Chỉ so sánh ──
    if args.compare_only:
        compare_old_new(OLD_EVAL_DIR, args.eval_base)
        exit(0)

    # ── Chỉ summary ──
    if args.summary_only:
        print_summary(args.eval_base)
        exit(0)

    # ── Chỉ eval ──
    if args.eval_only:
        evaluate_all(args.dataset_base, args.output_base,
                     args.eval_base, args.img_size)
        print_summary(args.eval_base)
        compare_old_new(OLD_EVAL_DIR, args.eval_base)
        exit(0)

    # ── Chỉ 1 deg type ──
    if args.deg_type:
        dt = args.deg_type
        print(f"\n{'='*52}")
        print(f"  Arles v2 — {dt.upper()}")
        print(f"  lambda_l1={args.lambda_l1} | lambda_feat={args.lambda_feat}")
        print(f"  epochs={args.epochs} | batch={args.batch_size}")
        print(f"{'='*52}")

        if not args.train_only and not args.eval_only:
            if not check_data(args.arles_dir): exit(1)
            build_dataset_for_deg(
                args.arles_dir,
                os.path.join(args.dataset_base, dt),
                dt, args.img_size
            )

        if not args.dataset_only:
            train_single(dt, args.dataset_base, args.output_base,
                         args.epochs, args.batch_size,
                         args.lambda_l1, args.lambda_feat, args.lr)
            evaluate_single(dt, args.dataset_base, args.output_base,
                            args.eval_base, args.img_size)

        print_summary(args.eval_base)
        exit(0)

    # ── Full pipeline 4 models ──
    print(f"\n{'='*52}")
    print(f"  ARLES V2 — Full pipeline (4 models)")
    print(f"  lambda_l1={args.lambda_l1} | lambda_feat={args.lambda_feat}")
    print(f"  epochs={args.epochs} | batch={args.batch_size}")
    print(f"{'='*52}")

    if not check_data(args.arles_dir): exit(1)

    if not args.train_only:
        build_all_datasets(args.arles_dir, args.dataset_base, args.img_size)

    train_all(args.dataset_base, args.output_base,
              args.epochs, args.batch_size,
              args.lambda_l1, args.lambda_feat, args.lr)

    evaluate_all(args.dataset_base, args.output_base,
                 args.eval_base, args.img_size)

    print_summary(args.eval_base)
    compare_old_new(OLD_EVAL_DIR, args.eval_base)

    print(f"\n{'='*52}")
    print(f"  HOÀN THÀNH ARLES V2!")
    print(f"{'='*52}")
    print(f"  Checkpoints: {args.output_base}/[deg_type]/checkpoints/generator_best.pth")
    print(f"\n  Cập nhật MODELS_CONFIG trong app.py:")
    for dt in DEG_TYPES:
        path = f"{args.output_base}/{dt}/checkpoints/generator_best.pth"
        print(f"    arles/{dt}: {path}")

# ── Ví dụ chạy ────────────────────────────────────────────────────────────────
#
# Khuyến nghị: chạy từng model để theo dõi kết quả
#   python train_arles_v2.py --deg_type grayscale
#   python train_arles_v2.py --deg_type blur
#   python train_arles_v2.py --deg_type blur_noise
#   python train_arles_v2.py --deg_type gray_blur_noise
#
# Nếu muốn thay đổi config (test thêm):
#   python train_arles_v2.py --deg_type grayscale --lambda_l1 100 --epochs 200
#   python train_arles_v2.py --deg_type blur --lambda_l1 75 --epochs 200
#
# Sau khi train xong, so sánh với kết quả cũ:
#   python train_arles_v2.py --compare_only
#
# Cập nhật app.py — thay MODELS_CONFIG["arles"] thành:
#   "grayscale":      "./output_arles_v2/grayscale/checkpoints/generator_best.pth"
#   "blur":           "./output_arles_v2/blur/checkpoints/generator_best.pth"
#   "blur_noise":     "./output_arles_v2/blur_noise/checkpoints/generator_best.pth"
#   "gray_blur_noise":"./output_arles_v2/gray_blur_noise/checkpoints/generator_best.pth"
