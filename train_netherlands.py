"""
TRAIN NETHERLANDS PERIOD — 4 models
=====================================
Train pix2pix cho thời kỳ Netherlands với 4 loại suy giảm,
theo đúng pattern của Arles hiện tại.

Cấu trúc thư mục cần có trước:
    vangogh_netherlands/
        train/   (72 ảnh)
        test/    (13 ảnh)

Cấu trúc output sau khi chạy:
    dataset_netherlands/
        grayscale/train + test
        blur/train + test
        blur_noise/train + test
        gray_blur_noise/train + test

    output_netherlands/
        grayscale/checkpoints/generator_best.pth
        blur/checkpoints/generator_best.pth
        blur_noise/checkpoints/generator_best.pth
        gray_blur_noise/checkpoints/generator_best.pth

    eval_netherlands/
        grayscale/metrics.csv + visuals/
        blur/metrics.csv + visuals/
        blur_noise/metrics.csv + visuals/
        gray_blur_noise/metrics.csv + visuals/

Cách dùng:
    # Chạy full pipeline tất cả 4 models:
    python train_netherlands.py

    # Chỉ tạo dataset (không train):
    python train_netherlands.py --dataset_only

    # Chỉ train model cụ thể:
    python train_netherlands.py --deg_type grayscale
    python train_netherlands.py --deg_type blur
    python train_netherlands.py --deg_type blur_noise
    python train_netherlands.py --deg_type gray_blur_noise

    # Chỉ evaluate (đã train xong):
    python train_netherlands.py --eval_only

    # Đổi đường dẫn dataset nếu cần:
    python train_netherlands.py --nl_dir ./my_netherlands_folder --epochs 150
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

# ── Cấu hình mặc định ─────────────────────────────────────────────────────────

DEFAULT_NL_DIR      = "./vangogh_netherlands"   # thư mục gốc có train/ và test/
DEFAULT_DATASET_DIR = "./dataset_netherlands"
DEFAULT_OUTPUT_DIR  = "./output_netherlands"
DEFAULT_EVAL_DIR    = "./eval_netherlands"

DEG_TYPES = ["grayscale", "blur", "blur_noise", "gray_blur_noise"]

# ── 4 hàm degradation (giống Arles) ──────────────────────────────────────────

def degrade_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def degrade_blur(img, ksize=17):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def degrade_blur_noise(img, ksize=13, sigma=10):
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    noise   = np.random.normal(0, sigma, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_gray_blur_noise(img, ksize=13, sigma=20):
    if ksize % 2 == 0:
        ksize += 1
    gray    = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    noise   = np.random.normal(0, sigma, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)


DEG_FN = {
    "grayscale":       degrade_grayscale,
    "blur":            degrade_blur,
    "blur_noise":      degrade_blur_noise,
    "gray_blur_noise": degrade_gray_blur_noise,
}

# ── Kiểm tra dataset ──────────────────────────────────────────────────────────

def check_data(nl_dir):
    train_dir = os.path.join(nl_dir, "train")
    test_dir  = os.path.join(nl_dir, "test")

    print(f"\n{'='*52}")
    print(f"  Kiểm tra dataset Netherlands")
    print(f"  {nl_dir}")
    print(f"{'='*52}")

    ok = True
    for split, d in [("train", train_dir), ("test", test_dir)]:
        if not os.path.exists(d):
            print(f"  [LỖI] Không tìm thấy thư mục: {d}")
            ok = False
            continue
        imgs = [p for p in Path(d).glob("*") if p.suffix.lower() in IMG_EXTS]
        print(f"  {split}: {len(imgs)} ảnh")
        if len(imgs) == 0:
            print(f"  [LỖI] Thư mục {split} không có ảnh!")
            ok = False

    if ok:
        print(f"  [OK] Dataset sẵn sàng")
    return ok


# ── Tạo dataset pix2pix ───────────────────────────────────────────────────────

def make_pair(original_img, degraded_img, size=256):
    """Ghép [degraded | original] thành ảnh 512×256 theo chuẩn pix2pix."""
    orig = cv2.resize(original_img, (size, size))
    degr = cv2.resize(degraded_img, (size, size))
    return np.concatenate([degr, orig], axis=1)


def build_dataset_for_deg(nl_dir, output_dir, deg_type, img_size=256):
    """
    Tạo dataset cho 1 deg type từ folder có sẵn train/ và test/.
    Không shuffle lại vì train/test đã chia sẵn.
    """
    degrade_fn = DEG_FN[deg_type]

    for split in ["train", "test"]:
        src_dir = os.path.join(nl_dir, split)
        dst_dir = os.path.join(output_dir, split)
        os.makedirs(dst_dir, exist_ok=True)

        img_paths = sorted([p for p in Path(src_dir).glob("*")
                            if p.suffix.lower() in IMG_EXTS])
        ok = skip = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                skip += 1
                continue
            degraded = degrade_fn(img)
            pair     = make_pair(img, degraded, size=img_size)
            out_path = os.path.join(dst_dir, img_path.stem + "_pair.jpg")
            cv2.imwrite(out_path, pair, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok += 1

        print(f"    {split}: {ok} ảnh" + (f" (bỏ qua {skip})" if skip else ""))

    print(f"  [OK] Dataset [{deg_type}] → {output_dir}")


def build_all_datasets(nl_dir, dataset_base, img_size=256):
    print(f"\n[Bước 1] Tạo 4 datasets Netherlands...")
    for deg_type in DEG_TYPES:
        out_dir = os.path.join(dataset_base, deg_type)
        print(f"\n  Đang tạo: {deg_type}")
        build_dataset_for_deg(nl_dir, out_dir, deg_type, img_size)
    print("\n  Xong tất cả datasets!")


# ── Train ─────────────────────────────────────────────────────────────────────

def run_cmd(cmd, desc):
    print(f"\n  ▶ {desc}")
    print(f"    {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)


def train_single_model(deg_type, dataset_base, output_base, epochs, batch_size):
    data_dir   = os.path.join(dataset_base, deg_type)
    output_dir = os.path.join(output_base,  deg_type)

    if not os.path.exists(os.path.join(data_dir, "train")):
        print(f"  [SKIP] Dataset chưa có: {data_dir}")
        return

    run_cmd([
        "python", "step2_train_pix2pix.py",
        "--data_dir",   data_dir,
        "--output_dir", output_dir,
        "--epochs",     str(epochs),
        "--batch_size", str(batch_size),
        "--save_every", "10",
    ], f"Train Netherlands [{deg_type}] — {epochs} epochs")


def train_all_models(dataset_base, output_base, epochs, batch_size):
    print(f"\n[Bước 2] Train 4 models Netherlands ({epochs} epochs mỗi model)...")
    for deg_type in DEG_TYPES:
        print(f"\n  {'─'*44}")
        print(f"  Model: {deg_type.upper()}")
        print(f"  {'─'*44}")
        train_single_model(deg_type, dataset_base, output_base, epochs, batch_size)
    print("\n  Xong tất cả training!")


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_single(deg_type, dataset_base, output_base, eval_base, img_size=256):
    ckpt_path = os.path.join(output_base, deg_type, "checkpoints", "generator_best.pth")
    data_test = os.path.join(dataset_base, deg_type, "test")
    eval_dir  = os.path.join(eval_base,   deg_type)

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Chưa có checkpoint: {ckpt_path}")
        return

    run_cmd([
        "python", "step3_evaluate.py",
        "--data_dir",   data_test,
        "--model_path", ckpt_path,
        "--output_dir", eval_dir,
        "--img_size",   str(img_size),
    ], f"Evaluate Netherlands [{deg_type}]")


def evaluate_all_models(dataset_base, output_base, eval_base, img_size=256):
    print(f"\n[Bước 3] Evaluate 4 models Netherlands...")
    for deg_type in DEG_TYPES:
        evaluate_single(deg_type, dataset_base, output_base, eval_base, img_size)
    print("\n  Xong evaluate!")


# ── Tóm tắt kết quả ──────────────────────────────────────────────────────────

def print_summary(eval_base):
    """In bảng tóm tắt kết quả 4 models sau khi eval xong."""
    try:
        import pandas as pd
    except ImportError:
        print("  (Cần pandas để in summary: pip install pandas)")
        return

    print(f"\n{'='*68}")
    print(f"  KẾT QUẢ 4 MODELS NETHERLANDS")
    print(f"{'='*68}")
    print(f"  {'Deg type':<20} {'SSIM↑':>8} {'PSNR↑':>8} {'ΔSSIM↑':>8} {'Tốt':>5} {'TB':>5} {'Fail':>5}")
    print(f"  {'─'*64}")

    for deg_type in DEG_TYPES:
        csv_path = os.path.join(eval_base, deg_type, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"  {deg_type:<20} {'—':>8} {'—':>8} {'—':>8} {'—':>5} {'—':>5} {'—':>5}")
            continue
        df = pd.read_csv(csv_path)
        ssim  = df["ssim_restored"].mean()
        psnr  = df["psnr_restored"].mean()
        delta = df["ssim_delta"].mean()
        good  = int((df["ssim_restored"] > 0.8).sum())
        mid   = int(((df["ssim_restored"] >= 0.6) & (df["ssim_restored"] <= 0.8)).sum())
        fail  = int((df["ssim_restored"] < 0.6).sum())
        print(f"  {deg_type:<20} {ssim:>8.4f} {psnr:>8.2f} {delta:>8.4f} {good:>5} {mid:>5} {fail:>5}")

    print(f"  {'─'*64}")
    print(f"  So sánh với Arles (grayscale_noise): SSIM=0.7530, PSNR=19.90")
    print(f"  (Netherlands tone tối hơn nên SSIM có thể thấp hơn Arles — bình thường)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 4 pix2pix models cho Netherlands period"
    )
    parser.add_argument("--nl_dir",       default=DEFAULT_NL_DIR,
                        help="Thư mục gốc Netherlands (có train/ và test/ bên trong)")
    parser.add_argument("--dataset_base", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_base",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_base",    default=DEFAULT_EVAL_DIR)
    parser.add_argument("--epochs",       default=100, type=int)
    parser.add_argument("--batch_size",   default=4,   type=int)
    parser.add_argument("--img_size",     default=256, type=int)

    # Chạy từng phần
    parser.add_argument("--dataset_only", action="store_true",
                        help="Chỉ tạo dataset, không train")
    parser.add_argument("--train_only",   action="store_true",
                        help="Chỉ train, không tạo dataset")
    parser.add_argument("--eval_only",    action="store_true",
                        help="Chỉ evaluate, không train")
    parser.add_argument("--summary_only", action="store_true",
                        help="Chỉ in bảng tóm tắt kết quả")

    # Train từng model cụ thể
    parser.add_argument("--deg_type",     default=None,
                        choices=DEG_TYPES,
                        help="Chỉ chạy 1 deg type cụ thể")

    args = parser.parse_args()

    # ── Chạy 1 deg type cụ thể ──
    if args.deg_type:
        print(f"\n  Chỉ chạy: {args.deg_type}")
        if not args.train_only and not args.eval_only:
            if not check_data(args.nl_dir): exit(1)
            print(f"\n[Bước 1] Tạo dataset [{args.deg_type}]...")
            build_dataset_for_deg(
                args.nl_dir,
                os.path.join(args.dataset_base, args.deg_type),
                args.deg_type, args.img_size
            )
        if not args.dataset_only:
            train_single_model(
                args.deg_type, args.dataset_base, args.output_base,
                args.epochs, args.batch_size
            )
        if not args.dataset_only and not args.train_only:
            evaluate_single(
                args.deg_type, args.dataset_base, args.output_base,
                args.eval_base, args.img_size
            )
        print_summary(args.eval_base)
        exit(0)

    # ── Full pipeline hoặc từng bước ──
    if args.summary_only:
        print_summary(args.eval_base)

    elif args.dataset_only:
        if not check_data(args.nl_dir): exit(1)
        build_all_datasets(args.nl_dir, args.dataset_base, args.img_size)

    elif args.train_only:
        train_all_models(args.dataset_base, args.output_base,
                         args.epochs, args.batch_size)
        evaluate_all_models(args.dataset_base, args.output_base,
                            args.eval_base, args.img_size)
        print_summary(args.eval_base)

    elif args.eval_only:
        evaluate_all_models(args.dataset_base, args.output_base,
                            args.eval_base, args.img_size)
        print_summary(args.eval_base)

    else:
        # Full pipeline
        print(f"\n{'='*52}")
        print(f"  NETHERLANDS PERIOD — Full pipeline")
        print(f"  epochs={args.epochs} | batch={args.batch_size}")
        print(f"{'='*52}")

        if not check_data(args.nl_dir): exit(1)
        build_all_datasets(args.nl_dir,    args.dataset_base, args.img_size)
        train_all_models(args.dataset_base, args.output_base,
                         args.epochs,       args.batch_size)
        evaluate_all_models(args.dataset_base, args.output_base,
                            args.eval_base,    args.img_size)
        print_summary(args.eval_base)

        print(f"\n{'='*52}")
        print(f"  HOÀN THÀNH NETHERLANDS!")
        print(f"{'='*52}")
        print(f"  Datasets:    {args.dataset_base}/")
        print(f"  Checkpoints: {args.output_base}/[deg_type]/checkpoints/")
        print(f"  Eval:        {args.eval_base}/")
        print(f"\n  Checkpoint paths cho app.py:")
        for dt in DEG_TYPES:
            p = f"{args.output_base}/{dt}/checkpoints/generator_best.pth"
            print(f"    {dt:<20}: {p}")

# ── Ví dụ chạy ────────────────────────────────────────────────────────────────
#
# Full pipeline (tất cả 4 models, ~4-6 giờ):
#   python train_netherlands.py
#
# Chạy từng model một (nếu muốn kiểm soát từng bước):
#   python train_netherlands.py --deg_type grayscale
#   python train_netherlands.py --deg_type blur
#   python train_netherlands.py --deg_type blur_noise
#   python train_netherlands.py --deg_type gray_blur_noise
#
# Nếu dataset ở thư mục khác:
#   python train_netherlands.py --nl_dir ./vangogh_netherlands --epochs 150
#
# Sau khi train xong, chỉ chạy evaluate:
#   python train_netherlands.py --eval_only
#
# Xem kết quả tóm tắt:
#   python train_netherlands.py --summary_only
