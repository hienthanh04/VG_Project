"""
BƯỚC 1: XÂY DỰNG DATASET PHỤC HỒI TRANH VAN GOGH
==================================================
Input : thư mục chứa tranh Van Gogh màu gốc (jpg/png)
Output: dataset theo chuẩn pix2pix (ảnh ghép trái-phải)

Cấu trúc output:
    dataset_restore/
        train/   (ảnh ghép: [degraded | original], 512x256)
        test/    (ảnh ghép: [degraded | original], 512x256)

Cách dùng:
    python step1_build_dataset.py --input_dir ./vangogh_color \
                                  --output_dir ./dataset_restore \
                                  --deg_type grayscale_noise \
                                  --train_ratio 0.8
"""

import os
import cv2
import numpy as np
import argparse
import random
import shutil
from pathlib import Path

# ── Các mức suy giảm ────────────────────────────────────────────────────────

def degrade_grayscale(img):
    """Mức nhẹ: chỉ grayscale → RGB lại để cùng kênh với target"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def degrade_grayscale_blur(img, ksize=5):
    """Mức vừa: grayscale + Gaussian blur"""
    gray = degrade_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return blurred

def degrade_grayscale_noise(img, sigma=25):
    """Mức mạnh: grayscale + Gaussian noise"""
    gray = degrade_grayscale(img)
    noise = np.random.normal(0, sigma, gray.shape).astype(np.float32)
    noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

DEGRADATION_MAP = {
    "grayscale":       degrade_grayscale,
    "grayscale_blur":  degrade_grayscale_blur,
    "grayscale_noise": degrade_grayscale_noise,
}

# ── Tạo cặp ảnh ghép (pix2pix format) ──────────────────────────────────────

def make_pair(original_img, degraded_img, size=256):
    """
    Ghép [degraded | original] thành 1 ảnh 512×256
    Đây là format chuẩn pix2pix: input bên trái, target bên phải.
    """
    orig = cv2.resize(original_img, (size, size))
    degr = cv2.resize(degraded_img, (size, size))
    return np.concatenate([degr, orig], axis=1)  # shape: (256, 512, 3)

# ── Pipeline chính ──────────────────────────────────────────────────────────

def build_dataset(input_dir, output_dir, deg_type, train_ratio, img_size, seed):
    random.seed(seed)
    np.random.seed(seed)

    degrade_fn = DEGRADATION_MAP[deg_type]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Thu thập ảnh
    all_images = sorted([
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in exts
    ])
    if not all_images:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {input_dir}")

    print(f"Tìm thấy {len(all_images)} ảnh gốc.")

    # Chia train/test
    random.shuffle(all_images)
    n_train = int(len(all_images) * train_ratio)
    splits = {
        "train": all_images[:n_train],
        "test":  all_images[n_train:],
    }
    print(f"  Train: {len(splits['train'])} | Test: {len(splits['test'])}")

    # Tạo thư mục output
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Xử lý từng ảnh
    stats = {"ok": 0, "skip": 0}
    for split, paths in splits.items():
        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [SKIP] Không đọc được: {img_path.name}")
                stats["skip"] += 1
                continue

            degraded = degrade_fn(img)
            pair = make_pair(img, degraded, size=img_size)

            out_name = img_path.stem + "_pair.jpg"
            out_path = os.path.join(output_dir, split, out_name)
            cv2.imwrite(out_path, pair, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["ok"] += 1

    print(f"\nHoàn thành! Đã tạo {stats['ok']} cặp ảnh, bỏ qua {stats['skip']} ảnh lỗi.")
    print(f"Dataset lưu tại: {output_dir}/")

    # Tạo thư mục sample để xem thử
    _save_samples(output_dir, n=6)


def _save_samples(output_dir, n=6):
    """Lưu n ảnh mẫu vào thư mục samples/ để kiểm tra nhanh."""
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    files = os.listdir(train_dir)[:n]
    for f in files:
        shutil.copy(os.path.join(train_dir, f), os.path.join(sample_dir, f))
    print(f"Xem {len(files)} ảnh mẫu tại: {sample_dir}/")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo dataset phục hồi tranh Van Gogh")
    parser.add_argument("--input_dir",   default="./vangogh_color",
                        help="Thư mục chứa tranh Van Gogh màu gốc")
    parser.add_argument("--output_dir",  default="./dataset_restore",
                        help="Thư mục lưu dataset đã xử lý")
    parser.add_argument("--deg_type",    default="grayscale_noise",
                        choices=list(DEGRADATION_MAP.keys()),
                        help="Kiểu suy giảm: grayscale | grayscale_blur | grayscale_noise")
    parser.add_argument("--train_ratio", default=0.8, type=float,
                        help="Tỉ lệ train (mặc định 0.8 = 80%%)")
    parser.add_argument("--img_size",    default=256, type=int,
                        help="Kích thước ảnh resize (mặc định 256)")
    parser.add_argument("--seed",        default=42, type=int,
                        help="Random seed để tái lập kết quả")
    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        deg_type=args.deg_type,
        train_ratio=args.train_ratio,
        img_size=args.img_size,
        seed=args.seed,
    )

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
# python step1_build_dataset.py --input_dir ./vangogh_color --deg_type grayscale_noise
# python step1_build_dataset.py --input_dir ./vangogh_color --deg_type grayscale_blur
# python step1_build_dataset.py --input_dir ./vangogh_color --deg_type grayscale
