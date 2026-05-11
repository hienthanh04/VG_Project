"""
PATCH CHO step1_build_dataset.py
=================================
Thêm 3 loại suy giảm mới cho bài toán phục hồi tranh bị mờ/mất nét:

  blur_only      — Gaussian blur nặng (ảnh mờ hoàn toàn, còn màu)
  motion_blur    — Motion blur ngang (giả lập ảnh bị rung khi chụp)
  jpeg_artifact  — Nén JPEG chất lượng thấp (vỡ block, mất chi tiết)

Cách dùng — copy 3 hàm này vào DEGRADATION_MAP của step1_build_dataset.py,
hoặc chạy file này độc lập như script (xem phần __main__ dưới).

Lệnh chạy:
    python step1_patch_blur.py --input_dir ./vangogh_color \
                               --output_dir ./dataset_blur \
                               --deg_type blur_only

    python step1_patch_blur.py --input_dir ./vangogh_color \
                               --output_dir ./dataset_motion \
                               --deg_type motion_blur

    python step1_patch_blur.py --input_dir ./vangogh_color \
                               --output_dir ./dataset_jpeg \
                               --deg_type jpeg_artifact

    # Kết hợp blur + noise (thực tế nhất):
    python step1_patch_blur.py --input_dir ./vangogh_color \
                               --output_dir ./dataset_blur_noise \
                               --deg_type blur_noise
"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ═══════════════════════════════════════════════════════════════════
#  3 HÀM DEGRADATION MỚI — chỉ cần copy vào DEGRADATION_MAP của step1
# ═══════════════════════════════════════════════════════════════════

def degrade_blur_only(img, ksize=17):
    """
    Gaussian blur nặng — ảnh vẫn giữ màu nhưng mờ hoàn toàn.
    ksize=17 cho độ mờ rõ rệt, có thể tăng lên 21-25 nếu muốn mờ hơn.

    Đây là loại suy giảm phổ biến nhất trong thực tế:
    - Tranh cũ bị phai nét theo thời gian
    - Ảnh scan độ phân giải thấp
    - Ảnh chụp không lấy nét
    """
    # Đảm bảo ksize lẻ
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def degrade_motion_blur(img, kernel_size=21, angle=0):
    """
    Motion blur theo hướng ngang (mặc định) hoặc chéo.
    Giả lập ảnh bị rung/nhòe theo hướng chuyển động.

    angle=0   → nhòe ngang (horizontal)
    angle=45  → nhòe chéo 45°
    angle=90  → nhòe dọc (vertical)

    Thực tế: ảnh tranh bị chụp khi camera rung, hoặc scan bị lệch.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0 / kernel_size  # kernel ngang

    if angle != 0:
        # Xoay kernel theo góc
        M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel /= (kernel.sum() + 1e-8)

    return cv2.filter2D(img, -1, kernel)


def degrade_jpeg_artifact(img, quality=10):
    """
    Nén JPEG với chất lượng rất thấp (quality=10 out of 100).
    Tạo ra các block artifact đặc trưng của JPEG nén mạnh.

    quality=10  → artifact nặng, block 8x8 rõ ràng
    quality=20  → artifact vừa
    quality=5   → artifact cực nặng (dùng để test)

    Thực tế: ảnh tranh được số hóa và lưu với chất lượng thấp.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def degrade_blur_noise(img, blur_ksize=13, noise_sigma=20):
    """
    Kết hợp Gaussian blur + Gaussian noise.
    Thực tế nhất: ảnh vừa mờ vừa nhiễu (scan chất lượng kém + bảo quản kém).
    """
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    noise   = np.random.normal(0, noise_sigma, blurred.shape).astype(np.float32)
    return np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_blur_jpeg(img, blur_ksize=11, quality=15):
    """
    Kết hợp blur + JPEG artifact.
    Giả lập tranh cũ bị scan rồi lưu nén.
    """
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", blurred, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def degrade_gray_blur_noise(img, blur_ksize=13, noise_sigma=20):
    """
    ★ Combo khó nhất: Grayscale + Gaussian blur + Gaussian noise
    
    Thứ tự áp dụng:
      1. Grayscale  → mất toàn bộ thông tin màu
      2. Blur       → mờ, mất chi tiết cạnh
      3. Noise      → thêm nhiễu hạt lên ảnh đã mờ
    
    Đây là mức suy giảm thực tế nhất cho tranh cũ bị:
      - Phai màu hoàn toàn (bảo quản kém)
      - Mờ nét (tuổi thọ vật liệu, scan kém)  
      - Nhiễu (hạt bụi, ẩm mốc, scan analog)
    
    Model phải học đồng thời: tô màu + khử mờ + khử nhiễu
    → Bài toán khó nhất, chứng minh khả năng tổng quát hóa cao nhất.
    """
    # Bước 1: grayscale → RGB 3 kênh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Bước 2: Gaussian blur
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(gray_3ch, (blur_ksize, blur_ksize), 0)
    
    # Bước 3: Gaussian noise
    noise = np.random.normal(0, noise_sigma, blurred.shape).astype(np.float32)
    result = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return result


# Map đầy đủ bao gồm cả 5 loại cũ + mới
DEGRADATION_MAP_FULL = {
    # ── Cũ (đã có trong step1) ──
    "grayscale":        lambda img: cv2.cvtColor(
                            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR),
    "grayscale_blur":   lambda img: cv2.GaussianBlur(
                            cv2.cvtColor(
                                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                cv2.COLOR_GRAY2BGR), (5,5), 0),
    "grayscale_noise":  lambda img: _grayscale_noise(img),

    # ── Mới ──
    "blur_only":        degrade_blur_only,
    "motion_blur":      degrade_motion_blur,
    "jpeg_artifact":    degrade_jpeg_artifact,
    "blur_noise":       degrade_blur_noise,
    "blur_jpeg":        degrade_blur_jpeg,

    # ── Combo khó nhất ──
    "gray_blur_noise":  degrade_gray_blur_noise,   # ★ grayscale + blur + noise
}

def _grayscale_noise(img, sigma=25):
    gray  = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    noise = np.random.normal(0, sigma, gray.shape).astype(np.float32)
    return np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
#  PIPELINE (copy từ step1, thêm deg_type mới)
# ═══════════════════════════════════════════════════════════════════

def make_pair(original_img, degraded_img, size=256):
    orig = cv2.resize(original_img, (size, size))
    degr = cv2.resize(degraded_img, (size, size))
    return np.concatenate([degr, orig], axis=1)


def build_dataset(input_dir, output_dir, deg_type, train_ratio, img_size, seed):
    random.seed(seed)
    np.random.seed(seed)

    degrade_fn = DEGRADATION_MAP_FULL[deg_type]

    all_images = sorted([
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ])
    if not all_images:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {input_dir}")

    print(f"Tìm thấy {len(all_images)} ảnh. Deg type: [{deg_type}]")

    random.shuffle(all_images)
    n_train = int(len(all_images) * train_ratio)
    splits  = {"train": all_images[:n_train], "test": all_images[n_train:]}
    print(f"  Train: {len(splits['train'])} | Test: {len(splits['test'])}")

    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    ok = skip = 0
    for split, paths in splits.items():
        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is None:
                skip += 1; continue
            degraded = degrade_fn(img)
            pair     = make_pair(img, degraded, size=img_size)
            out_path = os.path.join(output_dir, split, img_path.stem + "_pair.jpg")
            cv2.imwrite(out_path, pair, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok += 1

    print(f"Xong! {ok} cặp ảnh → {output_dir}/")


# ─── Hàm tiện ích: xem trước 1 ảnh với tất cả các loại suy giảm ──────────────

def preview_all_degradations(img_path, output_path="degradation_preview.jpg"):
    """
    Tạo ảnh preview so sánh tất cả loại suy giảm trên 1 ảnh mẫu.
    Dùng để chọn loại suy giảm phù hợp trước khi tạo toàn bộ dataset.

    Cách dùng:
        from step1_patch_blur import preview_all_degradations
        preview_all_degradations("./vangogh_color/starry_night.jpg")
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    deg_types = [
        ("Original",        img_bgr),
        ("grayscale",       DEGRADATION_MAP_FULL["grayscale"](img_bgr)),
        ("grayscale_noise", DEGRADATION_MAP_FULL["grayscale_noise"](img_bgr)),
        ("blur_only",       degrade_blur_only(img_bgr)),
        ("motion_blur",     degrade_motion_blur(img_bgr)),
        ("jpeg_artifact",   degrade_jpeg_artifact(img_bgr)),
        ("blur_noise",      degrade_blur_noise(img_bgr)),
        ("blur_jpeg",       degrade_blur_jpeg(img_bgr)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("So sánh các loại suy giảm — chọn loại phù hợp", fontsize=13)

    for ax, (title, img) in zip(axes.flat, deg_types):
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_show)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Preview lưu tại: {output_path}")
    print("→ Mở file này để chọn loại suy giảm muốn dùng trước khi chạy dataset.")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo dataset với degradation mới (blur/motion/jpeg)")
    parser.add_argument("--input_dir",   default="./vangogh_color")
    parser.add_argument("--output_dir",  default="./dataset_blur")
    parser.add_argument("--deg_type",    default="blur_only",
                        choices=list(DEGRADATION_MAP_FULL.keys()))
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--img_size",    default=256, type=int)
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--preview",     default=None,
                        help="Đường dẫn 1 ảnh mẫu để xem trước tất cả degradation")
    args = parser.parse_args()

    if args.preview:
        preview_all_degradations(args.preview)
    else:
        build_dataset(
            input_dir   = args.input_dir,
            output_dir  = args.output_dir,
            deg_type    = args.deg_type,
            train_ratio = args.train_ratio,
            img_size    = args.img_size,
            seed        = args.seed,
        )

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────
#
# Xem trước tất cả loại suy giảm trên 1 ảnh (làm trước!):
#   python step1_patch_blur.py --preview ./vangogh_color/sunflowers.jpg
#
# Tạo dataset mờ (blur_only):
#   python step1_patch_blur.py --input_dir ./vangogh_color --deg_type blur_only \
#       --output_dir ./dataset_blur
#
# Tạo dataset nhòe (motion_blur):
#   python step1_patch_blur.py --input_dir ./vangogh_color --deg_type motion_blur \
#       --output_dir ./dataset_motion
#
# Tạo dataset mờ + nhiễu (thực tế nhất):
#   python step1_patch_blur.py --input_dir ./vangogh_color --deg_type blur_noise \
#       --output_dir ./dataset_blur_noise
#
# Sau đó train như bình thường:
#   python step2_train_pix2pix.py --data_dir ./dataset_blur --output_dir ./output_blur
#   python step3_evaluate.py --data_dir ./dataset_blur/test \
#       --model_path ./output_blur/checkpoints/generator_best.pth \
#       --output_dir ./eval_blur
