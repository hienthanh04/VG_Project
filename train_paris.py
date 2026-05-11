"""
TRAIN PARIS V2 — Cải thiện output
===================================
Thay đổi so với v1:
  1. Data augmentation: flip + rotate + color jitter (×3 data, không cần ảnh mới)
  2. lambda_l1: 100 → 50 (cho GAN loss ảnh hưởng nhiều hơn → màu sắc richer)
  3. Epochs: 100 → 200 với LR decay sau epoch 100
  4. LR decay: giảm 50% mỗi 50 epoch sau epoch 100

Cách dùng:
    # Chạy full 4 models:
    python train_paris_v2.py

    # Chỉ 1 model cụ thể:
    python train_paris_v2.py --deg_type grayscale
    python train_paris_v2.py --deg_type blur
    python train_paris_v2.py --deg_type blur_noise
    python train_paris_v2.py --deg_type gray_blur_noise

    # Chỉ evaluate (đã train xong):
    python train_paris_v2.py --eval_only

    # Tùy chỉnh:
    python train_paris_v2.py --epochs 200 --lambda_l1 50 --batch_size 4
"""

import os
import cv2
import sys
import random
import subprocess
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import time

sys.path.insert(0, os.path.dirname(__file__))
from step2_train_pix2pix import Generator, Discriminator, init_weights

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_PARIS_DIR   = "./vangogh_paris"
DEFAULT_DATASET_DIR = "./dataset_paris_v2"
DEFAULT_OUTPUT_DIR  = "./output_paris_v2"
DEFAULT_EVAL_DIR    = "./eval_paris_v2"

DEG_TYPES = ["grayscale", "blur", "blur_noise", "gray_blur_noise"]

# ── Degradation functions ─────────────────────────────────────────────────────

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

# ── Dataset với augmentation ──────────────────────────────────────────────────

class ParisPix2PixDataset(Dataset):
    """
    Dataset pix2pix với augmentation cho Paris period.

    Augmentation áp dụng:
    - Horizontal flip (50%)
    - Random rotation ±15° (30%)
    - Color jitter: brightness/contrast/saturation ±20% (50%)
      → Giúp model học màu sắc Paris đa dạng hơn
    - Random crop + resize (20%)

    Lưu ý: Augmentation chỉ áp dụng cho train, KHÔNG cho test.
    Cùng transform áp dụng đồng thời cho cả degraded và original
    để giữ đúng cặp (degraded, original) tương ứng.
    """

    def __init__(self, split_dir, img_size=256, augment=False):
        self.paths   = sorted(
            [p for p in Path(split_dir).glob("*") if p.suffix.lower() in IMG_EXTS]
        )
        self.img_size = img_size
        self.augment  = augment

        # Color jitter chỉ cho ảnh original (target), không cho degraded
        
        self.to_tensor   = T.ToTensor()
        self.normalize   = T.Normalize([0.5]*3, [0.5]*3)

        print(f"    Dataset: {len(self.paths)} ảnh | augment={'ON' if augment else 'OFF'}")

    def __len__(self):
        return len(self.paths)

    def _augment_pair(self, degraded_pil, original_pil):
        """Áp dụng cùng spatial transform cho cả 2 ảnh."""
        # 1. Horizontal flip
        if random.random() < 0.5:
            degraded_pil = TF.hflip(degraded_pil)
            original_pil = TF.hflip(original_pil)

        # 2. Random rotation ±15°
        if random.random() < 0.3:
            angle = random.uniform(-7, 7)
            degraded_pil = TF.rotate(degraded_pil, angle, fill=0)
            original_pil = TF.rotate(original_pil, angle, fill=0)

        # 3. Random crop + resize (giữ đủ nội dung)
        if random.random() < 0.3:
            crop_size = random.randint(220, self.img_size)
            i = random.randint(0, self.img_size - crop_size)
            j = random.randint(0, self.img_size - crop_size)
            degraded_pil = TF.resized_crop(
                degraded_pil, i, j, crop_size, crop_size,
                (self.img_size, self.img_size)
            )
            original_pil = TF.resized_crop(
                original_pil, i, j, crop_size, crop_size,
                (self.img_size, self.img_size)
            )


        return degraded_pil, original_pil

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        w, h = img.size
        hw = w // 2

        degraded_pil = img.crop((0,  0, hw, h)).resize(
            (self.img_size, self.img_size), Image.LANCZOS)
        original_pil = img.crop((hw, 0, w,  h)).resize(
            (self.img_size, self.img_size), Image.LANCZOS)

        if self.augment:
            degraded_pil, original_pil = self._augment_pair(
                degraded_pil, original_pil)

        degraded = self.normalize(self.to_tensor(degraded_pil))
        original = self.normalize(self.to_tensor(original_pil))
        return degraded, original


# ── Build dataset từ paris_dir có sẵn ────────────────────────────────────────

def make_pair(original_img, degraded_img, size=256):
    orig = cv2.resize(original_img, (size, size))
    degr = cv2.resize(degraded_img, (size, size))
    return np.concatenate([degr, orig], axis=1)

def build_dataset_for_deg(paris_dir, output_dir, deg_type, img_size=256):
    degrade_fn = DEG_FN[deg_type]
    for split in ["train", "test"]:
        src_dir = os.path.join(paris_dir, split)
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
        print(f"    {split}: {ok} ảnh")
    print(f"  [OK] Dataset [{deg_type}] → {output_dir}")

def build_all_datasets(paris_dir, dataset_base, img_size=256):
    print(f"\n[Bước 1] Tạo 4 datasets Paris v2...")
    for deg_type in DEG_TYPES:
        out_dir = os.path.join(dataset_base, deg_type)
        print(f"\n  {deg_type}")
        build_dataset_for_deg(paris_dir, out_dir, deg_type, img_size)
    print("  Xong datasets!")


# ── Training loop với augmentation + LR decay ────────────────────────────────

def train_with_augmentation(data_dir, output_dir, epochs, batch_size,
                             lambda_l1, lr, img_size):
    """
    Train pix2pix với:
    - Augmentation bật trong DataLoader (không cần tạo ảnh augment trước)
    - lambda_l1 = 50 (thay vì 100) để màu sắc richer
    - LR decay: giảm 50% mỗi 50 epoch sau epoch 100
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device} | lambda_l1={lambda_l1} | epochs={epochs}")

    # Dataset với augmentation
    train_ds = ParisPix2PixDataset(
        os.path.join(data_dir, "train"), img_size, augment=True)
    test_ds  = ParisPix2PixDataset(
        os.path.join(data_dir, "test"),  img_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=2)

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(init_weights)
    D.apply(init_weights)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # LR schedulers: decay 50% mỗi 50 epochs sau epoch 100
    def lr_lambda(epoch):
        if epoch < 100:
            return 1.0
        return max(0.1, 0.5 ** ((epoch - 100) // 50))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    crit_GAN = nn.BCEWithLogitsLoss()
    crit_L1  = nn.L1Loss()

    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss_D,loss_G,loss_G_L1,lr\n")

    best_G_loss = float("inf")

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        ep_lD = ep_lG = ep_lG_L1 = 0.0
        t0 = time.time()

        for degraded, original in train_dl:
            degraded = degraded.to(device)
            original = original.to(device)
            fake     = G(degraded)

            # Train D
            opt_D.zero_grad()
            pred_real = D(degraded, original)
            pred_fake = D(degraded, fake.detach())
            loss_D = 0.5 * (
                crit_GAN(pred_real, torch.ones_like(pred_real)) +
                crit_GAN(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_D.backward()
            opt_D.step()

            # Train G
            opt_G.zero_grad()
            pred_fake_G = D(degraded, fake)
            loss_G_GAN  = crit_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_G_L1   = crit_L1(fake, original) * lambda_l1
            loss_G      = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

            ep_lD    += loss_D.item()
            ep_lG    += loss_G_GAN.item()
            ep_lG_L1 += loss_G_L1.item() / lambda_l1

        sched_G.step()
        sched_D.step()

        n        = len(train_dl)
        avg_lD   = ep_lD / n
        avg_lG   = ep_lG / n
        avg_l1   = ep_lG_L1 / n
        cur_lr   = opt_G.param_groups[0]['lr']
        elapsed  = time.time() - t0

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:4d}/{epochs} | D:{avg_lD:.4f} "
                  f"G:{avg_lG:.4f} L1:{avg_l1:.4f} "
                  f"lr:{cur_lr:.2e} | {elapsed:.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_lD:.6f},{avg_lG:.6f},{avg_l1:.6f},{cur_lr:.6f}\n")

        # Lưu samples mỗi 20 epoch
        if epoch % 20 == 0 or epoch == epochs:
            G.eval()
            with torch.no_grad():
                dg, og = next(iter(test_dl))
                dg = dg.to(device)
                fk = G(dg)
                grid = torch.cat([dg.cpu(), fk.cpu(), og], dim=3)
                save_image(
                    grid * 0.5 + 0.5,
                    os.path.join(sample_dir, f"epoch_{epoch:04d}.jpg"),
                    nrow=1
                )
            G.train()

            if avg_lG < best_G_loss:
                best_G_loss = avg_lG
                torch.save(G.state_dict(),
                           os.path.join(ckpt_dir, "generator_best.pth"))

        # Lưu checkpoint định kỳ mỗi 50 epoch
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch, "G": G.state_dict(), "D": D.state_dict()
            }, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pth"))

    print(f"    Train xong! Best checkpoint: {ckpt_dir}/generator_best.pth")


# ── Run command helper ────────────────────────────────────────────────────────

def run_cmd(cmd, desc):
    print(f"\n  ▶ {desc}")
    subprocess.run([str(c) for c in cmd], check=True)


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_single(deg_type, dataset_base, output_base, eval_base, img_size=256):
    ckpt_path = os.path.join(output_base, deg_type, "checkpoints", "generator_best.pth")
    data_test = os.path.join(dataset_base, deg_type, "test")
    eval_dir  = os.path.join(eval_base, deg_type)
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Chưa có checkpoint: {ckpt_path}")
        return
    run_cmd([
        "python", "step3_evaluate.py",
        "--data_dir",   data_test,
        "--model_path", ckpt_path,
        "--output_dir", eval_dir,
        "--img_size",   str(img_size),
    ], f"Evaluate Paris v2 [{deg_type}]")

def evaluate_all(dataset_base, output_base, eval_base, img_size=256):
    print("\n[Bước 3] Evaluate 4 models Paris v2...")
    for deg_type in DEG_TYPES:
        evaluate_single(deg_type, dataset_base, output_base, eval_base, img_size)


# ── Summary so sánh v1 vs v2 ─────────────────────────────────────────────────

def compare_v1_v2(eval_v1, eval_v2):
    try:
        import pandas as pd
    except ImportError:
        return

    print(f"\n{'='*72}")
    print("  SO SÁNH PARIS V1 vs V2")
    print(f"{'='*72}")
    print(f"  {'Deg type':<20} {'SSIM v1':>9} {'SSIM v2':>9} {'Δ':>7} "
          f"{'PSNR v1':>9} {'PSNR v2':>9} {'Δ':>7}")
    print(f"  {'─'*70}")

    for deg_type in DEG_TYPES:
        p1 = os.path.join(eval_v1, deg_type, "metrics.csv")
        p2 = os.path.join(eval_v2, deg_type, "metrics.csv")

        def read(p):
            if not os.path.exists(p): return None, None
            df = pd.read_csv(p)
            return df["ssim_restored"].mean(), df["psnr_restored"].mean()

        s1, psnr1 = read(p1)
        s2, psnr2 = read(p2)

        if s1 is None and s2 is None:
            print(f"  {deg_type:<20} {'—':>9} {'—':>9} {'—':>7} {'—':>9} {'—':>9} {'—':>7}")
        elif s1 is None:
            print(f"  {deg_type:<20} {'—':>9} {s2:>9.4f} {'—':>7} {'—':>9} {psnr2:>9.2f} {'—':>7}")
        elif s2 is None:
            print(f"  {deg_type:<20} {s1:>9.4f} {'—':>9} {'—':>7} {psnr1:>9.2f} {'—':>9} {'—':>7}")
        else:
            mark = "↑" if s2 > s1 else "↓"
            print(f"  {deg_type:<20} {s1:>9.4f} {s2:>9.4f} "
                  f"{mark}{abs(s2-s1):>6.4f} "
                  f"{psnr1:>9.2f} {psnr2:>9.2f} "
                  f"{mark}{abs(psnr2-psnr1):>6.2f}")

    print(f"  {'─'*70}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Paris v2 — augmentation + lambda 50 + 200 epochs")
    parser.add_argument("--paris_dir",    default=DEFAULT_PARIS_DIR)
    parser.add_argument("--dataset_base", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_base",  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_base",    default=DEFAULT_EVAL_DIR)
    parser.add_argument("--eval_v1",      default="./eval_paris",
                        help="Thư mục eval v1 để so sánh (tùy chọn)")
    parser.add_argument("--epochs",       default=200, type=int)
    parser.add_argument("--batch_size",   default=4,   type=int)
    parser.add_argument("--lambda_l1",    default=50,  type=int,
                        help="L1 loss weight (v1=100, v2=50 để màu richer)")
    parser.add_argument("--lr",           default=2e-4, type=float)
    parser.add_argument("--img_size",     default=256, type=int)
    parser.add_argument("--deg_type",     default=None, choices=DEG_TYPES)
    parser.add_argument("--dataset_only", action="store_true")
    parser.add_argument("--train_only",   action="store_true")
    parser.add_argument("--eval_only",    action="store_true")
    args = parser.parse_args()

    # ── Chỉ 1 deg type ──
    if args.deg_type:
        dt = args.deg_type
        print(f"\n{'='*52}")
        print(f"  Paris v2 — {dt.upper()}")
        print(f"  lambda_l1={args.lambda_l1} | epochs={args.epochs} | augment=ON")
        print(f"{'='*52}")

        if not args.train_only and not args.eval_only:
            build_dataset_for_deg(
                args.paris_dir,
                os.path.join(args.dataset_base, dt),
                dt, args.img_size
            )
        if not args.dataset_only and not args.eval_only:
            print(f"\n[Bước 2] Train [{dt}]...")
            train_with_augmentation(
                os.path.join(args.dataset_base, dt),
                os.path.join(args.output_base,  dt),
                args.epochs, args.batch_size,
                args.lambda_l1, args.lr, args.img_size
            )
        if not args.dataset_only and not args.train_only:
            evaluate_single(dt, args.dataset_base, args.output_base,
                            args.eval_base, args.img_size)
        compare_v1_v2(args.eval_v1, args.eval_base)
        exit(0)

    # ── Full pipeline ──
    if args.eval_only:
        evaluate_all(args.dataset_base, args.output_base,
                     args.eval_base, args.img_size)
        compare_v1_v2(args.eval_v1, args.eval_base)
        exit(0)

    print(f"\n{'='*52}")
    print(f"  PARIS V2 — Full pipeline (4 models)")
    print(f"  lambda_l1={args.lambda_l1} | epochs={args.epochs} | augment=ON")
    print(f"{'='*52}")

    if not args.train_only:
        build_all_datasets(args.paris_dir, args.dataset_base, args.img_size)

    print(f"\n[Bước 2] Train 4 models Paris v2 ({args.epochs} epochs)...")
    for deg_type in DEG_TYPES:
        print(f"\n  {'─'*44}")
        print(f"  Model: {deg_type.upper()}")
        print(f"  {'─'*44}")
        train_with_augmentation(
            os.path.join(args.dataset_base, deg_type),
            os.path.join(args.output_base,  deg_type),
            args.epochs, args.batch_size,
            args.lambda_l1, args.lr, args.img_size
        )

    evaluate_all(args.dataset_base, args.output_base,
                 args.eval_base, args.img_size)
    compare_v1_v2(args.eval_v1, args.eval_base)

    print(f"\n{'='*52}")
    print("  HOÀN THÀNH PARIS V2!")
    print(f"{'='*52}")
    print(f"  Checkpoints: {args.output_base}/[deg_type]/checkpoints/")
    print(f"  Eval:        {args.eval_base}/")
    print(f"\n  Checkpoint paths cho app.py:")
    for dt in DEG_TYPES:
        print(f"    {dt:<20}: {args.output_base}/{dt}/checkpoints/generator_best.pth")

# ── Ví dụ chạy ──────────────────────────────────────────────────────────────────
#
# Full 4 models:
#   python train_paris_v2.py
#
# Chạy từng model (gợi ý — dễ theo dõi hơn):
#   python train_paris_v2.py --deg_type grayscale
#   python train_paris_v2.py --deg_type blur
#   python train_paris_v2.py --deg_type blur_noise
#   python train_paris_v2.py --deg_type gray_blur_noise
#
# Sau khi train xong, so sánh v1 vs v2:
#   python train_paris_v2.py --eval_only --eval_v1 ./eval_paris
#
# Nếu muốn thử lambda khác:
#   python train_paris_v2.py --deg_type blur --lambda_l1 75 --epochs 150
