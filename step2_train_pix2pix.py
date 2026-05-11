"""
step2_train_pix2pix.py  — Phiên bản cải tiến theo chuẩn GitHub (Isola et al. 2017)
=====================================================================================
Cải tiến so với v1:
  1. InstanceNorm thay BatchNorm  → màu sắc ổn định, đẹp hơn với batch nhỏ
  2. Feature Matching Loss        → texture rõ nét, không bị muddy
  3. Linear LR Decay nửa sau     → hội tụ fine-grained, không overfit
  4. Image Buffer (pool 50 ảnh)  → Discriminator ổn định hơn
  5. Weight init chuẩn N(0,0.02) → khởi đầu training tốt hơn

Tương thích với: train_paris.py, train_netherlands.py, train_paris_v2.py
(Các file đó gọi: from step2_train_pix2pix import Generator, Discriminator, init_weights)
"""

import os, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Generator: U-Net + InstanceNorm ──────────────────────────────────────────

class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)


class Generator(nn.Module):
    """U-Net Generator chuẩn pix2pix (Isola 2017) với InstanceNorm."""
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        self.down1 = UNetDown(in_ch,  ngf,    normalize=False)
        self.down2 = UNetDown(ngf,    ngf*2)
        self.down3 = UNetDown(ngf*2,  ngf*4)
        self.down4 = UNetDown(ngf*4,  ngf*8)
        self.down5 = UNetDown(ngf*8,  ngf*8)
        self.down6 = UNetDown(ngf*8,  ngf*8)
        self.down7 = UNetDown(ngf*8,  ngf*8)
        self.down8 = UNetDown(ngf*8,  ngf*8, normalize=False)

        self.up1 = UNetUp(ngf*8,   ngf*8,  dropout=0.5)
        self.up2 = UNetUp(ngf*16,  ngf*8,  dropout=0.5)
        self.up3 = UNetUp(ngf*16,  ngf*8,  dropout=0.5)
        self.up4 = UNetUp(ngf*16,  ngf*8)
        self.up5 = UNetUp(ngf*16,  ngf*4)
        self.up6 = UNetUp(ngf*8,   ngf*2)
        self.up7 = UNetUp(ngf*4,   ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x);  d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4); d6 = self.down6(d5)
        d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8, d7); u2 = self.up2(u1, d6); u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4); u5 = self.up5(u4, d3); u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


# ── Discriminator: PatchGAN 70×70 + InstanceNorm ─────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator chuẩn (Isola 2017) với InstanceNorm.
    Trả về list features từ mỗi tầng để dùng Feature Matching Loss.
    """
    def __init__(self, in_ch=6, ndf=64, n_layers=3):
        super().__init__()
        sequence = [nn.Conv2d(in_ch, ndf, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True)]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev, nf = nf, min(nf * 2, 512)
            sequence += [nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                         nn.InstanceNorm2d(nf),
                         nn.LeakyReLU(0.2, inplace=True)]
        nf_prev, nf = nf, min(nf * 2, 512)
        sequence += [nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False),
                     nn.InstanceNorm2d(nf),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(nf, 1, 4, 1, 1)]

        # Tách thành từng block để lấy intermediate features
        self.layers = nn.ModuleList()
        buf = []
        for layer in sequence:
            buf.append(layer)
            if isinstance(layer, nn.LeakyReLU):
                self.layers.append(nn.Sequential(*buf)); buf = []
        if buf:
            self.layers.append(nn.Sequential(*buf))

    def forward(self, x, y, return_features=False):
        inp = torch.cat([x, y], dim=1)
        feats = []
        for layer in self.layers:
            inp = layer(inp)
            feats.append(inp)
        return feats if return_features else feats[-1]


# ── Image Buffer ──────────────────────────────────────────────────────────────

class ImageBuffer:
    """Replay buffer giữ pool ảnh fake để D training ổn định hơn."""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.buffer    = []

    def push_and_pop(self, images):
        if self.pool_size == 0:
            return images
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.pool_size:
                self.buffer.append(img); result.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.buffer[idx].clone()
                    self.buffer[idx] = img; result.append(old)
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


# ── Weight init ───────────────────────────────────────────────────────────────

def init_weights(m):
    """N(0,0.02) cho Conv, N(1,0.02) cho InstanceNorm — chuẩn GitHub."""
    cn = m.__class__.__name__
    if cn in ("Conv2d", "ConvTranspose2d"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
    elif cn == "InstanceNorm2d":
        if m.weight is not None: nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias   is not None: nn.init.constant_(m.bias.data,   0.0)


# ── Feature Matching Loss ─────────────────────────────────────────────────────

def feature_matching_loss(D, degraded, fake, real, lambda_feat=10):
    """
    Buộc G tạo ảnh có intermediate features gần với ảnh real.
    → Output texture rõ hơn, không bị muddy.
    """
    crit     = nn.L1Loss()
    feats_f  = D(degraded, fake, return_features=True)
    feats_r  = D(degraded, real, return_features=True)
    loss = sum(crit(ff, fr.detach())
               for ff, fr in zip(feats_f[:-1], feats_r[:-1]))
    return loss * lambda_feat


# ── LR Scheduler ─────────────────────────────────────────────────────────────

def get_lr_lambda(epochs, decay_start_ratio=0.5):
    """LR cố định nửa đầu, decay tuyến tính về 0 nửa sau."""
    decay_start = int(epochs * decay_start_ratio)
    def fn(epoch):
        if epoch < decay_start: return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (epochs - decay_start))
    return fn


# ── Dataset ───────────────────────────────────────────────────────────────────

class Pix2PixDataset(Dataset):
    def __init__(self, split_dir, img_size=256):
        self.paths = sorted([p for p in Path(split_dir).glob("*")
                             if p.suffix.lower() in IMG_EXTS])
        self.tf = T.Compose([T.Resize((img_size, img_size)),
                             T.ToTensor(),
                             T.Normalize([0.5]*3, [0.5]*3)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        w, h = img.size; hw = w // 2
        return self.tf(img.crop((0, 0, hw, h))), self.tf(img.crop((hw, 0, w, h)))


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs} | lambda_l1={args.lambda_l1} "
          f"| lambda_feat={args.lambda_feat} | lr={args.lr}")

    train_ds = Pix2PixDataset(os.path.join(args.data_dir, "train"), args.img_size)
    test_ds  = Pix2PixDataset(os.path.join(args.data_dir, "test"),  args.img_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size,
                          shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} ảnh | Test: {len(test_ds)} ảnh")

    G = Generator().to(device);  G.apply(init_weights)
    D = Discriminator().to(device); D.apply(init_weights)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_fn   = get_lr_lambda(args.epochs)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_fn)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_fn)

    crit_GAN = nn.BCEWithLogitsLoss()
    crit_L1  = nn.L1Loss()
    fake_buf = ImageBuffer(pool_size=50)

    ckpt_dir   = os.path.join(args.output_dir, "checkpoints")
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss_D,loss_G_gan,loss_G_l1,loss_G_feat,lr\n")

    best_G_loss = float("inf")
    print(f"\nBắt đầu train {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        ep_lD = ep_lG = ep_l1 = ep_feat = 0.0
        t0 = time.time()

        for degraded, original in train_dl:
            degraded = degraded.to(device)
            original = original.to(device)
            fake     = G(degraded)

            # Train D
            opt_D.zero_grad()
            fake_buf_img = fake_buf.push_and_pop(fake.detach())
            loss_D = 0.5 * (
                crit_GAN(D(degraded, original),     torch.ones_like(D(degraded, original))) +
                crit_GAN(D(degraded, fake_buf_img), torch.zeros_like(D(degraded, fake_buf_img)))
            )
            loss_D.backward(); opt_D.step()

            # Train G
            opt_G.zero_grad()
            loss_G_GAN  = crit_GAN(D(degraded, fake), torch.ones_like(D(degraded, fake)))
            loss_G_L1   = crit_L1(fake, original) * args.lambda_l1
            loss_G_feat = feature_matching_loss(
                D, degraded, fake, original, args.lambda_feat) \
                if args.lambda_feat > 0 else torch.tensor(0.0, device=device)
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_feat
            loss_G.backward(); opt_G.step()

            ep_lD   += loss_D.item()
            ep_lG   += loss_G_GAN.item()
            ep_l1   += loss_G_L1.item() / args.lambda_l1
            ep_feat += loss_G_feat.item() / (args.lambda_feat + 1e-8)

        sched_G.step(); sched_D.step()

        n = len(train_dl)
        avg_lD, avg_lG, avg_l1, avg_feat = ep_lD/n, ep_lG/n, ep_l1/n, ep_feat/n
        cur_lr  = opt_G.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"D:{avg_lD:.4f} G:{avg_lG:.4f} "
                  f"L1:{avg_l1:.4f} Feat:{avg_feat:.4f} | "
                  f"lr:{cur_lr:.2e} | {elapsed:.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_lD:.6f},{avg_lG:.6f},"
                    f"{avg_l1:.6f},{avg_feat:.6f},{cur_lr:.6f}\n")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                dg, og = next(iter(test_dl))
                dg = dg.to(device)
                fk = G(dg)
                grid = torch.cat([dg.cpu(), fk.cpu(), og], dim=3)
                save_image(grid * 0.5 + 0.5,
                           os.path.join(sample_dir, f"epoch_{epoch:04d}.jpg"), nrow=1)
            G.train()

            if avg_lG < best_G_loss:
                best_G_loss = avg_lG
                torch.save(G.state_dict(),
                           os.path.join(ckpt_dir, "generator_best.pth"))

        if epoch % 50 == 0:
            torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict()},
                       os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pth"))

    print(f"\nTrain xong! Checkpoint tốt nhất lưu tại: {ckpt_dir}/generator_best.pth")
    print(f"Ảnh mẫu lưu tại: {sample_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pix2pix cải tiến theo chuẩn GitHub (Isola 2017)")
    parser.add_argument("--data_dir",    default="./dataset_restore")
    parser.add_argument("--output_dir",  default="./output_pix2pix")
    parser.add_argument("--epochs",      default=200, type=int)
    parser.add_argument("--batch_size",  default=4,   type=int)
    parser.add_argument("--lr",          default=2e-4, type=float)
    parser.add_argument("--lambda_l1",   default=100,  type=int)
    parser.add_argument("--lambda_feat", default=10,   type=int,
                        help="Feature Matching Loss weight (0=tắt)")
    parser.add_argument("--img_size",    default=256,  type=int)
    parser.add_argument("--save_every",  default=10,   type=int)
    args = parser.parse_args()
    train(args)

# ── Ví dụ chạy qua train_paris.py ───────────────────────────────────────────
# train_paris.py gọi step2_train_pix2pix.py với subprocess nên tự dùng được.
#
# Hoặc chạy trực tiếp:
#   python step2_train_pix2pix.py \
#       --data_dir   ./dataset_paris/grayscale \
#       --output_dir ./output_paris/grayscale \
#       --epochs 200 --lambda_l1 100 --lambda_feat 10
#
# Smoke test:
#   python step2_train_pix2pix.py --epochs 5 --batch_size 2
