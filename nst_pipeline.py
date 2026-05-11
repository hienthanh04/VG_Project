import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif")
PERIODS = ["netherlands", "paris", "arles"]
SUBDIRS = ["trainA", "trainB", "testA", "testB"]


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def count_images(folder: str | Path) -> int:
    return len(list_images(folder))


def ensure_project_structure(project_dir: str | Path) -> dict[str, Path]:
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    raw_content_dir = project_dir / "raw_content"
    xray_content_dir = data_dir / "content_xray_256"
    results_dir = project_dir / "results"

    raw_content_dir.mkdir(parents=True, exist_ok=True)
    xray_content_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for period in PERIODS:
        for subdir in SUBDIRS:
            (data_dir / period / subdir).mkdir(parents=True, exist_ok=True)
        (results_dir / "NST" / period).mkdir(parents=True, exist_ok=True)
        (results_dir / "CycleGAN" / period).mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "raw_content_dir": raw_content_dir,
        "xray_content_dir": xray_content_dir,
        "results_dir": results_dir,
    }


def split_train_test_once(folder: str | Path, ratio: float = 0.85, seed: int = 42) -> None:
    folder = Path(folder)
    files = list_images(folder)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")

    random.seed(seed)
    random.shuffle(files)
    split_idx = int(len(files) * ratio)
    train_files, test_files = files[:split_idx], files[split_idx:]

    train_dir = folder.parent / "trainB"
    test_dir = folder.parent / "testB"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for src in train_files:
        shutil.copy2(src, train_dir / src.name)
    for src in test_files:
        shutil.copy2(src, test_dir / src.name)


def resize_folder(src_dir: str | Path, dst_dir: str | Path, size: int = 256) -> int:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for src in list_images(src_dir):
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.BICUBIC)
        img.save(dst_dir / src.name)
        count += 1
    return count


def backup_and_replace_with_resized(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    for period in PERIODS:
        train_b = data_dir / period / "trainB"
        test_b = data_dir / period / "testB"
        train_b_256 = data_dir / period / "trainB_256"
        test_b_256 = data_dir / period / "testB_256"
        train_b_raw = data_dir / period / "trainB_raw"
        test_b_raw = data_dir / period / "testB_raw"

        if train_b.exists() and not train_b_raw.exists():
            train_b.rename(train_b_raw)
        if test_b.exists() and not test_b_raw.exists():
            test_b.rename(test_b_raw)

        if train_b_256.exists() and not train_b.exists():
            shutil.copytree(train_b_256, train_b)
        if test_b_256.exists() and not test_b.exists():
            shutil.copytree(test_b_256, test_b)


def xray_fake(
    image_bgr: np.ndarray,
    out_size: int = 256,
    add_noise: bool = True,
    edge_method: str = "canny",
) -> np.ndarray:
    img = cv2.resize(image_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = cv2.GaussianBlur(gray, (5, 5), 0)

    if edge_method == "canny":
        edges = cv2.Canny(base, 50, 120)
    elif edge_method == "laplacian":
        lap = cv2.Laplacian(base, cv2.CV_32F, ksize=3)
        edges = np.uint8(np.absolute(lap))
        _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("edge_method must be 'canny' or 'laplacian'")

    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edge_soft = 255 - edges
    mix = cv2.addWeighted(base, 0.75, edge_soft, 0.25, 0)

    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    mix = clahe.apply(mix)
    mix = cv2.GaussianBlur(mix, (3, 3), 0)
    mix = cv2.normalize(mix, None, 40, 220, cv2.NORM_MINMAX)

    if add_noise:
        noise = np.random.normal(0, 1.5, mix.shape).astype(np.float32)
        mix = np.clip(mix.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(mix, cv2.COLOR_GRAY2BGR)


def batch_preprocess(
    input_dir: str | Path,
    output_dir: str | Path,
    out_size: int = 256,
    add_noise: bool = True,
    edge_method: str = "canny",
) -> int:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for src in list_images(input_dir):
        img = cv2.imread(str(src))
        if img is None:
            print(f"Skip unreadable: {src.name}")
            continue
        out = xray_fake(img, out_size=out_size, add_noise=add_noise, edge_method=edge_method)
        dst = output_dir / f"{src.stem}_xray.jpg"
        cv2.imwrite(str(dst), out)
        total += 1
    return total


def split_content_to_all_periods(
    xray_content_dir: str | Path,
    data_dir: str | Path,
    ratio: float = 0.85,
    seed: int = 42,
) -> None:
    xray_content_dir = Path(xray_content_dir)
    data_dir = Path(data_dir)

    files = list_images(xray_content_dir)
    if not files:
        raise FileNotFoundError(f"No xray images found in {xray_content_dir}")

    random.seed(seed)
    random.shuffle(files)
    split_idx = int(len(files) * ratio)
    train_files, test_files = files[:split_idx], files[split_idx:]

    for period in PERIODS:
        train_a = data_dir / period / "trainA"
        test_a = data_dir / period / "testA"
        train_a.mkdir(parents=True, exist_ok=True)
        test_a.mkdir(parents=True, exist_ok=True)

        for src in train_files:
            shutil.copy2(src, train_a / src.name)
        for src in test_files:
            shutil.copy2(src, test_a / src.name)

def prepare_gray_content(
    input_dir: str | Path,
    data_dir: str | Path,
    period: str = "arles",
    size: int = 256,
    ratio: float = 0.8,
    seed: int = 42,
) -> None:
    input_dir = Path(input_dir)
    data_dir = Path(data_dir)

    files = list_images(input_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    random.seed(seed)
    random.shuffle(files)

    split_idx = int(len(files) * ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    train_a = data_dir / period / "trainA"
    test_a = data_dir / period / "testA"

    train_a.mkdir(parents=True, exist_ok=True)
    test_a.mkdir(parents=True, exist_ok=True)

    # Xóa ảnh cũ trước khi copy mới
    for folder in [train_a, test_a]:
        for f in list_images(folder):
            f.unlink()

    for src in train_files:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.BICUBIC)
        img.save(train_a / src.name)

    for src in test_files:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.BICUBIC)
        img.save(test_a / src.name)

    print(f"{period}: trainA={count_images(train_a)}, testA={count_images(test_a)}")
def show_random_grid(folder: str | Path, title: str, n: int = 8) -> None:
    images = list_images(folder)
    if not images:
        print(f"Empty: {folder}")
        return

    picks = random.sample(images, min(n, len(images)))
    plt.figure(figsize=(16, 6))
    for idx, src in enumerate(picks, start=1):
        img = Image.open(src).convert("RGB")
        plt.subplot(2, 4, idx)
        plt.imshow(img)
        plt.title(src.name[:22], fontsize=8)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NSTRunner:
    def __init__(self, img_size: int = 256, device: torch.device | None = None):
        self.img_size = img_size
        self.device = device or get_device()
        self.loader = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.unloader = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.layer_map = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1",
        }
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        self.style_layer_weights = {
            "conv1_1": 1.0,
            "conv2_1": 0.8,
            "conv3_1": 0.5,
            "conv4_1": 0.3,
            "conv5_1": 0.1,
        }

    def load_image(self, path: str | Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = self.loader(img).unsqueeze(0)
        return img.to(self.device)

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        img = tensor.detach().cpu().squeeze(0)
        img = self.unloader(img)
        img = torch.clamp(img, 0, 1)
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def get_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_map:
                feats[self.layer_map[name]] = x
        return feats

    @staticmethod
    def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        _, c, h, w = feat.size()
        feat = feat.view(c, h * w)
        gram = torch.mm(feat, feat.t())
        return gram / (c * h * w)

    @staticmethod
    def tv_loss(x: torch.Tensor) -> torch.Tensor:
        return (
            torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
            + torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )

    def stylize(
        self,
        content_path: str | Path,
        style_path: str | Path,
        steps: int = 600,
        lr: float = 0.01,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-2,
        save_path: str | Path | None = None,
        log_every: int = 50,
    ) -> Path | None:
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)

        content_feats = self.get_features(content_img)
        style_feats = self.get_features(style_img)
        style_grams = {ly: self.gram_matrix(style_feats[ly]) for ly in self.style_layers}

        generated = content_img.clone().requires_grad_(True)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        min_norm = (0.0 - mean) / std
        max_norm = (1.0 - mean) / std

        optimizer = optim.Adam([generated], lr=lr)

        for step in range(1, steps + 1):
            gen_feats = self.get_features(generated)

            c_loss = torch.mean((gen_feats["conv4_2"] - content_feats["conv4_2"]) ** 2)
            c_loss += 0.2 * torch.mean((gen_feats["conv2_1"] - content_feats["conv2_1"]) ** 2)

            s_loss = 0.0
            for ly in self.style_layers:
                g_gram = self.gram_matrix(gen_feats[ly])
                s_gram = style_grams[ly]
                s_loss += self.style_layer_weights[ly] * torch.mean((g_gram - s_gram) ** 2)

            total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * self.tv_loss(generated)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                generated.clamp_(min_norm, max_norm)

            if step % log_every == 0:
                print(
                    f"Step {step}/{steps} | total={total_loss.item():.4f} | "
                    f"c={c_loss.item():.4f} | s={s_loss.item():.4f}"
                )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.tensor_to_image(generated).save(save_path)
            print(f"Saved NST output to: {save_path}")
            return save_path
        return None


def prepare_data(args: argparse.Namespace) -> None:
    paths = ensure_project_structure(args.project_dir)

    if args.resize_style:
        for period in PERIODS:
            for split_name in ["trainB", "testB"]:
                src = paths["data_dir"] / period / split_name
                dst = paths["data_dir"] / period / f"{split_name}_256"
                if not src.exists():
                    print(f"Missing: {src}")
                    continue
                n = resize_folder(src, dst, args.size)
                print(f"{period}/{split_name}: resized {n} images")

    if args.replace_style_with_resized:
        backup_and_replace_with_resized(paths["data_dir"])
        print("Backed up original trainB/testB and replaced with resized versions when available.")

    if args.preprocess_xray:
        total = batch_preprocess(
            paths["raw_content_dir"],
            paths["xray_content_dir"],
            out_size=args.size,
            add_noise=not args.no_noise,
            edge_method=args.edge_method,
        )
        print(f"Preprocessed {total} raw content images into xray images.")

    if args.split_content:
        split_content_to_all_periods(paths["xray_content_dir"], paths["data_dir"], ratio=args.ratio, seed=args.seed)
        for period in PERIODS:
            print(
                f"{period}: trainA={count_images(paths['data_dir'] / period / 'trainA')}, "
                f"testA={count_images(paths['data_dir'] / period / 'testA')}"
            )


def run_nst(args: argparse.Namespace) -> None:
    paths = ensure_project_structure(args.project_dir)
    period = args.period

    test_a_dir = paths["data_dir"] / period / "testA"
    test_b_dir = paths["data_dir"] / period / "testB"
    content_candidates = list_images(test_a_dir)
    style_candidates = list_images(test_b_dir)

    if args.content is not None:
        content_path = Path(args.content)
    else:
        if not content_candidates:
            raise FileNotFoundError(f"No content images in {test_a_dir}")
        content_path = random.choice(content_candidates)

    if args.style is not None:
        style_path = Path(args.style)
    else:
        if not style_candidates:
            raise FileNotFoundError(f"No style images in {test_b_dir}")
        style_path = random.choice(style_candidates)

    save_name = args.output_name or f"nst_{period}_{content_path.stem}_{style_path.stem}.jpg"
    save_path = paths["results_dir"] / "NST" / period / save_name

    print(f"Device      : {get_device()}")
    print(f"Period      : {period}")
    print(f"Content path: {content_path}")
    print(f"Style path  : {style_path}")
    print(f"Save path   : {save_path}")

    runner = NSTRunner(img_size=args.size)
    runner.stylize(
        content_path=content_path,
        style_path=style_path,
        steps=args.steps,
        lr=args.lr,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        save_path=save_path,
        log_every=args.log_every,
    )
def read_list_file(txt_path: str | Path) -> list[Path]:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"List file not found: {txt_path}")

    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Path(line))
    return items

def run_nst_batch(args: argparse.Namespace) -> None:
    paths = ensure_project_structure(args.project_dir)
    period = args.period

    content_list = read_list_file(args.content_list)
    style_list = read_list_file(args.style_list)

    save_dir = paths["results_dir"] / "NST" / period
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device       : {get_device()}")
    print(f"Period       : {period}")
    print(f"Content imgs : {len(content_list)}")
    print(f"Style imgs   : {len(style_list)}")
    print(f"Save dir     : {save_dir}")

    runner = NSTRunner(img_size=args.size)

    total_jobs = len(content_list) * len(style_list)
    job_idx = 0

    for content_path in content_list:
        if not content_path.exists():
            print(f"Skip missing content: {content_path}")
            continue

        for style_path in style_list:
            if not style_path.exists():
                print(f"Skip missing style: {style_path}")
                continue

            job_idx += 1
            output_name = f"{content_path.stem}__{style_path.stem}__nst.jpg"
            save_path = save_dir / output_name

            print(f"\n[{job_idx}/{total_jobs}]")
            print(f"Content: {content_path.name}")
            print(f"Style  : {style_path.name}")

            runner.stylize(
                content_path=content_path,
                style_path=style_path,
                steps=args.steps,
                lr=args.lr,
                content_weight=args.content_weight,
                style_weight=args.style_weight,
                tv_weight=args.tv_weight,
                save_path=save_path,
                log_every=args.log_every,
            )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert your NST Colab notebook into a reusable Python script.")
    parser.add_argument("--project-dir", default="./vangogh_style_transfer", help="Root project directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-data", help="Prepare style/content data like in the notebook")
    prep.add_argument("--size", type=int, default=256)
    prep.add_argument("--ratio", type=float, default=0.85)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--edge-method", choices=["canny", "laplacian"], default="canny")
    prep.add_argument("--no-noise", action="store_true")
    prep.add_argument("--resize-style", action="store_true")
    prep.add_argument("--replace-style-with-resized", action="store_true")
    prep.add_argument("--preprocess-xray", action="store_true")
    prep.add_argument("--split-content", action="store_true")
    prep.set_defaults(func=prepare_data)

    nst = subparsers.add_parser("run-nst", help="Run NST on one content image and one style image")
    nst.add_argument("--period", choices=PERIODS, default="arles")
    nst.add_argument("--content", help="Optional content image path")
    nst.add_argument("--style", help="Optional style image path")
    nst.add_argument("--size", type=int, default=256)
    nst.add_argument("--steps", type=int, default=600)
    nst.add_argument("--lr", type=float, default=0.01)
    nst.add_argument("--content-weight", type=float, default=1.0)
    nst.add_argument("--style-weight", type=float, default=1e5)
    nst.add_argument("--tv-weight", type=float, default=1e-2)
    nst.add_argument("--log-every", type=int, default=50)
    nst.add_argument("--output-name", help="Custom output filename")
    nst.set_defaults(func=run_nst)

    gray_prep = subparsers.add_parser("prepare-gray", help="Prepare grayscale content into trainA/testA")
    gray_prep.add_argument("--input-dir", required=True, help="Folder containing raw grayscale images")
    gray_prep.add_argument("--period", choices=PERIODS, default="arles")
    gray_prep.add_argument("--size", type=int, default=256)
    gray_prep.add_argument("--ratio", type=float, default=0.8)
    gray_prep.add_argument("--seed", type=int, default=42)
    gray_prep.set_defaults(func=run_prepare_gray)

    nst_batch = subparsers.add_parser("run-nst-batch", help="Run NST on fixed content/style lists")
    nst_batch.add_argument("--period", choices=PERIODS, default="arles")
    nst_batch.add_argument("--content-list", required=True, help="Path to content_list.txt")
    nst_batch.add_argument("--style-list", required=True, help="Path to style_list.txt")
    nst_batch.add_argument("--size", type=int, default=256)
    nst_batch.add_argument("--steps", type=int, default=600)
    nst_batch.add_argument("--lr", type=float, default=0.01)
    nst_batch.add_argument("--content-weight", type=float, default=1.0)
    nst_batch.add_argument("--style-weight", type=float, default=1e5)
    nst_batch.add_argument("--tv-weight", type=float, default=1e-2)
    nst_batch.add_argument("--log-every", type=int, default=50)
    nst_batch.set_defaults(func=run_nst_batch)
    return parser

def run_prepare_gray(args: argparse.Namespace) -> None:
    paths = ensure_project_structure(args.project_dir)
    prepare_gray_content(
        input_dir=args.input_dir,
        data_dir=paths["data_dir"],
        period=args.period,
        size=args.size,
        ratio=args.ratio,
        seed=args.seed,
    )
    
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
