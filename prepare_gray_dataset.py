from pathlib import Path
from PIL import Image

# Input: folder bạn đã chuẩn bị sẵn
TRAIN_INPUT = Path(r"D:\vangogh_style_transfer\raw_content_gray\train")
TEST_INPUT = Path(r"D:\vangogh_style_transfer\raw_content_gray\test")

# Output: folder pipeline
TRAIN_A = Path(r"D:\vangogh_style_transfer\data\arles\trainA")
TEST_A = Path(r"D:\vangogh_style_transfer\data\arles\testA")

SIZE = 256

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]

def clear_folder(folder):
    if folder.exists():
        for f in folder.iterdir():
            f.unlink()

def resize_and_copy(src_folder, dst_folder):
    dst_folder.mkdir(parents=True, exist_ok=True)
    clear_folder(dst_folder)

    files = list_images(src_folder)

    for f in files:
        img = Image.open(f).convert("RGB")
        img = img.resize((SIZE, SIZE), Image.BICUBIC)
        img.save(dst_folder / f.name)

    print(f"{dst_folder}: {len(files)} ảnh")

def main():
    resize_and_copy(TRAIN_INPUT, TRAIN_A)
    resize_and_copy(TEST_INPUT, TEST_A)

    print("Resize xong!")

if __name__ == "__main__":
    main()