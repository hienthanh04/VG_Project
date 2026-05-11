from pathlib import Path

testA_dir = Path(r"D:\vangogh_style_transfer\data\arles\testA")
testB_dir = Path(r"D:\vangogh_style_transfer\data\arles\testB")
exp_dir = Path(r"D:\vangogh_style_transfer\experiments\nst_arles_v1")
exp_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

content_files = sorted([p for p in testA_dir.iterdir() if p.suffix.lower() in image_exts])
style_files = sorted([p for p in testB_dir.iterdir() if p.suffix.lower() in image_exts])

selected_contents = content_files[:30]
selected_styles = [style_files[0], style_files[6], style_files[14]]

content_txt = exp_dir / "content_list.txt"
style_txt = exp_dir / "style_list.txt"

with open(content_txt, "w", encoding="utf-8") as f:
    for p in selected_contents:
        f.write(str(p) + "\n")

with open(style_txt, "w", encoding="utf-8") as f:
    for p in selected_styles:
        f.write(str(p) + "\n")

print("Đã tạo xong:")
print(content_txt)
print(style_txt)
print("content exists:", content_txt.exists())
print("style exists:", style_txt.exists())