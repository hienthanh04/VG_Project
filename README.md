# Pipeline Phục Hồi Tranh Van Gogh
## NST Baseline → pix2pix Restoration

---

## Cài đặt thư viện

```bash
pip install torch torchvision tqdm scikit-image opencv-python pandas matplotlib
```

---

## Bước 0 — Chuẩn bị ảnh gốc

Tải tranh Van Gogh màu gốc (gợi ý ~80–100 ảnh):
- Dataset WikiArt Van Gogh: https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings
- Hoặc dataset CycleGAN: http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip

Đặt tất cả ảnh vào thư mục:
```
vangogh_color/
    *.jpg   (hoặc *.png)
```

---

## Bước 1 — Xây dựng dataset phục hồi

```bash
# Mức suy giảm nhẹ (chỉ grayscale)
python step1_build_dataset.py --input_dir ./vangogh_color --deg_type grayscale

# Mức vừa (grayscale + blur) — khuyên dùng
python step1_build_dataset.py --input_dir ./vangogh_color --deg_type grayscale_blur

# Mức mạnh (grayscale + noise) — khuyên dùng cho báo cáo
python step1_build_dataset.py --input_dir ./vangogh_color --deg_type grayscale_noise
```

Output:
```
dataset_restore/
    train/    (~80% ảnh)
    test/     (~20% ảnh)
    samples/  (xem thử 6 ảnh mẫu)
```

**Kiểm tra ngay:** Mở thư mục `samples/` để xem ảnh ghép [degraded | original] trông đúng chưa.

---

## Bước 2 — Train pix2pix

```bash
# Smoke test (5 epoch, kiểm tra pipeline chạy được)
python step2_train_pix2pix.py --epochs 5 --batch_size 2

# Train thật (100 epoch, ~2–4 giờ tùy GPU)
python step2_train_pix2pix.py --epochs 100 --batch_size 4

# GPU mạnh
python step2_train_pix2pix.py --epochs 200 --batch_size 8
```

Output:
```
output_pix2pix/
    checkpoints/
        generator_best.pth         (model tốt nhất)
        checkpoint_epoch_XXXX.pth  (checkpoint định kỳ)
    samples/
        epoch_0010.jpg             (3 cột: degraded | restored | original)
        epoch_0020.jpg
        ...
    train_log.csv                  (loss theo epoch)
```

**Theo dõi training:** Xem ảnh trong `samples/` sau mỗi 10 epoch để biết model đang học tốt không.

---

## Bước 3 — Đánh giá kết quả

```bash
python step3_evaluate.py \
    --data_dir   ./dataset_restore/test \
    --model_path ./output_pix2pix/checkpoints/generator_best.pth \
    --output_dir ./eval_results
```

Output:
```
eval_results/
    metrics.csv            (SSIM, PSNR, Edge IoU từng ảnh)
    summary.txt            (báo cáo tổng hợp)
    ssim_comparison.jpg    (scatter plot cải thiện SSIM)
    visuals/
        *_compare.jpg      (ảnh 3 chiều: degraded | restored | original)
```

---

## Cấu trúc metric trong báo cáo

| Metric | Ý nghĩa | Càng cao càng tốt? |
|---|---|---|
| SSIM | Giữ cấu trúc/nội dung | ✅ |
| PSNR | Sai số pixel so với gốc | ✅ |
| Edge IoU | Giữ cạnh/nét | ✅ |
| SSIM delta | Cải thiện so với ảnh suy giảm | ✅ (phải > 0) |

**Điểm mạnh của hướng này:** Cả 3 metric đều có ground truth thật để so sánh,
khác với NST (không có ground truth).

---

## Vai trò từng model trong báo cáo

| Model | Vai trò | Dữ liệu |
|---|---|---|
| NST | Baseline (đã có kết quả) | 30 ảnh × 3 style |
| pix2pix | Model chính phục hồi | paired dataset |
| CycleGAN | Mở rộng (nếu còn thời gian) | unpaired |

---

## Câu đóng khung đề tài (dùng trong báo cáo)

> Trong hướng nghiên cứu chính, nhóm xây dựng bài toán phục hồi tranh
> Van Gogh từ phiên bản suy giảm chất lượng bằng cách tạo dữ liệu cặp
> từ tranh gốc màu và phiên bản grayscale/noise tương ứng. Với dạng dữ
> liệu này, nhóm ưu tiên sử dụng mô hình pix2pix để học ánh xạ phục hồi
> vì phù hợp với bài toán image-to-image có paired data. Bên cạnh đó,
> NST và CycleGAN được giữ lại như các hướng tham khảo và mở rộng.
