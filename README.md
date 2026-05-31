# Van Gogh Artwork Restoration Project

## 1. Project Objective
This project aims to **restore degraded Van Gogh paintings** using a Pix2Pix model, focusing on three painting periods: **Arles, Paris, Netherlands**. The main goals are:

- Reconstruct degraded images (grayscale, blur, noise, combined)
- Improve restoration quality measured by **SSIM / PSNR**
- Build a structured data pipeline with profiling and dataset quality monitoring

---

## 2. Dataset

- Total images: 426
- Distribution per period:
  - Arles: 188 images
  - Paris: 153 images
  - Netherlands: 85 images
- All images have been **resized to 256×256 px** and converted to **JPG**
- Dataset has been **verified**: no corrupted or duplicate images



## 3. Data Profiling & Analysis

The dataset was analyzed to extract important features:

- **Brightness, Saturation, Edge Density** per painting period
- **Width / Height Distribution**
- **Total Images per Period**

**Illustrative charts:**

- **Boxplots of Brightness/Saturation/Edge Density by Period**
<img width="435" height="285" alt="image" src="https://github.com/user-attachments/assets/614fc2b0-5a12-465a-9176-c8952153b553" />
<img width="442" height="283" alt="image" src="https://github.com/user-attachments/assets/59e9401c-48ea-49a1-be93-a740f7a1f780" />
<img width="431" height="281" alt="image" src="https://github.com/user-attachments/assets/f326cc0c-8045-49c1-ae23-7858a7a72e2b" />


- **Histogram of Width / Height**
<img width="515" height="285" alt="image" src="https://github.com/user-attachments/assets/8db4503c-6f02-48e9-97a0-203f49d2f1cc" />
<img width="513" height="279" alt="image" src="https://github.com/user-attachments/assets/b21ef6ce-217a-4d0a-8608-a9c9ad46dfaf" />


- **Total Images per Period**
<img width="426" height="283" alt="image" src="https://github.com/user-attachments/assets/06242ae5-f88c-4df2-b765-6843de77c5c9" />

**Purpose of profiling:** to understand the dataset and guide preprocessing, augmentation, and balancing strategies.

---

## 4. Preprocessing & Augmentation

- Resized all images to **256×256 px**
- Applied **rotation, flipping, and other augmentations** to increase dataset size
- Generated **degraded datasets**: grayscale, blur, noise, gray+blur+noise
- Aspect ratio maintained using **padding or cropping** when necessary

---

## 5. Model Training & Evaluation

- Model: **Pix2Pix**
- Input: degraded images
- Output: restored images
- Evaluation metrics: **SSIM, PSNR, Edge IoU**
- Performance comparison between model versions:

1. **SSIM Improvement Across Versions**
   - The following chart shows how SSIM improved from version 1 (v1) to version 5 (v5) after tuning model architecture and hyperparameters.

   <img width="1076" height="648" alt="image" src="https://github.com/user-attachments/assets/006a430b-7692-4fa6-b370-272d06110275" />


2. **SSIM Comparison Across Degradation Types**
   - Comparison of SSIM between Light and Heavy degraded images (grayscale, blur, blur+noise, gray+blur+noise) for versions v1 and v2.

   <img width="1075" height="648" alt="image" src="https://github.com/user-attachments/assets/2942dcd7-1369-49bf-bdac-56d31440ba12" />


3. **SSIM Across Painting Periods**
   - SSIM trends across the three Van Gogh periods (Arles, Paris, Netherlands) under different degradation types.

   <img width="1076" height="648" alt="image" src="https://github.com/user-attachments/assets/5aaf2818-a6d5-4906-96b3-5cb519776976" />

---

## 6. Conclusion

- Data pipeline ensures a **consistent and high-quality dataset** for training
- Model performance improved with **hyperparameter tuning**
- Charts and CSV files provide **dataset monitoring and model evaluation**, demonstrating a **data product mindset**
- Future work: extend to more degraded types or additional painting periods

---

## 7. Links

- [Colab Notebooks] https://colab.research.google.com/drive/1v4iQ06UaMcVnS0M1JPz9CtZXZvfPx5qb?hl=vi#scrollTo=d_-RB7mwqWm8
