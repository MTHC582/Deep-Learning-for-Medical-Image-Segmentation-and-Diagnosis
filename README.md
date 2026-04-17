# Skin Lesion Segmentation using Progressive U-Net Variants

A structured study on skin lesion segmentation using progressively improved U-Net architectures on the ISIC 2016 dataset. The project demonstrates systematic improvements from a baseline U-Net to an enhanced Attention U-Net with interpretability using Grad-CAM.

---

## Project Overview

This project follows a three-stage experimental progression to improve segmentation performance:

1. **Stage 1 — Baseline U-Net**

   * Standard encoder-decoder architecture
   * Binary Cross-Entropy (BCE) loss
   * No augmentation

2. **Stage 2 — U-Net with Augmentation and Combined Loss**

   * Data augmentation using Albumentations
   * Combined BCE + Dice loss
   * Extended training

3. **Stage 3 — Improved Attention U-Net**

   * Attention gates on skip connections
   * Batch Normalization and Spatial Dropout
   * Deeper architecture (4 encoder-decoder levels)
   * AdamW optimizer with weight decay
   * Cosine Annealing with Warm Restarts (SGDR)
   * Gradient clipping and early stopping
   * Grad-CAM for interpretability

---

## Key Results

| Model                                 | Dice Score | IoU Score  |
| ------------------------------------- | ---------- | ---------- |
| Stage 1 — Baseline U-Net              | 0.7596     | 0.6182     |
| Stage 2 — U-Net + Augmentation + Dice | 0.8701     | 0.7752     |
| Stage 3 — Attention U-Net             | **0.9095** | **0.8348** |

**Total Improvement:**

* Dice: +0.1499 (+19.7%)
* IoU: +0.2166 (+35.0%)

---

## Dataset

**ISIC 2016 (ISBI Part 1) — Skin Lesion Segmentation**

* Total Images: 900
* Train/Validation Split: 80% / 20% (720 / 180)
* Input Size: 256 × 256 (resized)
* Channels: RGB (3-channel)
* Output: Binary segmentation mask
* Class Imbalance: ~16% lesion, ~84% background

---

## Methodology

### Stage 1 — Baseline

* 3-level U-Net (64 → 128 → 256 → 512)
* BCEWithLogitsLoss
* Adam optimizer (lr = 1e-3)
* No regularization or augmentation

### Stage 2 — Improvements

* Augmentations:

  * Horizontal flip
  * Vertical flip
  * Random rotation (90°)
  * Brightness and contrast adjustment
* Combined loss:

  * BCE + Dice Loss
* Training extended to 50 epochs

### Stage 3 — Advanced Model

* 4-level U-Net (up to 1024 channels)
* Attention gates in skip connections
* BatchNorm + Spatial Dropout (p = 0.15)
* Kaiming weight initialization
* AdamW optimizer (lr = 3e-4, wd = 1e-4)
* CosineAnnealingWarmRestarts scheduler
* Gradient clipping (max_norm = 1.0)
* Early stopping (patience = 15)

---

## Interpretability — Grad-CAM

Grad-CAM is applied to the final model to visualize important regions influencing predictions.

Findings:

* Strong activation in lesion regions
* Suppression of background and artefacts
* Confirms meaningful feature learning

---

## Repository Structure

```
skin-lesion-segmentation/
├── 1_basic_unet.ipynb
├── 2_unet_augmented_loss.ipynb
├── 3_attention_unet_grad_cam.ipynb
├── requirements.txt
└── README.md
```

---

## How to Run (Google Collab recommended)

### 1. Clone Repository

```bash
git clone https://github.com/MTHC582/Deep-Learning-for-Medical-Image-Segmentation-and-Diagnosis.git
cd Deep-Learning-for-Medical-Image-Segmentation-and-Diagnosis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

Download and unzip dataset from:
https://drive.google.com/drive/folders/1n26atMzMNBzVBD27l2w8CrNH-q_eMkwW

First two cells in each notebook are for unzipping. It is recommended to unzip manually and ignore the first 2 cells.

Update paths in 3rd cell of the notebooks (in accordance with manual unzipping):
```python
RAW_IMG_DIR = "path/to/images"
RAW_MASK_DIR = "path/to/masks"
```

---

### 4. Run Notebooks

Run in order:

1. `1_basic_unet.ipynb`
2. `2_unet_augmented_loss.ipynb`
3. `3_attention_unet_grad_cam.ipynb`

**Recommended:** Use Google Colab with GPU enabled.

---

## Evaluation Metrics

* **Dice Score** — overlap between prediction and ground truth
* **IoU (Intersection over Union)** — segmentation accuracy
* **BCE Loss** — pixel-wise classification
* **Dice Loss** — region-level overlap optimization

---

## Key Insights

* Loss function design significantly impacts performance
* Data augmentation improves generalization on small datasets
* Attention mechanisms enhance feature selection
* Training strategies (AdamW, SGDR, regularization) improve convergence
* Interpretability methods validate model behavior

---

## Limitations

* Results are based on a validation split, not the official ISIC test set
* Performance may vary with different dataset splits
* Small lesions and artefacts remain challenging cases

---

## Conclusion

This project demonstrates that systematic improvements in architecture, loss functions, and training strategies can significantly enhance segmentation performance. The final Attention U-Net achieves competitive results while maintaining interpretability through Grad-CAM, making it suitable for further research in medical image analysis.

---
