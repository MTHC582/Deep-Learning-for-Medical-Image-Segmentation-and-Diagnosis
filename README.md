# Skin Lesion Segmentation using U-Net Variants

A comprehensive study on skin lesion segmentation using progressive improvements to the U-Net architecture, implemented on the ISIC 2016 dataset. This project demonstrates iterative enhancements from a baseline U-Net to an advanced Attention U-Net with Grad-CAM visualization.

## 📋 Project Overview

This project implements three progressively improved models for automated melanoma detection through skin lesion segmentation:

1. **Basic U-Net**: Baseline implementation following the original U-Net architecture
2. **U-Net with Augmentations and Combined Loss**: Enhanced with combined BCE + Dice loss and learning rate scheduling along with Augmentations applied using Albumentations library
3. **Attention U-Net with Grad-CAM**: Advanced model with attention mechanisms, batch normalization, dropout, and interpretability through Grad-CAM visualizations

## 🎯 Key Results

| Model | Dice Score | IoU Score | Improvement |
|-------|-----------|-----------|-------------|
| Basic U-Net | 0.7596 | 0.6182 | Baseline |
| U-Net + Augmented Loss | 0.8701 | 0.7752 | +14.5% Dice, +25.4% IoU |
| Attention U-Net | **0.9095** | **0.8348** | +19.7% Dice, +35.0% IoU |

## 🏗️ Architecture Improvements

### Version 1: Basic U-Net
- Standard encoder-decoder architecture with skip connections
- 3 encoding and 3 decoding levels
- MaxPooling for downsampling, ConvTranspose2d for upsampling
- Binary Cross-Entropy (BCE) loss
- Simple Adam optimizer with fixed learning rate (1e-3)

### Version 2: U-Net with Augmentation and Combined Loss
- **Combined Loss Function**: BCE + Dice loss for better boundary detection
- **Learning Rate Scheduler**: ReduceLROnPlateau for adaptive learning
- **Extended Training**: 50 epochs vs 20 epochs
- **Common Augmentations**: Horizontal flip, Vertical flip, Random rotate and Random brightness Contrast

### Version 3: Attention U-Net
- **Attention Gates**: Focus on relevant features at each decoder level
- **Batch Normalization**: Improved training stability
- **Dropout Regularization**: Reduced overfitting (p=0.1)
- **Deeper Architecture**: 4 encoding/decoding levels for better feature extraction
- **Advanced Optimization**:
  - Cosine annealing LR scheduler with warm restarts
  - Gradient clipping (max_norm=1.0)
  - Kaiming weight initialization
  - Early stopping (patience=15)
- **Grad-CAM Visualization**: Model interpretability and attention analysis

## 📊 Dataset

**ISIC 2016 Challenge - Skin Lesion Segmentation**
- Training Images: 900 dermoscopic images
- Resolution: Resized to 256×256 for computational efficiency
- Train/Val Split: 80/20 (720/180)
- Data Format: RGB images with binary masks

## 🚀 How to Run

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/skin-lesion-segmentation.git
cd skin-lesion-segmentation

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the ISIC 2016 dataset:
   - Training Data: `ISBI2016_ISIC_Part1_Training_Data.zip`
   - Ground Truth: `ISBI2016_ISIC_Part1_Training_GroundTruth.zip`

2. Update the paths in the notebooks:
   ```python
   RAW_IMG_DIR = "path/to/ISBI2016_ISIC_Part1_Training_Data"
   RAW_MASK_DIR = "path/to/ISBI2016_ISIC_Part1_Training_GroundTruth"
   ```

### Running the Notebooks

**For Google Colab:**
1. Upload the notebooks to Google Colab
2. Mount your Google Drive and update dataset paths
3. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
4. Run all cells sequentially

**For Local Execution:**
```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run the notebooks in order:
# 1. 1_basic_unet.ipynb
# 2. 2_unet_augmented_loss.ipynb
# 3. 3_Attention_unet_grad_cam.ipynb
```

### Training

Each notebook follows this workflow:
1. Data preprocessing and augmentation
2. Model architecture definition
3. Training loop with validation
4. Performance evaluation (Dice & IoU metrics)
5. Visualization of predictions

**Training Configuration:**
- Batch Size: 8
- Image Size: 256×256
- Device: CUDA (GPU recommended)
- Epochs: 20 (Basic), 50 (Augmented), 60 (Attention)

## 📈 Evaluation Metrics

- **Dice Score**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index for segmentation accuracy
- **Binary Cross-Entropy Loss**: Primary optimization objective
- **Dice Loss**: Additional regularization for boundary precision

## 🔬 Technical Highlights

### Attention Mechanism
The attention gates selectively emphasize relevant features from encoder skip connections, improving localization of lesion boundaries while suppressing irrelevant background features.

### Grad-CAM Visualization
Gradient-weighted Class Activation Mapping provides insight into which regions the model focuses on during segmentation, enabling verification that the model learns clinically relevant features.

### Loss Function Design
The combined BCE + Dice loss addresses both pixel-level classification (BCE) and global region overlap (Dice), resulting in more accurate and contiguous segmentations.

## 📁 Repository Structure

```
skin-lesion-segmentation/
├── 1_basic_unet.ipynb              # Baseline U-Net implementation
├── 2_unet_augmented_loss.ipynb     # U-Net with enhanced loss
├── 3_Attention_unet_grad_cam.ipynb # Attention U-Net with visualization
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **tqdm**: Progress tracking
