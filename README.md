# README

## Project Overview
This repository contains code for **Forest Fire Detection** using a baseline **ResNet‑18** CNN and an **Improved ViT‑Tiny** transformer.  The pipeline includes data preprocessing, model training, evaluation, explainability (FAFS/Grad‑CAM), zero‑shot out‑of‑distribution testing, and computational‑complexity profiling.
# 🔥 Forest Fire Detection using Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

A deep learning-based forest fire detection system using **Vision Transformers (ViT)** achieving **94.82% accuracy** on test data. This project leverages transfer learning and state-of-the-art computer vision techniques for real-time fire detection.

## 📋 Table of Contents
- [Overview](#overview)
- [Models & Algorithms](#models--algorithms)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [Citation](#citation)

## 🎯 Overview

Forest fires pose significant threats to ecosystems, wildlife, and human settlements. Early detection is crucial for effective fire management. This project implements an automated fire detection system using deep learning to classify images as **fire** or **no-fire** scenarios.

**Key Features:**
- ✅ Vision Transformer (ViT) architecture for superior feature extraction
- ✅ Transfer learning from ImageNet pretrained models
- ✅ 94.82% test accuracy
- ✅ Data augmentation for robust generalization
- ✅ Multiple model architectures (ViT, ResNet18)
- ✅ Utility scripts for mask generation

## 🧠 Models & Algorithms

### Vision Transformer (ViT)

**Architecture:** `vit_tiny_patch16_224`

Vision Transformers represent a paradigm shift from traditional CNNs by applying the transformer architecture (originally designed for NLP) to computer vision tasks.

#### How ViT Works:

1. **Patch Embedding**
   - Input image is divided into fixed-size patches (16×16 pixels)
   - Each patch is linearly embedded into a vector
   - Positional encodings are added to retain spatial information

2. **Transformer Encoder**
   - Multi-head self-attention mechanism captures global dependencies
   - Unlike CNNs, ViT can model long-range relationships from the first layer
   - Layer normalization and MLP blocks process the attended features

3. **Classification Head**
   - A special [CLS] token aggregates information from all patches
   - Final MLP layer outputs class probabilities (fire/no-fire)

**Mathematical Foundation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

where:
- Q (Query), K (Key), V (Value) are learned projections
- d_k is the dimension of the key vectors
- Multi-head attention runs this in parallel with different learned projections
```

**Why ViT for Fire Detection?**
- **Global Context**: Captures entire image context, crucial for detecting fire patterns
- **Spatial Relationships**: Understands how smoke, flames, and surroundings interact
- **Transfer Learning**: Pretrained on ImageNet (1.2M images) provides robust feature extraction
- **Scalability**: Efficient for varying image sizes and conditions

### ResNet18 (Comparison Model)

**Architecture:** Residual Neural Network with 18 layers

ResNet introduced skip connections (residual connections) to solve the vanishing gradient problem in deep networks.

**Key Components:**
- **Residual Blocks**: `F(x) + x` where F(x) is learned residual mapping
- **Batch Normalization**: Stabilizes training
- **Global Average Pooling**: Reduces spatial dimensions before classification

**Comparison:**
| Model | Parameters | Test Accuracy | Inference Speed |
|-------|-----------|---------------|-----------------|
| ViT-Tiny | ~5.7M | 94.82% | Fast |
| ResNet18 | ~11.7M | ~92-93% (est.) | Very Fast |

### Training Strategy

**Transfer Learning Pipeline:**
1. **Initialization**: Load pretrained weights from ImageNet
2. **Feature Freezing**: Freeze all layers except classification head
3. **Fine-tuning**: Train only the final layer for fire/no-fire classification
4. **Optimization**: AdamW optimizer with learning rate 1e-4

**Loss Function:**
```python
CrossEntropyLoss = -Σ y_i * log(ŷ_i)

where:
- y_i is the true label (one-hot encoded)
- ŷ_i is the predicted probability
```

### Data Augmentation Techniques

**Training Augmentations:**
- **Random Horizontal Flip**: Simulates different camera angles
- **Color Jitter**: 
  - Brightness: ±40%
  - Contrast: ±40%
  - Saturation: ±40%
  - Hue: ±10%
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Purpose**: Augmentation prevents overfitting and improves model generalization to various lighting conditions, weather, and fire intensities.

## 📊 Dataset

**Source:** [Roboflow Universe - master_proje](https://universe.roboflow.com/karabuk-un/master_proje/dataset/1)

- **Total Images:** 1,690
- **Format:** YOLOv8 annotations
- **Classes:** Binary (Fire / No Fire)
- **Resolution:** Resized to 640×640
- **License:** CC BY 4.0

**Data Split:**
- Training: ~70%
- Validation: ~20%
- Test: ~10%

## 🎯 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.82%** |
| Precision | ~95% |
| Recall | ~94% |
| F1-Score | ~94.5% |

### Model Checkpoints

Three trained models are available:
1. `vit_forest_fire.pth` - Initial ViT model
2. `resnet18_fire.pth` - ResNet18 baseline (44.8 MB)
3. `vit_forest_fire_improved.pth` - Optimized ViT (22.2 MB)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/forest-fire-detection.git
cd forest-fire-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision timm scikit-learn matplotlib opencv-python pillow
```

## 💻 Usage

### Training

```python
# Open and run the Jupyter notebook
jupyter notebook fire_detection.ipynb

# Or run training script
python train.py --model vit --epochs 10 --batch-size 32
```

### Inference

```python
import torch
from torchvision import transforms
from PIL import Image
from timm import create_model

# Load model
model = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('vit_forest_fire.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
    
print("Fire Detected!" if prediction == 0 else "No Fire")
```

### Mask Generation

```bash
# Convert JSON annotations to PNG masks
python jsontopng.py
```

## 📁 Project Structure

```
forest-fire-detection/
├── fire_detection.ipynb          # Main training notebook
├── train.py                       # Training script
├── jsontopng.py                   # Mask generation utility
├── data.yaml                      # Dataset configuration
├── models/
│   ├── vit_forest_fire.pth
│   ├── resnet18_fire.pth
│   └── vit_forest_fire_improved.pth
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── results/                       # Training logs and visualizations
└── README.md
```

## 🔬 Technical Details

### Hardware Requirements
- **Minimum:** 8GB RAM, CPU
- **Recommended:** 16GB RAM, NVIDIA GPU with 6GB+ VRAM

### Software Stack
- **Framework:** PyTorch 2.5.1
- **Model Library:** timm (PyTorch Image Models)
- **Data Processing:** torchvision, OpenCV, Pillow
- **Evaluation:** scikit-learn

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Epochs | 3-5 |
| Image Size | 128×128 (training), 224×224 (ViT standard) |
| Weight Decay | 0.01 |

## 🔮 Future Improvements

- [ ] **Real-time Detection**: Implement video stream processing
- [ ] **Model Deployment**: Convert to ONNX/TensorRT for edge devices
- [ ] **Explainability**: Add Grad-CAM visualizations
- [ ] **Multi-class**: Extend to classify fire severity levels
- [ ] **Mobile App**: Develop iOS/Android application
- [ ] **API Service**: Create REST API for integration
- [ ] **Ensemble Methods**: Combine ViT + ResNet predictions

## 📚 Citation

If you use this project, please cite:

```bibtex
@misc{forest_fire_detection_2023,
  author = {Your Name},
  title = {Forest Fire Detection using Vision Transformers},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/forest-fire-detection}
}
```

**Dataset Citation:**
```bibtex
@misc{master_proje_dataset,
  title = {master_proje Dataset},
  author = {Karabuk University},
  year = {2023},
  url = {https://universe.roboflow.com/karabuk-un/master_proje/dataset/1},
  license = {CC BY 4.0}
}
```

## 📄 License

This project is licensed under CC BY 4.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

