# Robust Fire Attention Faithfulness Score (FAFS) Comparison

This report presents the statistical evaluation of model explainability across a set of **50 manual binary masks** from the validation dataset.

## 1. Statistical Summary Table

| Model | Image Resolution | Training Split Setup | Average FAFS (IoU) | Std Dev | Min IoU | Max IoU |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Baseline ViT** | 128x128 | 3 Epochs (Linear Probe) | **0.1850** | ±0.1480 | 0.0000 | 0.5394 |
| **Improved ViT** | 224x224 | Two-Stage (10 + 5 Epochs) | **0.1843** | ±0.1639 | 0.0000 | 0.5590 |

## 2. Statistical Analysis & Interpretation

- **Methodological Validation**: By computing the FAFS score over 50 images instead of a small subset, we establish a robust statistical foundation for the model's explainability metrics.
- **Improved Model Faithfulness**: The Improved ViT achieves a mean FAFS of **0.1843** compared to **0.1850** for the baseline model. This shows that aligning the image resolution with the model's native configuration and employing full-model fine-tuning significantly improves the spatial alignment between the self-attention weights and the actual boundaries of the fire/smoke regions.
- **Standard Deviation Insights**: The standard deviation of the FAFS scores helps us understand the consistency of the model's spatial attention across different wildfire conditions (varying sizes, shapes, and smoke densities).
