# Zero-Shot Cross-Dataset Generalization Report

This report evaluates the **out-of-distribution (OOD) generalizability** of the fine-tuned forest fire detection models. The models, trained on aerial and forest wildfire images, are tested in a zero-shot manner (without retraining) on **FireNet validation images** (ground-based urban and domestic fires) combined with local non-fire backgrounds.

## 1. Quantitative Generalization Summary

| Model | Dataset Domain | Overall Accuracy | Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **ResNet-18 Baseline** | OOD (FireNet + local) | **0.9395** | Fire<br>Non-Fire | 1.0000<br>0.9183 | 0.8111<br>1.0000 | 0.8957<br>0.9574 | 90.0<br>191.0 |
| **Improved ViT** | OOD (FireNet + local) | **0.9822** | Fire<br>Non-Fire | 0.9670<br>0.9895 | 0.9778<br>0.9843 | 0.9724<br>0.9869 | 90.0<br>191.0 |

## 2. Confusion Matrices

### ResNet-18 Confusion Matrix
```
Actual \ Predicted | Predicted Fire | Predicted Non-Fire
-------------------|----------------|-------------------
Actual Fire        | 73             | 17               
Actual Non-Fire    | 0              | 191              
```

### Improved ViT Confusion Matrix
```
Actual \ Predicted | Predicted Fire | Predicted Non-Fire
-------------------|----------------|-------------------
Actual Fire        | 88             | 2                
Actual Non-Fire    | 3              | 188              
```

## 3. Scientific Analysis & Domain Generalization Insights

- **Domain Shift Impact**: The models were trained exclusively on high-altitude, wide-area forest fire images, where features are dominated by large-scale smoke plumes, tree silhouettes, and forest foliage. Testing them on **FireNet** represents a significant shift to ground-level views, containing indoor flame sources, candles, burning vehicles, and urban close-up environments. Despite this extreme domain shift, both models maintain high generalization performance, proving that they have learned generalizable fire features (e.g. flame combustion textures, illumination, color hues) rather than just memorizing local forest backgrounds.

- **Model Performance Discrepancy (ViT vs. CNN)**:
  - The **Improved ViT** achieved a zero-shot accuracy of **0.9822** with a Fire recall (sensitivity) of **0.9778** (only 2 fire images missed out of 90).
  - The **ResNet-18 Baseline** achieved a zero-shot accuracy of **0.9395** with a Fire recall of **0.8111** (missing 17 fire images).
  - The Vision Transformer shows superior generalization capability under severe domain shift. The global self-attention mechanism in the ViT enables it to model contextual features and long-range dependencies, making it less susceptible to localized changes in scale, viewpoint, and background textures compared to the local feature filters of the ResNet CNN.
