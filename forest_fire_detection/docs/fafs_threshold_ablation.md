# Attention Binarization Threshold Ablation Study (N=50)

This report presents a systematic ablation study on the binarization threshold ($t$) used to extract binary attention focus masks from continuous self-attention rollout maps.

## 1. Threshold Comparison Table

| Threshold ($t$) | Baseline ViT FAFS (Mean IoU) | Baseline Std Dev | Improved ViT FAFS (Mean IoU) | Improved Std Dev |
| :---: | :---: | :---: | :---: | :---: |
| **0.1** | 0.2448 | ±0.1324 | 0.2346 | ±0.1220 |
| **0.2** | 0.2995 | ±0.1659 | 0.2793 | ±0.1640 |
| **0.3** | 0.3475 | ±0.1921 | 0.3150 | ±0.1850 |
| **0.4** | 0.3407 | ±0.1993 | 0.3087 | ±0.1942 |
| **0.5** | 0.2727 | ±0.1860 | 0.2611 | ±0.1959 |
| **0.6** | 0.1850 | ±0.1480 | 0.1843 | ±0.1639 |
| **0.7** | 0.1004 | ±0.1000 | 0.1057 | ±0.1129 |
| **0.8** | 0.0437 | ±0.0553 | 0.0522 | ±0.0660 |
| **0.9** | 0.0117 | ±0.0174 | 0.0135 | ±0.0205 |

## 2. Key Scientific Observations

- **Optimal Thresholds**: For both models, the FAFS score (IoU) is strongly dependent on the chosen binarization threshold. The optimal threshold that maximizes mean IoU against human segmentations is key to reporting a fair faithfulness metric.
- **Analysis of Attention Spread**: A lower threshold (e.g. $0.1$ to $0.3$) yields a larger binarized attention region, which overlaps more broadly with the fire/smoke mask, but may include some foliage. A higher threshold (e.g. $0.6$ to $0.9$) narrows down the focus to the most critical 'peaks' of attention.
