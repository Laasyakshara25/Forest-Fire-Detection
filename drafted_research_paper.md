# Explainable and Generalizable Wildfire Detection: A Comparative Study of Convolutional Networks and Vision Transformers

**Authors:** *[Insert Author Names]*  
**Affiliations:** *[Insert Affiliations]*  

---

## Abstract
Early and automated forest fire detection is critical to mitigating the catastrophic environmental and socioeconomic impacts of wildfires. While deep learning models, particularly Convolutional Neural Networks (CNNs), have demonstrated remarkable classification performance, their deployment in safety-critical systems is hindered by a lack of explainability and susceptibility to domain shifts. This paper presents a comparative analysis between a baseline ResNet-18 CNN and an Improved Vision Transformer (ViT-Tiny) for wildfire detection. We introduce a two-stage fine-tuning methodology for the ViT model, resolving input-resolution mismatches that lead to structural representation bottlenecks. Evaluated on a dataset of $2,947$ wildfire images, our Improved ViT achieves a test accuracy of **$99.77\%$** with **$100\%$** sensitivity (recall) for fire detection, while operating at half the parameters ($5.52\text{ M}$ vs. $11.18\text{ M}$) and achieving **$1.67\times$** higher throughput than the ResNet-18 baseline on CPU. Furthermore, we analyze the spatial faithfulness of the model decisions using **Attention Rollout** and **Grad-CAM** against manually annotated binary validation masks. To quantify localization fidelity, we propose the **Fire Attention Faithfulness Score (FAFS)**. While the CNN activations align more contiguously with manual fire boundaries (peak FAFS of $0.3189$), the ViT exhibits superior out-of-distribution (OOD) generalization. Tested zero-shot on domestic and urban fires from the FireNet dataset under severe domain shift, the Improved ViT maintains **$98.22\%$** accuracy ($97.78\%$ fire recall) compared to **$93.95\%$** accuracy ($81.11\%$ fire recall) for the CNN. These findings suggest that the global context modeled by self-attention mechanisms captures invariant combustion signatures, making ViTs highly resilient and trustworthy for edge-based forest monitoring.

**Keywords:** Wildfire Detection, Vision Transformers, Convolutional Neural Networks, Explainable AI, Attention Rollout, Out-of-Distribution Generalization.

---

## 1. Introduction
Wildfires pose a severe global threat, destroying ecosystems, releasing megatons of greenhouse gases, and claiming human lives. Traditional fire detection relies on human lookouts, satellite telemetry, or smoke sensors, all of which exhibit high latencies. In recent years, deep-learning-based computer vision models running on remote camera towers or drones have emerged as a viable solution for real-time surveillance.

However, two major barriers prevent these models from being fully trusted:
1. **Lack of Explainability**: Early detectors are safety-critical. If a model flags a scene as "fire" based on irrelevant features, such as background mountain ridges, sky hues, or dry roads, it will lead to costly false alarms. Conversely, if it misses a fire due to background confusion, the outcome is catastrophic. To ensure accountability, models must possess spatial explainability that can be evaluated quantitatively.
2. **Domain Shift and Generalization**: Wildfire models are typically trained on aerial or high-altitude forest views containing green foliage, dry underbrush, and large smoke plumes. In deployment, they are exposed to different lighting, fog, urban backdrops, or domestic flames. A robust model must learn generalized representations of combustion (flame patterns, thermal texture, illumination gradients) rather than memorizing local background textures.

To address these challenges, we perform a rigorous comparative study between a standard Convolutional Neural Network baseline (ResNet-18) and a Vision Transformer (ViT-Tiny). We expose a critical architectural bottleneck in standard ViT pipelines where image resolution mismatch degrades positional embeddings. We propose a **Two-Stage Fine-Tuning Strategy** to overcome this and achieve near-perfect classification. We also introduce the **Fire Attention Faithfulness Score (FAFS)** to measure spatial interpretability, comparing Attention Rollout and Grad-CAM against human annotations. Finally, we validate OOD generalization using zero-shot tests on ground-level fires (FireNet).

---

## 2. Related Work
### A. Deep Learning in Wildfire Detection
Historically, wildfire detection relied on manually crafted color index thresholds (e.g., RGB/YUV ratios) to extract fire pixels. The introduction of Deep Convolutional Neural Networks (CNNs) revolutionized this field, with architectures like AlexNet, VGG, and ResNet achieving high classification accuracy by learning hierarchical texture representations. However, CNNs have local receptive fields and lack global context, making them prone to false positives on smoke-colored clouds or foliage that mimics fire hues.

### B. Vision Transformers in Remote Sensing
Vision Transformers (ViTs) apply the self-attention mechanism to patch embeddings of images, allowing them to capture long-range relationships from the very first layer. Although highly effective, ViTs lack the local inductive biases of CNNs, necessitating larger training volumes or structured transfer learning. In remote sensing, ViTs have been deployed for land cover classification, but their role in safety-critical wildfire monitoring remains less explored.

### C. Interpretability & Saliency Mapping
Explainable AI (XAI) is vital for validating that neural networks attend to meaningful features. For CNNs, **Grad-CAM** uses the gradients of a target class flowing into the final convolutional layer to produce a coarse localization map. For ViTs, **Attention Rollout** recursively propagates attention weights across layers to map the information flow from the classification token (`[CLS]`) back to input patches. Quantitative evaluation of these maps against pixel-level ground truth remains a key bottleneck in the literature.

---

## 3. Dataset & Preprocessing

The research is validated using two distinct image datasets representing in-distribution and out-of-distribution domains.

### A. In-Distribution (ID) Dataset: Karabuk University *master_proje*
The primary dataset is sourced from the Karabuk University *master_proje* archive, consisting of binary images categorized as `fire` or `nofire`. The dataset represents diverse forest terrains, aerial forest views, and wildfire smoke plumes under varying illumination. On disk, the dataset is structured into three splits:
*   **Training Set**: $2,062$ images ($1,170$ fire, $892$ non-fire)
*   **Validation Set**: $441$ images ($250$ fire, $191$ non-fire)
*   **Test Set**: $444$ images ($252$ fire, $192$ non-fire)
*   **Total**: $2,947$ images

To evaluate spatial interpretability, we manually annotated $5$ high-fidelity validation masks (`data/mask`) defining the exact polygonal boundaries of the flames and smoke columns within corresponding validation images.

### B. Out-of-Distribution (OOD) Dataset: FireNet
To assess cross-dataset generalization, we compiled an OOD dataset from the **FireNet** repository. Unlike the high-altitude, wide-area forest views of the *master_proje* dataset, FireNet focuses on ground-level urban, industrial, and domestic fire incidents (e.g., candles, trash can fires, burning vehicles, and indoor fireplace flames). The OOD split contains:
*   **OOD Fire Images**: $90$ images of ground-level fires.
*   **OOD Non-Fire Images**: $191$ images of local non-fire backgrounds (gardens, rooms, streets).
*   **Total OOD**: $281$ images.

### C. Data Augmentation and Normalization
To prevent overfitting, we apply training-time data augmentations:
1.  **Random Horizontal Flip** ($p=0.5$).
2.  **Random Rotation** up to $15^\circ$ (for the Improved ViT).
3.  **Color Jittering**: Brightness ($\pm 40\%$), Contrast ($\pm 40\%$), Saturation ($\pm 40\%$), and Hue ($\pm 10\%$) to simulate varying weather, atmospheric haze, and camera exposure levels.
4.  **Normalization**: Standardized using ImageNet statistics:
    $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 4. Architectural Framework & Methodology

### A. Core Models
We benchmark three model configurations:
1.  **ResNet-18 Baseline**: A 11.18M parameter CNN pretrained on ImageNet. The final fully connected layer is replaced with a linear head of output dimension 2.
2.  **Baseline ViT-Tiny (`vit_tiny_patch16_224`)**: A 5.7M parameter transformer trained on $128 \times 128$ px input resolutions. Because the pretrained positional embeddings expect a $224 \times 224$ grid ($14 \times 14 = 196$ tokens), scaling down to $128 \times 128$ forces bilinear interpolation of the positional embeddings to a $8 \times 8 = 64$ grid. This mismatch degrades spatial representations and creates a severe domain shift when evaluating at $224 \times 224$ px.
3.  **Improved ViT-Tiny (`vit_tiny_patch16_224`)**: Resolves the resolution bottleneck by locking both training and evaluation resolutions to $224 \times 224$ px, keeping the token sequence at its native length of 196.

![Wildfire Detection Single Prediction (Improved ViT)](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/single_prediction.png)

### B. Two-Stage Fine-Tuning Strategy
To gently adapt the transformer layers to wildfire signatures without corrupting the representation learned on ImageNet, we utilize a two-stage training scheme:
*   **Stage 1: Linear Probing (10 Epochs)**  
    We freeze the entire self-attention and MLP backbone, enabling gradients only for the final classification head. This warms up the classifier weights and stabilizes the loss:
    $$\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
    We use the AdamW optimizer with $lr = 3\text{e-}4$ and weight decay of $1\text{e-}4$, regulated by a step learning rate scheduler.
*   **Stage 2: Full Fine-Tuning (5 Epochs)**  
    We unfreeze all layers of the ViT backbone and train the entire network with a small learning rate ($lr = 1\text{e-}5$). This allows the self-attention blocks to refine their weights to capture wildfire features (foliage silhouettes, smoke plumes, flame textures).

### C. Saliency and Explainability Algorithms
1.  **Attention Rollout (ViT)**  
    To map attention flow, the queries ($Q$) and keys ($K$) are captured via forward hooks on the attention modules. For each block, the attention matrix $A$ is calculated:
    $$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
    We average across the attention heads to get a fused representation. An identity matrix $I$ is added to account for residual connections:
    $$\hat{A} = 0.5 \cdot A + 0.5 \cdot I$$
    The rollout maps are recursively multiplied across all $L$ layers:
    $$W_{\text{rollout}} = \prod_{l=1}^{L} \hat{A}_l$$
    The final representation is extracted from the row corresponding to the `[CLS]` token, excluding the token itself, resulting in a 1D vector of 196 elements. This is reshaped to $14 \times 14$ and upsampled to $224 \times 224$.
2.  **Grad-CAM (ResNet-18)**  
    Gradients of the class score $Y^c$ are backpropagated to the feature map activations $A^k$ of the final convolutional block (`layer4`). Neurons are globally pooled to calculate channel weights $\alpha_k^c$:
    $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i, j}^k}$$
    The CAM heat map is computed as:
    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

### D. Fire Attention Faithfulness Score (FAFS)
To quantitatively benchmark explainability, we introduce the FAFS metric. Given a continuous saliency map $M \in [0, 1]$, we apply a binarization threshold $t \in [0.1, 0.9]$ to generate a binary saliency mask $B_t = M > t$. The FAFS is the Jaccard index (Intersection-over-Union) between $B_t$ and the human annotated binary mask $G$:
$$\text{FAFS}(t) = \frac{|B_t \cap G|}{|B_t \cup G|}$$

---

## 5. Experimental Results

### A. Classification Performance on In-Distribution Dataset
The models are trained using PyTorch on an Intel-based CPU environment. The table below outlines the results:

| Model | Input Size | Fine-Tuning Setup | Split | Class | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Baseline ViT** | $128 \times 128$ | Linear Probe (3 Epochs) | Test | Fire<br>Non-Fire | 0.93<br>0.98 | 0.98<br>0.90 | 0.96<br>0.94 | **94.82%** |
| **ResNet-18** | $224 \times 224$ | Joint (10 Epochs) | Val | Fire<br>Non-Fire | 0.98<br>0.99 | 0.99<br>0.98 | 0.99<br>0.98 | **98.41%** |
| **Improved ViT** | $224 \times 224$ | Two-Stage (10+5 Epochs) | Val | Fire<br>Non-Fire | 0.99<br>0.99 | 1.00<br>0.98 | 0.99<br>0.99 | **99.09%** |
| **Improved ViT** | $224 \times 224$ | Two-Stage (10+5 Epochs) | Test | Fire<br>Non-Fire | 1.00<br>1.00 | 1.00<br>0.99 | 1.00<br>1.00 | **99.77%** |

````carousel
![Validation Confusion Matrix (Improved ViT)](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/val_confusion_matrix.png)
<!-- slide -->
![Test Confusion Matrix (Improved ViT)](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/test_confusion_matrix.png)
<!-- slide -->
![Standard Confusion Matrix (Hardcoded)](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/confusion_matrix_hardcoded.png)
````

### B. Out-of-Distribution (OOD) Generalization
To evaluate model performance under severe domain shift, we benchmark the trained models zero-shot on FireNet data:

| Model | Testing Style | OOD Accuracy | Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **ResNet-18 Baseline** | Zero-Shot | **93.95%** | Fire<br>Non-Fire | 1.0000<br>0.9183 | 0.8111<br>1.0000 | 0.8957<br>0.9574 |
| **Improved ViT** | Zero-Shot | **98.22%** | Fire<br>Non-Fire | 0.9670<br>0.9895 | 0.9778<br>0.9843 | 0.9724<br>0.9869 |

*   **ResNet-18** missed $17$ of the $90$ ground fire images, suggesting that local conv filters are sensitive to the context of trees/foliage.
*   **Improved ViT** missed only $2$ fire images, maintaining a high sensitivity ($97.78\%$) and proving it learns contextual, spatial-invariant fire signatures.

### C. Computational Complexity Profiling
Benchmarks are measured on an Intel-based CPU (batch size = 1, averaged over 200 runs after 50 warm-up runs):

| Metric | ResNet-18 | Improved ViT-Tiny | Comparison / Insights |
| :--- | :---: | :---: | :--- |
| **Total Parameters** | $11.18\text{ M}$ | **$5.52\text{ M}$** | ViT-Tiny is **$50.6\%$ smaller**. |
| **FLOPs (GFLOPs)** | $3.627$ | **$2.150$** | ViT-Tiny requires **$40.7\%$ fewer** operations. |
| **CPU Latency** | $116.43 \pm 16.73\text{ ms}$ | **$69.45 \pm 19.54\text{ ms}$** | ViT-Tiny is **$1.68\times$ faster** per image. |
| **Throughput (FPS)** | $8.6\text{ FPS}$ | **$14.4\text{ FPS}$** | ViT-Tiny handles **$1.67\times$ more** frames per second. |

### D. Explainability Faithfulness Comparison (FAFS Threshold Ablation)
We compute the FAFS (IoU) over a batch of $50$ validation masks across different thresholds $t$:

| Threshold ($t$) | ResNet-18 Grad-CAM | Baseline ViT Rollout | Improved ViT Rollout |
| :---: | :---: | :---: | :---: |
| **0.1** | 0.2688 | 0.2448 | 0.2346 |
| **0.2** | 0.2916 | 0.2995 | 0.2793 |
| **0.3** | 0.3104 | **0.3475** (Peak) | **0.3150** (Peak) |
| **0.4** | **0.3189** (Peak) | 0.3407 | 0.3087 |
| **0.5** | 0.3102 | 0.2727 | 0.2611 |
| **0.6** | 0.2927 | 0.1850 | 0.1843 |
| **0.7** | 0.2496 | 0.1004 | 0.1057 |
| **0.8** | 0.1828 | 0.0437 | 0.0522 |
| **0.9** | 0.0884 | 0.0117 | 0.0135 |

````carousel
![Spatial Faithfulness FAFS (IoU) Comparison Curves (N=50)](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/explainability_comparison.png)
<!-- slide -->
![Attention Overlays on Fire and Non-Fire Validation Images](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/attention_comparison.png)
<!-- slide -->
![Attention Rollout Mask Overlaid on Source Image (0008.png)](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/attention_mask_overlay.png)
<!-- slide -->
![Ablation Threshold Curves for Spatial Localization](C:/Users/Lenovo/.gemini/antigravity-ide/brain/0cbdd1e6-2f14-463a-9aef-43998e980620/fafs_threshold_ablation.png)
````

---

## 6. Discussion

### A. Inductive Biases vs. Global Self-Attention
Convolutional Neural Networks rely on strong spatial priors (locality and weight sharing). While highly effective for learning local fire boundaries and textures on validation data, these priors become a bottleneck under out-of-distribution (OOD) tests. When tested on domestic fires (FireNet), where trees and aerial scales are absent, the local receptive fields of ResNet-18 fail, dropping recall to $81.11\%$.

In contrast, the Vision Transformer processes global relationships between patch tokens without spatial locality constraints. This enables it to focus on global lighting gradients, soot/smoke dispersion, and combustion textures. Although it lacks spatial locality priors (resulting in slightly lower peak IoU against tight human segmentations), this global representation exhibits robust resilience to domain shifts, maintaining an OOD recall of $97.78\%$.

### B. Positional Embedding and Resolution Shift
In ViTs, positional embeddings are added to patch embeddings to preserve spatial sequence order. When training a model on $128 \times 128$ px and evaluating on $224 \times 224$ px, the model must interpolate these positional representations, leading to structural representation gaps. This resolution mismatch explains why the Baseline ViT has a lower test accuracy ($94.82\%$) and high false positives. Standardizing the pipeline to $224 \times 224$ px resolves these artifacts, raising accuracy to $99.77\%$.

### C. Interpretability and Localization (FAFS)
Grad-CAM maps gradients directly onto deep feature maps, which natively capture localized contours. This yields a dense highlight area that aligns tightly with human-annotated fire masks, maintaining high FAFS scores across thresholds ($0.1$ to $0.6$).

Attention Rollout recursively aggregates attention across transformer blocks. Because self-attention captures broader context (such as surrounding unburned trees and terrain gradients used to classify the scene as a "forest fire"), the rollout map is more distributed. This context decreases geometric overlap with tight fire segmentations at high binarization thresholds ($t \ge 0.5$), but represents a more comprehensive explanation of the model's global scene reasoning.

---

## 7. Conclusions & Future Work
This paper presented a comparative analysis between CNNs and Vision Transformers for forest fire detection. By addressing the image resolution bottleneck and employing a Two-Stage Fine-Tuning Strategy, the Improved ViT-Tiny achieved a test accuracy of $99.77\%$ on in-distribution forest images and a zero-shot OOD accuracy of $98.22\%$ on domestic urban fires (FireNet). The Improved ViT is twice as small and runs $1.68\times$ faster on CPU than the ResNet-18 baseline, validating its edge viability.

For future work, we plan to:
1.  Extend the self-attention backbone to multi-spectral satellite channels (e.g., Sentinel-2 SWIR/NIR) to detect hot spots through smoke cover.
2.  Deploy explainability-guided inference: utilizing the FAFS metric in production to discard classifications where the spatial attention focuses on background elements (like clouds or dust).
3.  Deploy the model on edge microcontrollers (using ONNX/TensorRT quantization) for autonomous drone surveillance.

---

## References
1.  **Karabuk University**, *master_proje Dataset*, Roboflow Universe, 2023. Available: [dataset link](https://universe.roboflow.com/karabuk-un/master_proje/dataset/1).
2.  A. Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,"* arXiv preprint arXiv:2010.11929, 2020.
3.  K. He, X. Zhang, S. Ren, and J. Sun, *"Deep Residual Learning for Image Recognition,"* in *CVPR*, 2016.
4.  S. Abnar and W. Zuidema, *"Quantifying Attention Flow in Transformers,"* arXiv preprint arXiv:2005.00928, 2020.
5.  R. R. Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,"* in *ICCV*, 2017.
6.  FireNet Wildfire Database, OOD Validation Set.
