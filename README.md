# Real-time Domain Adaptation in Semantic Segmentation

## Project Overview

This project tackles the challenge of **domain adaptation** in **semantic segmentation**, focusing on enabling real-time models to maintain high accuracy across varying visual domains. Semantic segmentation involves assigning semantic labels to every pixel in an image, which is crucial in applications like autonomous driving, remote sensing, and medical imaging.

A persistent challenge is **domain shift**, where models trained on one domain (source) perform poorly on another (target) due to differences in data distributions, textures, and scene layouts. This project explores methods to alleviate the effects of domain shift on real-time segmentation systems.

## Dataset

We use the **LoveDA** dataset, a high spatial resolution dataset comprising satellite images from two distinct domains: urban and rural. The dataset's domain diversity makes it an ideal benchmark for studying domain adaptation in land-cover semantic segmentation.

## Models and Architectures

- **DeepLabV2**: A classical semantic segmentation network employing atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context, serving as a high-accuracy baseline.
- **PIDNet**: A modern real-time semantic segmentation network inspired by Proportional-Integral-Derivative (PID) controllers, designed to balance accuracy with inference speed, used as the primary real-time baseline.
- **STDC (Short-Term Dense Concatenate)**: A lightweight backbone architecture optimized for fast and efficient feature extraction.
- **PEM (Prototype-based Efficient MaskFormer)**: An advanced architecture leveraging prototype learning and efficient multi-scale context aggregation for improved segmentation accuracy.

## Domain Adaptation Techniques

- **Adversarial Learning**: Features domain-invariant representation learning by jointly training the segmentation model with a discriminator network that distinguishes source from target domain features, encouraging better alignment between domains.
- **Domain Adaptive Cross-Entropy Sampling (DACS)**: A data-centric approach that mixes source and target domain image patches during training with pseudo-label supervision, improving model robustness without requiring additional discriminators.

## Methodology

The project proceeds through several key stages:

1. **Baseline Training and Evaluation**  
   Training and assessing DeepLabV2 and PIDNet models on the source (urban) domain, then evaluating on the target (rural) domain to characterize the domain shift impact.

2. **Data Augmentation**  
   Applying augmentations such as horizontal flip, Gaussian blur, and color jitter on the source domain to enhance generalization.

3. **Domain Adaptation**  
   Implementing adversarial learning and DACS to mitigate domain shift and improve segmentation on the target domain.

4. **Model Extensions**  
   Investigating STDC and PEM architectures to analyze trade-offs in efficiency and accuracy under domain shift.

## Evaluation Metrics

Performance is measured primarily by mean Intersection-over-Union (mIoU) across classes, along with per-class IoU and pixel accuracy. Computational metrics such as FLOPs, parameter count, and inference latency evaluate efficiency, essential for real-time applicability.

## Key Findings

- Significant degradation in segmentation performance occurs when models trained on urban images are applied directly to rural images without adaptation.
- Both adversarial learning and DACS show effective domain shift mitigation, with DACS offering greater training stability and consistent improvements for rare classes.
- PIDNet provides a strong trade-off between speed and accuracy for real-time segmentation compared to classical DeepLabV2.
- Extensions with STDC and PEM highlight how advanced architectural components can further enhance domain adaptation outcomes while maintaining efficiency.

## Future Directions

The project encourages further exploration in:

- Self-supervised and test-time adaptation methods to improve model robustness dynamically.
- Hybrids of adversarial and data mixing approaches to leverage complementary strengths.
- Broader applicability across diverse real-world domain adaptation scenarios.

## References

The work builds upon prominent advances in semantic segmentation, domain adaptation methods, and real-time architectures, referencing foundational research nd recent developments in the field.

---
