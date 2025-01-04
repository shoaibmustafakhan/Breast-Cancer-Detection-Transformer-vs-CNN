# Comparative Analysis of Swin Transformer and ResNet Models for Breast Cancer Detection

This repository contains the code and research findings for a comparative study between two deep learning architectures, Swin Transformer and ResNet, for breast cancer detection using mammography images.

Important Note: NONE OF THESE MODELS ARE SUBSTITUTE FOR PRACTICAL MEDICAL ADVISORY. PLEASE CONSULT A LICENSED MEDICAL PRACTITIONER FOR CONSULTATION.

## Abstract
This ongoing experimental study evaluates the performance of Swin Transformer and ResNet architectures on the CBIS-DDSM dataset for breast cancer detection. The models were compared based on classification metrics, including accuracy, precision, recall, and F1-score. The Swin Transformer showed high sensitivity for benign cases, while ResNet demonstrated more balanced performance across benign and malignant classifications.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
    - [Dataset](#dataset)
    - [Model Architectures](#model-architectures)
        - [Swin Transformer](#swin-transformer)
        - [ResNet](#resnet)
    - [Training Protocol](#training-protocol)
3. [Results and Analysis](#results-and-analysis)
4. [Discussion](#discussion)
5. [Limitations and Future Work](#limitations-and-future-work)
6. [Conclusion](#conclusion)

---

## Introduction
Early detection of breast cancer through mammography is critical for improving outcomes. Advances in artificial intelligence, particularly deep learning, offer promising solutions for automating detection. This study compares Swin Transformer, a state-of-the-art vision transformer, and ResNet, a well-established convolutional neural network (CNN), for mammography-based breast cancer detection.

---

## Methodology

### Dataset
The study used the **CBIS-DDSM dataset**, which includes:
- Categories: Calcifications and masses
- Data: Full mammogram images and ROI crops
- Labels: Benign and malignant
- Metadata: Patient ID, breast density, and image view (CC/MLO)

**Data Preprocessing**:
- Augmentations: Random flips, rotations (±10°), and affine transformations
- Standardization: ImageNet mean and std values
- Split: 90% training and 10% validation

### Model Architectures

#### Swin Transformer
Key features:
- Hierarchical feature representation with window-based attention
- Input resolution: 384×384 pixels
- Dropout: 0.1 for hidden layers and attention probabilities
- Architecture: 4 stages with [2, 2, 18, 2] layers

#### ResNet
Key features:
- Residual learning framework with skip connections
- Input resolution: 224×224 pixels
- Architecture: 50 layers with bottleneck blocks
- Global average pooling for dimensionality reduction

### Training Protocol
#### Swin Transformer:
- Optimizer: AdamW
- Learning rate: 2e-5
- Loss: Weighted CrossEntropyLoss
- Scheduler: Cosine annealing

#### ResNet:
- Optimizer: Adam
- Learning rate: 1e-4
- Loss: CrossEntropyLoss
- Batch size: 16

---

## Results and Analysis

### Performance Metrics
| Metric       | Swin Transformer | ResNet |
|--------------|------------------|--------|
| Accuracy     | 64.35%           | 63.64% |
| Recall (Benign) | 89%          | 70%    |
| Recall (Malignant) | 25%       | 51%    |
| Precision (Benign) | 65%       | 69%    |
| Precision (Malignant) | 61%    | 53%    |

### Confusion Matrices
**Swin Transformer**:
|               | Predicted Benign | Predicted Malignant |
|---------------|------------------|---------------------|
| Actual Benign | 383              | 45                  |
| Actual Malignant | 206           | 70                  |

**ResNet**:
|               | Predicted Benign | Predicted Malignant |
|---------------|------------------|---------------------|
| Actual Benign | 301              | 127                 |
| Actual Malignant | 134           | 142                 |

---

## Discussion
### Model Characteristics
**Swin Transformer**:
- High specificity for benign cases
- Conservative in malignant classification

**ResNet**:
- Balanced classification for benign and malignant cases
- Suitable for general diagnostic support

### Clinical Implications
- Swin Transformer may reduce unnecessary biopsies.
- ResNet offers balanced diagnostic utility.
- Neither model is yet suitable for clinical-grade applications.

---

## Limitations and Future Work

### Current Limitations
- Moderate accuracy levels
- Limited dataset size
- Computational resource constraints

### Future Directions
- Pre-training on larger datasets
- Ensemble learning approaches
- Integration of clinical metadata

---

## Conclusion
This study highlights the comparative strengths and weaknesses of Swin Transformer and ResNet models for breast cancer detection. While both models achieve comparable accuracy, they exhibit distinct classification behaviors, making them suitable for different clinical scenarios. Further research is needed to achieve clinical-grade performance.

---

## How to Use This Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/shoaibmustafakhan/Breast-Cancer-Detection-Transformer-vs-CNN.git

