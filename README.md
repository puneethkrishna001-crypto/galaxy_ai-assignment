# EO-SAR Change Detection using Siamese U-Net

## Overview

This project performs binary change detection using multimodal EO (Electro-Optical) and SAR (Synthetic Aperture Radar) satellite imagery.

The objective is to identify structural and environmental changes between pre-event EO imagery and post-event SAR imagery using deep learning-based semantic segmentation.

This project was developed as part of the AI Research Internship technical assessment for GalaxEye Space.

---

# Problem Statement

Given:
- Pre-event EO satellite image
- Post-event SAR satellite image

The model predicts:
- Binary change mask indicating damaged or changed regions.

This is a challenging multimodal remote sensing problem because EO and SAR imagery have very different visual characteristics.

---

# Key Features

- EO-SAR multimodal fusion
- Siamese U-Net architecture
- Shared ResNet18 encoder
- Binary semantic segmentation
- BCE + Dice hybrid loss
- IoU / Precision / Recall / F1 evaluation
- Prediction visualization pipeline
- Invalid-region masking for no-data SAR regions
- Modular and reproducible pipeline

---

# Dataset Structure

```
data/
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

### Data Modalities

| Input | Description |
|---|---|
| Pre-event | RGB EO satellite image |
| Post-event | Grayscale SAR image |
| Target | Binary change mask |

---

# Model Architecture

The project uses a Siamese U-Net architecture with a shared ResNet18 encoder.

## Pipeline

```
Pre-event EO Image              Post-event SAR Image
        │                               │
        └────────────┬──────────────────┘
                     │
              Shared Encoder
                     │
         ┌───────────┴───────────┐
         │                       │
    Feature Map 1          Feature Map 2
         │                       │
         └───────────┬───────────┘
                     │
            Feature Difference
                     │
              U-Net Decoder
                     │
           Binary Change Mask
```

### Why Siamese U-Net?

- Shared encoders learn comparable semantic representations
- Feature differencing highlights temporal changes
- U-Net decoder preserves spatial localization
- Lightweight and efficient for segmentation tasks

---

# Training Details

| Parameter | Value |
|---|---|
| Image Size | 256 × 256 |
| Encoder | ResNet18 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | BCE + Dice Loss |
| Batch Size | 2 |
| Epochs | 2 |

---

# Loss Function

The project uses a hybrid loss:

## BCE Loss
Handles pixel-wise binary classification.

## Dice Loss
Improves overlap learning for sparse change regions caused by severe class imbalance.

### Motivation

The dataset contains:
- Large no-change regions
- Sparse damaged regions

Dice Loss helps improve segmentation quality under imbalance.

---

# Major Challenges

## 1. EO-SAR Modality Gap

EO and SAR images have very different visual properties:
- EO captures optical appearance
- SAR captures radar backscatter and texture

This causes:
- feature inconsistency
- texture ambiguity
- false positives

---

## 2. Severe Class Imbalance

Most pixels belong to the background/no-change class.

Approximate distribution:
- ~96% no-change
- ~4% change

---

## 3. Invalid No-Data Regions

Several samples contained large triangular black regions caused by:
- sensor coverage mismatch
- geometric inconsistencies
- missing SAR data

The model initially overpredicted changes in these areas.

### Solution

A valid-region masking strategy was introduced during inference to suppress false positives in no-data regions.

---

# Evaluation Metrics

The model was evaluated using:

- IoU (Intersection over Union)
- Precision
- Recall
- F1 Score

---

# Final Test Results

| Metric | Score |
|---|---|
| IoU Score | 0.2092 |
| Precision | 0.2907 |
| Recall | 0.4275 |
| F1 Score | 0.3460 |

---

# Qualitative Results

## Sample Predictions

The repository includes:
- Ground truth masks
- Predicted masks
- EO inputs
- SAR inputs

All outputs are stored in:

```
outputs/predictions/
```

---

# Project Structure

```
galaxy_ai-assignment/
├── datasets/
│   ├── __init__.py
│   └── change_dataset.py
├── models/
│   └── siamese_unet.py
├── utils/
│   └── losses.py
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── results/
├── notebooks/
├── train.py
├── eval.py
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone <your-repo-link>
cd galaxy_ai-assignment
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Training

```bash
python train.py
```

## Evaluation

```bash
python eval.py
```

## Inference

```bash
python inference.py
```

Generated predictions are saved in:

```
outputs/predictions/
```

---

# Future Improvements

Potential future enhancements include:

- Attention-based feature fusion
- Transformer-based encoders
- Multi-scale segmentation
- Advanced augmentation pipeline
- Better SAR denoising
- Domain adaptation techniques
- Improved no-data masking

---

# Key Learnings

This project provided hands-on experience in:

- Remote sensing AI
- EO-SAR multimodal fusion
- Semantic segmentation
- Change detection
- Dataset imbalance handling
- Failure analysis
- Model evaluation
- Deep learning pipeline engineering

---

# Author

**Puneeth Krishna**

B.Tech Civil Engineering
NITK Surathkal
