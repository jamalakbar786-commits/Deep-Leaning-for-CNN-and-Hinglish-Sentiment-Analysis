# Solar Panel Defect Detection Using Convolutional Neural Networks (CNN) on Electroluminescence Images

## Overview

This project focuses on the automated detection of defects in solar cells using Electroluminescence (EL) images and Deep Learning techniques. The goal is to classify solar cells as **Functional** or **Defective** and assist industrial quality inspection by reducing dependence on manual visual inspection.

The project utilizes the **ELPV (Electroluminescence Photovoltaic) Dataset** and implements image preprocessing, data augmentation, custom CNN architectures, transfer learning models, and comprehensive model evaluation.

---

## Problem Statement

Solar panel manufacturers use Electroluminescence (EL) imaging to identify micro-cracks and other defects that can reduce power output and shorten panel lifespan.

Manual inspection:

- Is time-consuming
- Requires expert knowledge
- Is difficult to scale in manufacturing environments

This project develops an automated computer vision pipeline capable of detecting defective solar cells from EL images.

---

## Business Applications

- Automated quality control in solar manufacturing plants
- Early detection of defective cells before module assembly
- Reduction of warranty and maintenance costs
- Improved production efficiency
- Predictive maintenance for deployed solar farms
- Decision support for quality inspectors

---

## Dataset

### ELPV Dataset

- 2,624 grayscale EL images
- Image size: 300 × 300 pixels
- Monocrystalline and Polycrystalline solar cells
- Defect probability labels: 0, 0.33, 0.66, 1.0

Dataset Repository:
https://github.com/zae-bayern/elpv-dataset

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Future Improvements

- Grad-CAM defect localization
- Vision Transformers (ViT)
- Ensemble Learning
- Real-time deployment

---

## Author

Jamal Akbar


