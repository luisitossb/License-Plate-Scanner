# License Plate Scanner

Automatic license plate detection and recognition using deep learning.  
**CSC671 — Deep Learning · SFSU Spring 2026**  
**Team:** David Lei, Luis Gabriel Ortiz-Anguiano, Ryan Kurian George

---

## Overview

This project builds a two-stage pipeline for automatic license plate recognition (ALPR):

1. **Detection** — A fine-tuned ResNet CNN predicts a bounding box around the license plate in any input image.
2. **OCR** — The detected region is cropped and passed to Tesseract OCR to read the plate text.

The model is trained and evaluated on 10,637 labeled license plate images sourced from Kaggle.

---

## Pipeline

```
Input Image
    │
    ▼
ResNet Detector  ──►  Bounding Box [xmin, ymin, xmax, ymax]
    │
    ▼
Crop Plate Region
    │
    ▼
Tesseract OCR  ──►  Plate Text (e.g. "ABC1234")
```

---

## Dataset

- **Source:** [License Plate Recognition](https://www.kaggle.com/datasets/adilshamim8/license-plate-recognition) on Kaggle (CC BY 4.0)
- **Size:** 10,637 images with bounding box annotations
- **Split:** 7,357 train / 2,195 validation / 1,085 test
- **Format:** Each split has a `_annotations.csv` with columns `filename`, `width`, `height`, `xmin`, `ymin`, `xmax`, `ymax`

The dataset is **not included in this repo** (~500MB). It is downloaded automatically by the notebook via the Kaggle CLI (see Setup below).

---

## Model Architecture

We use a **pretrained ResNet** (ImageNet weights) as a feature extractor, with the original classification head replaced by a lightweight regression head:

```
ResNet Backbone (resnet18 or resnet50)
    └── Global Average Pool  →  feature vector (512d or 2048d)
         └── Linear(feat_dim, 256)
              └── ReLU + Dropout(0.3)
                   └── Linear(256, 4)
                        └── Sigmoid  →  [xmin, ymin, xmax, ymax] ∈ [0, 1]
```

Bounding box coordinates are normalized to `[0, 1]` relative to image dimensions, so the model output is resolution-independent.

**Loss function:** Smooth L1 (Huber loss) — robust to outlier boxes while still penalizing large errors.  
**Evaluation metric:** Mean IoU (Intersection over Union) — measures how well the predicted box overlaps the ground truth (0 = no overlap, 1 = perfect).

---

## Hyperparameter Search

A grid search is run over two dimensions before full training:

| Parameter | Values searched |
|---|---|
| Learning rate | `1e-3`, `5e-4`, `1e-4` |
| Backbone | `resnet18`, `resnet50` |

Each of the 6 combinations trains for 5 epochs. The combination with the lowest final validation loss is used for full training.

---

## Notebook Walkthrough

Everything runs in a single notebook: `notebooks/license_plate_detection.ipynb`

| Section | Description |
|---|---|
| 0 — Setup | Install Kaggle CLI, download and extract the dataset |
| 1 — EDA | Sample images with ground-truth boxes, bounding box size distributions |
| 2 — DataLoader | `LicensePlateDataset` class — resizes images to 224×224, normalizes boxes to `[0,1]` |
| 3 — Model | `LicensePlateDetector` class, `compute_iou` function, forward pass sanity check |
| 4 — Training Utilities | `train_one_epoch` and `evaluate` functions |
| 5 — Hyperparameter Search | 6-config grid search (LR × backbone), validation loss + IoU plots |
| 6 — Full Training | 30-epoch training with best config, `ReduceLROnPlateau` scheduler, saves best checkpoint |
| 7 — Evaluation | Loads best checkpoint, reports test loss and mean IoU |
| 8 — Visualization | Side-by-side ground truth (green) vs prediction (red) boxes on test images |
| 9 — OCR | Tesseract setup, plate crop preprocessing, `read_plate_text` function |
| 10 — Full Pipeline | `detect_and_read()` — takes any image path, returns bounding box + plate text |

---

## Project Structure

```
License-Plate-Scanner/
├── notebooks/
│   └── license_plate_detection.ipynb   # entire project — run this
├── src/
│   ├── dataset.py                      # LicensePlateDataset class
│   ├── model.py                        # LicensePlateDetector + compute_iou
│   └── train.py                        # train_one_epoch + evaluate
├── data/
│   ├── raw/                            # downloaded dataset (gitignored)
│   └── processed/                      # reserved for future preprocessing
├── models/                             # saved model checkpoints (gitignored)
├── outputs/                            # example outputs (gitignored)
├── instructions/
│   ├── Project Instructions.pdf
│   └── Project Proposal.pdf
├── requirements.txt
└── README.md
```

> `src/` contains the same classes and functions as the notebook, extracted into importable modules. The notebook is self-contained and does not depend on `src/`.

---

## Setup & Running Guide

### On Google Colab (recommended — free GPU)

1. Open `notebooks/license_plate_detection.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Add a cell at the top to set up Kaggle credentials:
   ```python
   from google.colab import files
   files.upload()  # upload your kaggle.json
   !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   Download `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
4. Run all cells: Runtime → Run all
5. Total time: ~60–90 minutes (data download ~5 min, hyperparameter search ~30 min, full training ~30 min)

### Locally

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR (for Section 9)
# macOS:
brew install tesseract
# Ubuntu/Debian:
sudo apt install tesseract-ocr

# Place kaggle.json at ~/.kaggle/kaggle.json, then launch Jupyter
jupyter notebook notebooks/license_plate_detection.ipynb
```

---

## Requirements

```
torch
torchvision
opencv-python
numpy
pandas
Pillow
matplotlib
scikit-learn
jupyter
kaggle
pytesseract
```

Install all with: `pip install -r requirements.txt`

---

## Results

After training, the notebook produces:
- Hyperparameter search plots comparing all 6 backbone × LR configurations
- Loss and IoU curves over 30 epochs
- Test set mean IoU score
- Visual examples of predicted vs ground-truth bounding boxes
- OCR output on cropped plate regions
- End-to-end demo images with detection box and plate text overlay
