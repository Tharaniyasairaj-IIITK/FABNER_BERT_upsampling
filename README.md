# FABNER_BERT_upsampling

# Evaluating Oversampling Techniques on Transformer Embeddings for Imbalanced Multi-Class NER

This repository analyses how different oversampling strategies (**SMOTE**, **ADASYN**, **SMOTE-ENN**, **SMOTE-Tomek**) affect the minority classes of a highly imbalanced, 768-dimensional embedding dataset. It combines class-distribution analysis, PCA visualisation, and nearest-neighbour distance analysis to compare the quality and realism of the synthetic samples each method generates.

> Notebook: `Project Phase 2` (Google Colab)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Methodology](#methodology)
- [Results](#results)
- [Key Observations](#key-observations)
- [Future Work](#future-work)

---

## Overview

Token-level classification tasks often suffer from severe class imbalance. In this project, the training set contains 195,072 embedding vectors (768-D) across many classes, where some classes have as few as **1 sample**. Four resampling techniques were applied, and this notebook inspects what they actually do to the embedding space.

**Goals**

1. Compare class counts before and after resampling.
2. Visualise the original vs. synthetic samples of a minority class (S-BIOP, class 37) using a shared PCA projection.
3. Quantify how far synthetic samples lie from real ones using nearest-neighbour distances in the original 768-D space.

---

## Dataset

Pre-computed embeddings stored as `.npy` files:

| Dataset      | Samples (X shape) | Files                                              |
|--------------|-------------------|----------------------------------------------------|
| Original     | 195,072 × 768     | `X_train_embeddings.npy`, `y_train.npy`            |
| SMOTE        | 202,711 × 768     | `X_train_smote.npy`, `y_train_smote.npy`           |
| ADASYN       | 202,623 × 768     | `X_train_adasyn.npy`, `y_train_adasyn.npy`         |
| SMOTE-ENN    | 192,315 × 768     | `X_train_smote_enn.npy`, `y_train_smote_enn.npy`   |
| SMOTE-Tomek  | 201,625 × 768     | `X_train_smote_tomek.npy`, `y_train_smote_tomek.npy` |

**Rarest classes (original training set)**

| Class | Samples | | Class | Samples |
|-------|---------|-|-------|---------|
| 31    | 1       | | 27    | 35      |
| 8     | 4       | | 29    | 51      |
| 20    | 4       | | 37    | 57      |
| 1     | 14      | | 33    | 71      |
| 13    | 14      | | 34    | 75      |
| 28    | 17      | | 0     | 120     |
| 24    | 21      | | 12    | 120     |
| 44    | 21      | |       |         |

**Selected minority classes analysed**

| Class ID | Label    |
|----------|----------|
| 37       | S-BIOP   |
| 29       | I-MACEQ  |
| 33       | I-PARA   |
| 34       | I-PRO    |
| 0        | B-APPL   |

---

## Project Structure

```
.
├── README.md
├── notebook.ipynb            # Main analysis notebook
└── Embeddings/               # (not tracked) .npy files stored on Google Drive
    ├── X_train_embeddings.npy
    ├── y_train.npy
    ├── X_train_smote.npy / y_train_smote.npy
    ├── X_train_adasyn.npy / y_train_adasyn.npy
    ├── X_train_smote_enn.npy / y_train_smote_enn.npy
    └── X_train_smote_tomek.npy / y_train_smote_tomek.npy
```

---

## Requirements

- Python 3.8+
- numpy
- scikit-learn
- matplotlib
- Google Colab (for Drive mounting) — or adapt `base_path` for local use

```bash
pip install numpy scikit-learn matplotlib
```

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Place the `.npy` embedding files in a folder and update `base_path` in the notebook:
   ```python
   base_path = "/content/drive/MyDrive/PROJECT PHASE2/Embeddings"
   ```
3. Open the notebook in Jupyter or Google Colab and run all cells.

---

## Methodology

### 1. Class distribution analysis
Counts per class are computed with `collections.Counter` to identify the rarest classes, then compared across all four resampled datasets for the selected minority classes.

### 2. PCA visualisation (S-BIOP, class 37)
All S-BIOP embeddings from the five datasets are stacked and projected with **one common PCA** (`n_components=2`, `random_state=42`) so the plots are directly comparable.

- PC1 explains ~35.4% of variance, PC2 ~16.2% (**~51.6% total**).
- Plots: all methods together, plus Original vs. each method individually.

### 3. Nearest-neighbour distance analysis
For every resampled S-BIOP point, the Euclidean distance to its nearest **original** S-BIOP sample is computed in the full 768-D space (`sklearn.neighbors.NearestNeighbors`).

- First pass includes points identical to originals (distance ≈ 0).
- Corrected pass isolates **synthetic-only** samples using a numerical tolerance (`TOLERANCE = 1e-5`), since floating-point error produced tiny non-zero distances (~1e-7) for copies of original points, which made a strict `> 1e-8` filter unreliable.

---

## Results

### Class counts after resampling (selected classes)

| Class | Original | SMOTE | ADASYN | SMOTE-ENN | SMOTE-Tomek |
|-------|----------|-------|--------|-----------|-------------|
| 37    | 57       | 500   | 505    | 498       | 500         |
| 29    | 51       | 500   | 496    | 495       | 500         |
| 33    | 71       | 500   | 500    | 498       | 500         |
| 34    | 75       | 500   | 498    | 489       | 500         |
| 0     | 120      | 500   | 496    | 485       | 498         |

### Nearest-neighbour distance to original S-BIOP samples

| Method      | Mean   | Median | Max     |
|-------------|--------|--------|---------|
| SMOTE       | 1.9303 | 1.5273 | 10.6429 |
| ADASYN      | 2.9345 | 2.4338 | 12.0627 |
| SMOTE-ENN   | 2.0625 | 1.6884 | 9.9368  |
| SMOTE-Tomek | 2.0601 | 1.6884 | 9.9368  |

*(The tolerance-based synthetic-only analysis was in progress at the end of the notebook.)*

---

## Key Observations

- All methods bring minority classes up to roughly **500 samples**; the cleaning variants (ENN/Tomek) end up slightly under that after removing noisy points.
- **SMOTE** synthetics stay close to original samples (lowest mean/median distance), interpolating mostly within existing clusters.
- **ADASYN** produces the most spread-out samples (highest mean/max distance), visibly extending along "bridges" between clusters in the PCA plot — it focuses on harder regions, which can also introduce more noise.
- **SMOTE-ENN / SMOTE-Tomek** behave similarly to SMOTE but remove ambiguous samples; their distance statistics are nearly identical to each other.
- A strict `distance > 1e-8` test misidentifies original-sample copies as synthetic because of float32 precision, hence the tolerance-based correction.
- 2D PCA captures only ~52% of the variance, so overlap seen in plots should be interpreted with caution; the 768-D distances are more reliable.

---

## Future Work

- Handle ultra-rare classes (e.g., class 31 with 1 sample) with alternative strategies such as data augmentation or class-weighted losses, since SMOTE needs multiple neighbours.
- Try t-SNE / UMAP for non-linear visualisation.
- Evaluate downstream classifier performance (per-class precision/recall/F1) on each resampled dataset.


---
