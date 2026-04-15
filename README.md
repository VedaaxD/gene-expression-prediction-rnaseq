# gene-expression-prediction-rnaseq
Deep learning pipeline for predicting gene expression patterns from RNA-seq data with evaluation and interpretability analysis.
# Gene Expression Prediction using RNA-Seq (GTEx)

## Overview
This project focuses on predicting genome-wide gene expression levels from a subset of landmark genes using machine learning and deep learning models.

Using GTEx RNA-seq data, we model the relationship between a small set of measured genes (landmark genes) and thousands of target genes — a problem inspired by the LINCS L1000 assay.

---

## Problem Statement
Given expression values of ~978 landmark genes, predict the expression of ~2000 target genes across different tissue samples.

This is a high-dimensional regression problem with biological significance in:
- Drug response prediction
- Functional genomics
- Transcriptomic profiling

---

## Pipeline

### 1. Data Preprocessing
Implemented in data_prep.py :contentReference[oaicite:0]{index=0}

- Downloaded GTEx v8 median TPM dataset
- Log transformation: log2(TPM + 1)
- Filtered low-expression genes
- Randomly split into:
  - Landmark genes (inputs)
  - Target genes (outputs)
- Data augmentation using Gaussian noise (due to limited samples)
- Standardization using StandardScaler
- Train / Validation / Test split

---

### 2. MLP Model
Implemented in train_mlp.py :contentReference[oaicite:1]{index=1}

**Architecture:**
Input → Linear → BatchNorm → ReLU → Dropout (×3) → Output


**Key Features:**
- Deep fully connected network
- Regularization via BatchNorm + Dropout
- Early stopping
- Cosine learning rate scheduler

**Evaluation:**
- Pearson correlation (per gene)
- R² score
- MSE / MAE
- Gene-level performance ranking

---

### 3. Transformer Model + Interpretability
Implemented in train_transformer.py :contentReference[oaicite:2]{index=2}

**Key Idea:**
Treat each gene as a "token" and learn gene–gene interactions using self-attention.

**Architecture:**
Gene Embedding → Transformer Encoder → Mean Pool → MLP Head → Output


**Highlights:**
- Multi-head self-attention across genes
- Positional embeddings for gene identity
- Captures gene-gene dependencies

---

## Interpretability (Captum)
- Used **Integrated Gradients** (Captum)
- Identified most influential landmark genes
- Generated:
  - Feature importance plots
  - Attribution heatmaps

---

## Results

### Transformer Performance
- **Pearson (mean): 0.987**
- **R² Score: 0.9967**
- **MSE: 0.0149**
- **MAE: 0.0840**

Transformer significantly outperforms baseline MLP.

---

## Project Structure
.
├── data/
│ ├── landmark_genes.txt
│ ├── target_genes.txt
│ └── scaler.pkl
│
├── outputs/
│ ├── mlp/
│ └── transformer/
│
├── data_prep.py
├── train_mlp.py
├── train_transformer.py
├── requirements.txt
└── README.md


---

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Prepare data
python data_prep.py

### 3.Train MLP
python train_mlp.py

### 4.Train Transformer + Interpretability
python train_transformer.py


## Tech Stack
Python
PyTorch
NumPy / Pandas
Scikit-learn
Matplotlib
Captum (for interpretability)

## Key Contributions
Built end-to-end ML pipeline for gene expression prediction
Compared MLP vs Transformer architectures
Applied self-attention to biological sequence-like data
Integrated model interpretability using Captum
Achieved high predictive performance (R² ~ 0.997)
