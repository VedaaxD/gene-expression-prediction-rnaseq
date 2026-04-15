"""
Download GTEx v8 gene expression data and prepares tensors for model training.

What this does:
    1. Downloads GTEx median TPM matrix (tissues × genes)
    2. Log-normalises: log2(TPM + 1)
    3. Splits into landmark genes (inputs) and target genes (outputs)
    4. Train/val/test split
    5. Saves tensors to data/ folder

Run:
    python data_prep.py
"""

import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import os


# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("/home/ibab/gene_exp_prediction/data")
DATA_DIR.mkdir(exist_ok=True)

# GTEx v8 median TPM per tissue (54 tissues × ~57k genes) — ~60 MB
GTEX_URL  = (
    "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)
GTEX_FILE = DATA_DIR / "gtex_median_tpm.gct.gz"

N_LANDMARK  = 978       # number of input (landmark) genes — mimics L1000 assay
N_TARGETS   = 2000      # number of target genes to predict
MIN_TPM     = 1.0       # filter out genes with median TPM < 1 across all tissues
RANDOM_SEED = 42

# STEP 1: Download
# def download_gtex():
#     if GTEX_FILE.exists():
#         print(f"[✓] Already downloaded: {GTEX_FILE}")
#         return
#     print(f"Downloading GTEx data (~60 MB)...")
#     urllib.request.urlretrieve(GTEX_URL, GTEX_FILE)
#     print(f" Saved to {GTEX_FILE}")


# STEP 2: Parse GCT format
def load_gtex(filepath) -> pd.DataFrame:
    """
    GTEx .gct format:
        Row 1: version
        Row 2: dimensions
        Row 3+: data (gene_id, gene_name, tissue1, tissue2, ...)
    Returns DataFrame: genes × tissues (TPM values)
    """
    print("Parsing GTEx GCT file...")
    with gzip.open(filepath, "rt") as f:
        # Skip first 2 header lines
        _ = f.readline()
        _ = f.readline()
        df = pd.read_csv(f, sep="\t", index_col=1)  # gene_name as index

    # Drop gene_id column, keep only tissue columns
    df = df.drop(columns=["Name"], errors="ignore")
    df.index.name = "gene"

    print(f"  Loaded: {df.shape[0]} genes × {df.shape[1]} tissues")
    return df   # genes × tissues


# STEP 3: Filter + normalise
def preprocess(df: pd.DataFrame):
    print(f"Filtering genes with median TPM < {MIN_TPM}...")
    # Keep genes expressed in at least one tissue above threshold
    mask = (df.max(axis=1) >= MIN_TPM)
    df   = df[mask]
    print(f"  Kept {df.shape[0]} expressed genes")

    # Log-normalise
    log_df = np.log2(df + 1)
    print(f"  Log2(TPM+1) normalisation done")

    return log_df   # genes × tissues, log-scaled


# STEP 4: Select landmark + target genes
def split_genes(log_df: pd.DataFrame):
    """
    Randomly select landmark genes (inputs) and target genes (outputs).
    In production you'd use actual L1000 landmark gene list.
    """
    genes = log_df.index.tolist()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(genes)

    landmark_genes = genes[:N_LANDMARK]
    target_genes   = genes[N_LANDMARK : N_LANDMARK + N_TARGETS]

    print(f"  Landmark genes (inputs) : {len(landmark_genes)}")
    print(f"  Target genes (outputs)  : {len(target_genes)}")

    return landmark_genes, target_genes


# STEP 5: Build sample matrix
def build_tensors(log_df, landmark_genes, target_genes):
    """
    Each TISSUE is one training sample.
    X: landmark gene expressions for that tissue (978-dim)
    Y: target gene expressions for that tissue  (2000-dim)
    """
    # Transpose: tissues × genes
    data_T = log_df.T   # tissues × genes

    X = data_T[landmark_genes].values.astype(np.float32)   # (54, 978)
    Y = data_T[target_genes].values.astype(np.float32)     # (54, 2000)

    print(f"\n  X shape (inputs) : {X.shape}")
    print(f"  Y shape (targets): {Y.shape}")
    print(f"  Samples (tissues): {X.shape[0]}")

    # NOTE: 54 tissues is small. We augment by adding Gaussian noise.
    X, Y = augment(X, Y, n_copies=50)
    print(f"  After augmentation: {X.shape[0]} samples")

    # Standardise X
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Train / val / test split
    X_tv, X_test, Y_tv, Y_test = train_test_split(X, Y, test_size=0.1,  random_state=RANDOM_SEED)
    X_tr, X_val, Y_tr, Y_val   = train_test_split(X_tv, Y_tv, test_size=0.15, random_state=RANDOM_SEED)

    print(f"\n  Train : {X_tr.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    return (
        torch.tensor(X_tr),   torch.tensor(Y_tr),
        torch.tensor(X_val),  torch.tensor(Y_val),
        torch.tensor(X_test), torch.tensor(Y_test),
        scaler
    )


def augment(X, Y, n_copies=50, noise_std=0.05):
    """Add Gaussian noise copies to expand the small GTEx tissue dataset."""
    X_aug, Y_aug = [X], [Y]
    for _ in range(n_copies):
        X_aug.append(X + np.random.normal(0, noise_std, X.shape).astype(np.float32))
        Y_aug.append(Y + np.random.normal(0, noise_std, Y.shape).astype(np.float32))
    return np.vstack(X_aug), np.vstack(Y_aug)


def save_tensors(X_tr, Y_tr, X_val, Y_val, X_test, Y_test, scaler,
                 landmark_genes, target_genes):
    torch.save({"X": X_tr,  "Y": Y_tr},  DATA_DIR / "train.pt")
    torch.save({"X": X_val, "Y": Y_val}, DATA_DIR / "val.pt")
    torch.save({"X": X_test,"Y": Y_test},DATA_DIR / "test.pt")

    with open(DATA_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(DATA_DIR / "landmark_genes.txt", "w") as f:
        f.write("\n".join(landmark_genes))
    with open(DATA_DIR / "target_genes.txt", "w") as f:
        f.write("\n".join(target_genes))

    print(f"\n[✓] Tensors saved to {DATA_DIR}/")
    print(f"    train.pt, val.pt, test.pt, scaler.pkl")
    print(f"    landmark_genes.txt ({len(landmark_genes)} genes)")
    print(f"    target_genes.txt   ({len(target_genes)} genes)")


def main():
    print("=" * 60)
    print("DAY 1 — GTEx Data Preprocessing")
    print("=" * 60)

    df             = load_gtex(GTEX_FILE)
    log_df         = preprocess(df)
    landmark_genes, target_genes = split_genes(log_df)

    tensors = build_tensors(log_df, landmark_genes, target_genes)
    X_tr, Y_tr, X_val, Y_val, X_test, Y_test, scaler = tensors

    save_tensors(X_tr, Y_tr, X_val, Y_val, X_test, Y_test,
                 scaler, landmark_genes, target_genes)


if __name__ == "__main__":
    main()
