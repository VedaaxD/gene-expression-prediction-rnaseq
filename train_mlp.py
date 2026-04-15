"""
Trains an MLP to predict gene expression (target genes) from landmark genes.

Architecture:
    Input (978) → Linear → BN → ReLU → Dropout
                → Linear → BN → ReLU → Dropout
                → Linear → BN → ReLU → Dropout
                → Output (2000)

Evaluation metrics:
    - Pearson correlation (per gene, averaged)
    - R² score
    - MSE / MAE
    - Gene-level error analysis (best/worst predicted genes)

Run:
    python train_mlp.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

DATA_DIR    = Path("data")
OUTPUT_DIR  = Path("outputs/mlp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIM   = 978
OUTPUT_DIM  = 2000

HIDDEN_DIMS = [1024, 1024, 512]   # MLP layer sizes
DROPOUT     = 0.3
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EPOCHS      = 150
BATCH_SIZE  = 64
PATIENCE    = 20                   # early stopping
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneExprMLP(nn.Module):
    """
    Multi-layer perceptron for gene expression prediction.
    BatchNorm + Dropout after each hidden layer for stability.
    """
    def __init__(self, in_dim, out_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev   = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data():
    train = torch.load(DATA_DIR / "train.pt")
    val   = torch.load(DATA_DIR / "val.pt")
    test  = torch.load(DATA_DIR / "test.pt")

    def to_loader(d, shuffle=True):
        ds = torch.utils.data.TensorDataset(d["X"], d["Y"])
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return to_loader(train), to_loader(val, False), test["X"], test["Y"]

def train(model, train_loader, val_loader):
    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_cnt  = 0
    history       = {"train_loss": [], "val_loss": []}

    print(f"\nTraining MLP on {DEVICE}")
    print(f"Architecture: {INPUT_DIM} → {HIDDEN_DIMS} → {OUTPUT_DIM}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), Y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_mlp.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    return history



def evaluate(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test.to(DEVICE)).cpu().numpy()
    Y_true = Y_test.numpy()

    # Per-gene Pearson correlation
    gene_pearson = []
    for i in range(Y_true.shape[1]):
        r, _ = pearsonr(Y_true[:, i], Y_pred[:, i])
        gene_pearson.append(r)
    gene_pearson = np.array(gene_pearson)

    metrics = {
        "pearson_mean":  float(np.nanmean(gene_pearson)),
        "pearson_median":float(np.nanmedian(gene_pearson)),
        "r2":            float(r2_score(Y_true.flatten(), Y_pred.flatten())),
        "mse":           float(mean_squared_error(Y_true, Y_pred)),
        "mae":           float(mean_absolute_error(Y_true, Y_pred)),
    }

    print("\n── Test Metrics ─────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    return metrics, gene_pearson, Y_true, Y_pred


# GENE-LEVEL ERROR ANALYSIS
def gene_error_analysis(gene_pearson, output_dir):
    """Identify best and worst predicted genes — for CV / interpretability."""
    with open(Path("data") / "target_genes.txt") as f:
        target_genes = [l.strip() for l in f.readlines()]

    ranked = sorted(zip(target_genes, gene_pearson), key=lambda x: x[1], reverse=True)

    print("\n── Top 10 Best Predicted Genes ──────────────")
    for gene, r in ranked[:10]:
        print(f"  {gene:20s}  r = {r:.3f}")

    print("\n── Top 10 Worst Predicted Genes ─────────────")
    for gene, r in ranked[-10:]:
        print(f"  {gene:20s}  r = {r:.3f}")

    # Save full ranking
    with open(output_dir / "gene_pearson_ranking.json", "w") as f:
        json.dump({"genes": [g for g,_ in ranked],
                   "pearson": [float(r) for _,r in ranked]}, f, indent=2)

    # Plot distribution
    plt.figure(figsize=(8, 4))
    plt.hist(gene_pearson, bins=50, color="#2196F3", edgecolor="white", linewidth=0.5)
    plt.axvline(np.nanmean(gene_pearson), color="red", linestyle="--",
                label=f"Mean r = {np.nanmean(gene_pearson):.3f}")
    plt.xlabel("Pearson Correlation (per gene)")
    plt.ylabel("Number of Genes")
    plt.title("MLP: Gene-level Prediction Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gene_pearson_distribution.png", dpi=150)
    plt.close()
    print(f"\n[✓] Gene error analysis saved to {output_dir}/")


#PLOT TRAINING CURVES
def plot_history(history, output_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


#(predicted vs actual)
def plot_scatter(Y_true, Y_pred, output_dir):
    # Sample 5000 random points for clarity
    idx  = np.random.choice(Y_true.size, size=min(5000, Y_true.size), replace=False)
    yt   = Y_true.flatten()[idx]
    yp   = Y_pred.flatten()[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, alpha=0.2, s=5, color="#1B5E20")
    plt.plot([yt.min(), yt.max()], [yt.min(), yt.max()], "r--", lw=1.5, label="y=x")
    plt.xlabel("True Expression (log2 TPM+1)")
    plt.ylabel("Predicted Expression")
    plt.title("MLP: Predicted vs True Gene Expression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_true_vs_pred.png", dpi=150)
    plt.close()
    print(f"Scatter plot saved")


def main():
    print("=" * 60)
    print("DAY 2 — MLP Gene Expression Prediction")
    print("=" * 60)

    train_loader, val_loader, X_test, Y_test = load_data()   # ← once here

    actual_input_dim  = X_test.shape[1]
    actual_output_dim = Y_test.shape[1]
    print(f"Input dim: {actual_input_dim}, Output dim: {actual_output_dim}")

    model = GeneExprMLP(actual_input_dim, actual_output_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    history = train(model, train_loader, val_loader)
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_mlp.pt", map_location=DEVICE))
    metrics, gene_pearson, Y_true, Y_pred = evaluate(model, X_test, Y_test)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_history(history, OUTPUT_DIR)
    plot_scatter(Y_true, Y_pred, OUTPUT_DIR)
    gene_error_analysis(gene_pearson, OUTPUT_DIR)

if __name__ == "__main__":
    main()
