"""
Lightweight Transformer encoder for gene expression prediction.
Treats each landmark gene as a "token" — the model learns gene-gene
attention patterns before predicting target gene expression.

Architecture:
    Embedding (978 genes × embed_dim)
         ↓
    TransformerEncoder (n_layers, n_heads, dim_feedforward)
         ↓
    Global mean pool over gene tokens
         ↓
    Linear → Output (2000 target genes)

Run:
    python train_transformer.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs/transformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIM   = 978
OUTPUT_DIM  = 2000
EMBED_DIM   = 64       # each gene projected to 64-dim embedding
N_HEADS     = 4        # attention heads
N_LAYERS    = 2        # transformer encoder layers
FF_DIM      = 256      # feedforward dim inside transformer
DROPOUT     = 0.2

LR          = 5e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 150
BATCH_SIZE  = 64
PATIENCE    = 20
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneExprTransformer(nn.Module):
    """
    Each input gene is treated as one token.
    Token embedding: scalar expression value → embed_dim vector.
    TransformerEncoder learns cross-gene attention.
    Output: predict all target gene expression values.
    """
    def __init__(self, n_genes, out_dim, embed_dim, n_heads, n_layers,
                 ff_dim, dropout):
        super().__init__()
        self.n_genes   = n_genes
        self.embed_dim = embed_dim

        # Each gene's scalar expression → embed_dim via linear projection
        self.gene_embed = nn.Linear(1, embed_dim)

        # Learnable positional (gene identity) embedding
        self.pos_embed  = nn.Embedding(n_genes, embed_dim)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True    # (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Head: pool + project to output
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        """
        x: (batch, n_genes)  — landmark gene expression values
        """
        B, G = x.shape

        # Embed each gene: (B, G, 1) → (B, G, embed_dim)
        gene_tokens = self.gene_embed(x.unsqueeze(-1))

        # Add positional (gene identity) embeddings
        positions   = torch.arange(G, device=x.device).unsqueeze(0)  # (1, G)
        gene_tokens = gene_tokens + self.pos_embed(positions)

        # Transformer: (B, G, embed_dim) → (B, G, embed_dim)
        encoded = self.transformer(gene_tokens)

        # Global mean pool over gene tokens: (B, embed_dim)
        pooled  = encoded.mean(dim=1)

        # Project to output: (B, out_dim)
        return self.head(pooled)


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

    best_val  = float("inf")
    patience  = 0
    history   = {"train_loss": [], "val_loss": []}

    print(f"\nTraining Transformer on {DEVICE}")
    print(f"Embed dim={EMBED_DIM}, Heads={N_HEADS}, Layers={N_LAYERS}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(X_b), Y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                val_loss += criterion(model(X_b), Y_b).item()
        val_loss /= len(val_loader)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_transformer.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    return history


# ── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test.to(DEVICE)).cpu().numpy()
    Y_true = Y_test.numpy()

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


# ── CAPTUM INTERPRETABILITY ───────────────────────────────────────────────────
def run_captum(model, X_test, output_dir):
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("\n Captum not installed. Run: pip install captum")
        return

    print("\n Captum Integrated Gradients")

    model.eval()
    X_sample = X_test[:20].to(DEVICE)
    baseline = torch.zeros_like(X_sample)

    #generate gene names matching actual input size
    actual_n = X_test.shape[1]
    with open(DATA_DIR / "landmark_genes.txt") as f:
        landmark_genes = [l.strip() for l in f.readlines()]
    with open(DATA_DIR / "target_genes.txt") as f:
        target_genes = [l.strip() for l in f.readlines()]

    # Pad to match actual dimensions
    while len(landmark_genes) < actual_n:
        landmark_genes.append(f"gene_{len(landmark_genes)}")
    while len(target_genes) < X_test.shape[1]:
        target_genes.append(f"target_{len(target_genes)}")


    ig = IntegratedGradients(model)
    n_targets_to_explain = 5
    all_attrs = []

    for target_idx in range(n_targets_to_explain):
        attrs = ig.attribute(X_sample, baselines=baseline,
                             target=target_idx, n_steps=50)
        all_attrs.append(attrs.detach().cpu().numpy())

    all_attrs  = np.array(all_attrs)
    mean_attr  = np.abs(all_attrs).mean(axis=1)
    overall_importance = mean_attr.mean(axis=0)
    top_idx    = np.argsort(overall_importance)[::-1][:20]
    top_genes  = [landmark_genes[i] for i in top_idx]
    top_vals   = overall_importance[top_idx]

    print(f"\n  Top 20 most influential landmark genes:")
    for gene, val in zip(top_genes, top_vals):
        print(f"    {gene:20s}  importance={val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_genes[::-1], top_vals[::-1], color="#1B5E20")
    ax.set_xlabel("Mean |Integrated Gradient|")
    ax.set_title("Captum: Top Landmark Genes Driving\nGene Expression Predictions")
    plt.tight_layout()
    plt.savefig(output_dir / "captum_feature_importance.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mean_attr[:, top_idx], aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(20))
    ax.set_xticklabels(top_genes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_targets_to_explain))
    ax.set_yticklabels([target_genes[i] for i in range(n_targets_to_explain)], fontsize=8)
    ax.set_title("Attribution Heatmap: Landmark → Target Genes")
    plt.colorbar(im, ax=ax, label="|IG attribution|")
    plt.tight_layout()
    plt.savefig(output_dir / "captum_attribution_heatmap.png", dpi=150)
    plt.close()

    print(f"\n Captum plots saved to {output_dir}/")


def compare_models(mlp_metrics_path, transformer_metrics, output_dir):
    """Bar chart comparing MLP vs Transformer on key metrics."""
    try:
        with open(mlp_metrics_path) as f:
            mlp_metrics = json.load(f)
    except FileNotFoundError:
        print("[!] MLP metrics not found — run train_mlp.py first for comparison")
        return

    metrics_to_compare = ["pearson_mean", "r2", "mae"]
    labels = ["Pearson r (↑)", "R² (↑)", "MAE (↓)"]

    mlp_vals = [mlp_metrics[m]         for m in metrics_to_compare]
    tf_vals  = [transformer_metrics[m] for m in metrics_to_compare]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, mlp_vals, w, label="MLP",         color="#1565C0")
    ax.bar(x + w/2, tf_vals,  w, label="Transformer", color="#1B5E20")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("MLP vs Transformer — Gene Expression Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mlp_vs_transformer.png", dpi=150)
    plt.close()
    print(f" Model comparison plot saved")


def main():
    print("=" * 60)
    print("DAY 3 — Transformer + Captum Interpretability")
    print("=" * 60)

    train_loader, val_loader, X_test, Y_test = load_data()  # ← once, at top

    actual_input_dim  = X_test.shape[1]
    actual_output_dim = Y_test.shape[1]
    print(f"Input dim: {actual_input_dim}, Output dim: {actual_output_dim}")

    model = GeneExprTransformer(
        n_genes=actual_input_dim, out_dim=actual_output_dim,
        embed_dim=EMBED_DIM, n_heads=N_HEADS,
        n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    history = train(model, train_loader, val_loader)
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_transformer.pt", map_location=DEVICE))
    metrics, gene_pearson, Y_true, Y_pred = evaluate(model, X_test, Y_test)

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # training curves
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"],   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Transformer Training Curves"); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()

    run_captum(model, X_test, OUTPUT_DIR)
    compare_models(Path("outputs/mlp/metrics.json"), metrics, OUTPUT_DIR)

    print(f"    Transformer: Pearson r = {metrics['pearson_mean']:.4f}, R² = {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()