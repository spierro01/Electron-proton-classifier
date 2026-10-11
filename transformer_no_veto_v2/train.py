"""Train the ParticleTransformer PID classifier on the preprocessed tensors.

Recipe: class-weighted cross-entropy (inverse frequency, so rare classes are
not ignored), AdamW + OneCycleLR, checkpoint on best validation macro-F1.
The checkpoint is self-contained — weights + normalisation stats + class names
+ hyperparameters — so it can be reloaded without re-reading the training data.

Usage:
    python train.py --data derived/pid --epochs 30 --batch_size 2048
    python train.py --data derived/pid --epochs 1 --limit 50000   # smoke test

Outputs:
    <--ckpt>/best.pt              best checkpoint (by val macro-F1)
    <--data>/split_indices.npz    train/val/test row indices (reproducible)
    <--data>/test_metrics.json    final metrics on the held-out test split
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from pid_model import ParticleTransformer


class TensorDataset(Dataset):
    """Indexes into the memory-mapped X and normalises on the fly."""

    def __init__(self, X, y, indices, mean, std):
        self.X, self.y, self.idx = X, y, indices
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = (self.X[j].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x), int(self.y[j])


def compute_stats(X, indices, chunk=200_000):
    """Per-(layer, feature) mean/std over the TRAIN split only (no leakage)."""
    s = np.zeros(X.shape[1:], dtype=np.float64)
    s2 = np.zeros(X.shape[1:], dtype=np.float64)
    for start in range(0, len(indices), chunk):
        b = X[indices[start:start + chunk]].astype(np.float64)
        s += b.sum(axis=0)
        s2 += (b ** 2).sum(axis=0)
    n = len(indices)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    std[std < 1e-6] = 1.0  # constant channels -> leave untouched
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=range(num_classes)).tolist(),
    }
    try:
        if num_classes == 2:
            out["macro_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            out["macro_auc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Train the ParticleTransformer on preprocessed (N,21,8) tensors.")
    ap.add_argument("--data", type=Path, default=Path("derived/pid"),
                    help="directory produced by preprocess.py")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/pid_transformer"),
                    help="where to write best.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--ffn_dim", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N events (smoke test); still class-mixed")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.ckpt.mkdir(parents=True, exist_ok=True)

    X = np.load(args.data / "X.npy", mmap_mode="r")
    y = np.load(args.data / "y.npy")
    classes = np.load(args.data / "classes.npy").tolist()
    n = len(y) if args.limit is None else min(args.limit, len(y))
    num_classes = len(classes)
    print(f"Events: {n:,}   classes: {classes}   device: {device}")

    rng = np.random.default_rng(args.seed)
    # rows in the file are ordered by class -> always permute the FULL range
    # before truncating, so a --limit subset stays class-mixed
    perm = rng.permutation(len(y))[:n]
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    idx_test, idx_val, idx_train = np.split(perm, [n_test, n_test + n_val])
    np.savez(args.data / "split_indices.npz",
             train=idx_train, val=idx_val, test=idx_test, seed=args.seed)

    print("Computing normalisation stats on train split …")
    mean, std = compute_stats(X, np.sort(idx_train))

    loaders = {}
    for name, idx, shuffle in [("train", idx_train, True),
                               ("val", idx_val, False),
                               ("test", idx_test, False)]:
        loaders[name] = DataLoader(
            TensorDataset(X, y, idx, mean, std),
            batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
            persistent_workers=args.num_workers > 0)

    counts = np.bincount(y[idx_train], minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    print("Class weights (inv_freq):",
          {c: round(w, 3) for c, w in zip(classes, weights)})

    model = ParticleTransformer(
        num_classes, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_dim=args.ffn_dim, dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {model.num_parameters:,}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(loaders["train"]))

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen, correct = time.time(), 0.0, 0, 0
        for x, yb in loaders["train"]:
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)

        val = evaluate(model, loaders["val"], device, num_classes)
        print(f"Epoch {epoch:>3}/{args.epochs} | "
              f"train loss {running/seen:.4f} acc {correct/seen:.4f} | "
              f"val macro-F1 {val['macro_f1']:.4f} acc {val['accuracy']:.4f} "
              f"AUC {val['macro_auc']:.4f} | {time.time()-t0:.1f}s")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "val_macro_f1": best_f1,
                "args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
            }, args.ckpt / "best.pt")

    print("\nEvaluating best checkpoint on the held-out test split …")
    ckpt = torch.load(args.ckpt / "best.pt", map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, loaders["test"], device, num_classes)
    (args.data / "test_metrics.json").write_text(json.dumps(
        {"classes": classes, **test}, indent=2))
    print(json.dumps({k: v for k, v in test.items() if k != "confusion"},
                     indent=2))
    print("Confusion matrix (rows = true, cols = pred):")
    for c, row in zip(classes, test["confusion"]):
        print(f"  {c:>3}: {row}")
    print(f"\nCheckpoint → {args.ckpt / 'best.pt'}")


if __name__ == "__main__":
    main()
