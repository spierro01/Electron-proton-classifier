"""Run a trained ParticleTransformer on events and print/save its predictions.

The checkpoint is self-contained: it carries the weights, the normalisation
stats, the class list and the architecture hyperparameters — so you do NOT need
the training data (or preprocess.py) to run inference. Point this at a parquet
and go.

If the parquet has a `particle_type` column (the MC files do), the truth is used
to also report accuracy and a confusion matrix. Real test-beam data has no
truth labels; predictions are still produced.

Usage:
    # try the provided pretrained model on some events
    python predict.py --input /path/to/dumpMC_spectra.parquet --limit 20000

    # your own checkpoint, save the predictions
    python predict.py --ckpt checkpoints/pid_transformer/best.pt \
        --input data.parquet --out predictions.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import pid_data as PD
from pid_model import ParticleTransformer

CHUNK = 100_000


def load_model(ckpt_path: Path, device: str):
    """Rebuild the network exactly as it was trained and load the weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    a = ckpt["args"]
    classes = ckpt["classes"]
    model = ParticleTransformer(
        num_classes=len(classes),
        d_model=a["d_model"], n_heads=a["n_heads"],
        n_layers=a["n_layers"], ffn_dim=a["ffn_dim"], dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return model, classes, mean, std, ckpt


@torch.no_grad()
def predict_matrix(model, X: np.ndarray, mean, std, device: str,
                   batch_size: int = 4096):
    """(N, 21, 8) raw -> (probabilities (N, C), predicted index (N,))."""
    probs = []
    for s in range(0, len(X), batch_size):
        xb = (X[s:s + batch_size] - mean) / std          # same norm as training
        t = torch.from_numpy(xb.astype(np.float32)).to(device)
        probs.append(torch.softmax(model(t), dim=1).cpu().numpy())
    P = np.concatenate(probs)
    return P, P.argmax(1)


def main():
    ap = argparse.ArgumentParser(
        description="Run a trained ParticleTransformer on a parquet of events.")
    ap.add_argument("--ckpt", type=Path, default=Path("pretrained/best.pt"),
                    help="checkpoint (default: the provided pretrained/best.pt)")
    ap.add_argument("--input", type=Path, required=True,
                    help="parquet with the detector columns")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N events (quick try)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional CSV to write predictions to")
    ap.add_argument("--batch_size", type=int, default=4096)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, mean, std, ckpt = load_model(args.ckpt, device)
    print(f"Checkpoint : {args.ckpt}")
    print(f"  classes  : {classes}")
    print(f"  val macro-F1 (at training time): {ckpt['val_macro_f1']:.4f}")
    print(f"  params   : {model.num_parameters:,}   device: {device}")

    have = set(pq.ParquetFile(args.input).schema.names)
    need = PD.required_columns()
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(
            f"{args.input} is missing {len(missing)} required detector columns, "
            f"e.g. {missing[:4]}.\nThis script expects the parquet schema "
            f"(RAW_TR1_HG_0_0, TR1_posx_0, …), not the older .pkl naming.")
    has_truth = "particle_type" in have
    cols = need + (["particle_type"] if has_truth else [])

    print(f"\nReading {args.input} …")
    df = pd.read_parquet(args.input, columns=cols)
    if args.limit is not None:
        # rows are ordered by class -> sample randomly, else you get one class
        rng = np.random.default_rng(0)
        n = min(args.limit, len(df))
        df = df.iloc[np.sort(rng.choice(len(df), n, replace=False))]
        df = df.reset_index(drop=True)
    print(f"  {len(df):,} events")

    preds, probs = [], []
    for s in range(0, len(df), CHUNK):
        X = PD.build_matrix(df.iloc[s:s + CHUNK])
        P, idx = predict_matrix(model, X, mean, std, device, args.batch_size)
        probs.append(P)
        preds.append(idx)
    probs = np.concatenate(probs)
    pred_idx = np.concatenate(preds)
    pred_name = np.array(classes)[pred_idx]

    print("\nPredicted composition:")
    for c, k in zip(*np.unique(pred_name, return_counts=True)):
        print(f"  {c:>3}: {k:>8,}  ({k / len(pred_name):6.2%})")

    if has_truth:
        # map truth into the checkpoint's class space (it may have been trained
        # on a different/larger class set than this file contains)
        truth = df["particle_type"].map(PD.LABEL_MAP)
        known = truth.isin(classes).to_numpy()
        if known.any():
            lut = {c: i for i, c in enumerate(classes)}
            y_true = truth[known].map(lut).to_numpy()
            y_pred = pred_idx[known]
            acc = float((y_true == y_pred).mean())
            print(f"\nAccuracy on {known.sum():,} labelled events: {acc:.4f}")
            print("Per-class accuracy:")
            for i, c in enumerate(classes):
                m = y_true == i
                if m.any():
                    print(f"  {c:>3}: {(y_pred[m] == i).mean():.4f}  "
                          f"(n={m.sum():,})")
            print("Confusion (rows = true, cols = pred, order "
                  f"{classes}):")
            for i, c in enumerate(classes):
                m = y_true == i
                if m.any():
                    row = [int((y_pred[m] == j).sum()) for j in range(len(classes))]
                    print(f"  {c:>3}: {row}")
        skipped = (~known).sum()
        if skipped:
            print(f"\n({skipped:,} events have a particle_type the model was "
                  f"not trained on — excluded from the metrics above.)")

    if args.out is not None:
        out = pd.DataFrame({"pred": pred_name})
        for i, c in enumerate(classes):
            out[f"p_{c}"] = probs[:, i]
        if has_truth:
            out["truth"] = df["particle_type"].map(PD.LABEL_MAP).to_numpy()
        out.to_csv(args.out, index=False)
        print(f"\nPredictions → {args.out}")


if __name__ == "__main__":
    main()
