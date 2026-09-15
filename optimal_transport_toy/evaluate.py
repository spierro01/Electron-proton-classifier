"""Evaluate test-set closure for the neural OT toy exercise."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model import NeuralOT


def auc_for_two_samples(first: np.ndarray, second: np.ndarray, seed: int = 42) -> float:
    features = np.vstack([first, second])
    labels = np.r_[np.zeros(len(first), dtype=int), np.ones(len(second), dtype=int)]
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    classifier = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed
    )
    classifier.fit(x_train, y_train)
    return roc_auc_score(y_holdout, classifier.predict_proba(x_holdout)[:, 1])


def common_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    combined = np.vstack(arrays)
    low, high = np.quantile(combined, [0.005, 0.995], axis=0)
    margin = 0.08 * (high - low)
    return (low[0] - margin[0], high[0] + margin[0]), (low[1] - margin[1], high[1] + margin[1])


def plot_closure(
    source: np.ndarray, target: np.ndarray, mapped: np.ndarray, output: Path
) -> None:
    xlim, ylim = common_limits(source, target, mapped)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, values, title, colour in zip(
        axes[0],
        (source, target, mapped),
        ("Source test", "Target test", "Transported source test"),
        ("#1f77b4", "#d62728", "#6f42c1"),
    ):
        axis.scatter(values[:, 0], values[:, 1], s=4, alpha=0.28, c=colour, rasterized=True)
        axis.set(title=title, xlim=xlim, ylim=ylim, xlabel="x0", ylabel="x1")

    for coordinate, axis in enumerate(axes[1, :2]):
        axis.hist(target[:, coordinate], bins=60, density=True, alpha=0.55, label="target", color="#d62728")
        axis.hist(mapped[:, coordinate], bins=60, density=True, alpha=0.55, label="transported", color="#6f42c1")
        axis.set(title=f"Marginal x{coordinate}", xlabel=f"x{coordinate}", ylabel="density")
        axis.legend()
    axes[1, 2].axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/toy_ot_data.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/neural_ot.pt"))
    parser.add_argument("--output", type=Path, default=Path("figures/neural_ot_closure.png"))
    args = parser.parse_args()

    data = np.load(args.data)
    source_test, target_test = data["source_test"], data["target_test"]
    transport = NeuralOT.load(args.checkpoint)
    mapped = transport.transform(source_test)

    # Serialization is part of the requested deliverable.
    reloaded = NeuralOT.load(args.checkpoint, device=transport.device)
    reload_difference = np.max(np.abs(mapped[:100] - reloaded.transform(source_test[:100])))
    assert np.allclose(mapped[:100], reloaded.transform(source_test[:100]), atol=1e-6)

    print("per-coordinate statistics")
    for name, values in (("source", source_test), ("target", target_test), ("transported", mapped)):
        print(f"{name:12s} mean={values.mean(axis=0)}  std={values.std(axis=0)}")
    print(f"reload max absolute difference: {reload_difference:.3e}")
    print(f"ROC AUC before transport: {auc_for_two_samples(source_test, target_test):.4f}")
    print(f"ROC AUC after transport:  {auc_for_two_samples(mapped, target_test):.4f}")
    plot_closure(source_test, target_test, mapped, args.output)
    print(f"closure plot saved to {args.output}")


if __name__ == "__main__":
    main()