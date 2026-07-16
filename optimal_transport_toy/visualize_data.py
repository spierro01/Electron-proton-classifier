"""Validate and visualise the unpaired source and target toy samples."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EXPECTED_KEYS = (
    "source_train",
    "target_train",
    "source_test",
    "target_test",
)


def load_and_validate(path: Path) -> dict[str, np.ndarray]:
    """Load the exercise data and fail early if its schema is unexpected."""
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    with np.load(path) as archive:
        missing = [key for key in EXPECTED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Missing arrays in {path}: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in EXPECTED_KEYS}

    for name, values in arrays.items():
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                f"{name} must have shape (n_samples, 2), got {values.shape}"
            )
        if len(values) == 0:
            raise ValueError(f"{name} is empty")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values")

    if len(arrays["source_train"]) != len(arrays["target_train"]):
        raise ValueError("Source and target train arrays have different sizes")
    if len(arrays["source_test"]) != len(arrays["target_test"]):
        raise ValueError("Source and target test arrays have different sizes")
    return arrays


def print_summary(arrays: dict[str, np.ndarray]) -> None:
    print(f"{'array':<14} {'shape':>12} {'mean':>20} {'std':>20}")
    print("-" * 70)
    for name in EXPECTED_KEYS:
        values = arrays[name]
        mean = np.array2string(values.mean(axis=0), precision=3)
        std = np.array2string(values.std(axis=0), precision=3)
        print(f"{name:<14} {str(values.shape):>12} {mean:>20} {std:>20}")


def shared_ranges(arrays: dict[str, np.ndarray]) -> list[tuple[float, float]]:
    """Robust common plotting ranges, with a small margin around the data."""
    combined = np.concatenate(list(arrays.values()), axis=0)
    ranges = []
    for feature in range(2):
        low, high = np.quantile(combined[:, feature], [0.002, 0.998])
        margin = 0.05 * (high - low)
        ranges.append((float(low - margin), float(high + margin)))
    return ranges


def make_figure(
    arrays: dict[str, np.ndarray], output: Path, bins: int = 80
) -> None:
    x_range, y_range = shared_ranges(arrays)
    plot_range = [x_range, y_range]

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.05, 0.8))

    titles = {
        "source_train": "Source train",
        "target_train": "Target train",
        "source_test": "Source test",
        "target_test": "Target test",
    }
    for column, name in enumerate(EXPECTED_KEYS):
        ax = fig.add_subplot(grid[0, column])
        image = ax.hist2d(
            arrays[name][:, 0],
            arrays[name][:, 1],
            bins=bins,
            range=plot_range,
            density=True,
            cmap="viridis",
            cmin=1e-12,
        )
        fig.colorbar(image[3], ax=ax, label="density", shrink=0.82)
        ax.set_title(titles[name])
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    colors = {"source": "tab:blue", "target": "tab:orange"}
    styles = {"train": "-", "test": "--"}
    for feature in range(2):
        ax = fig.add_subplot(grid[1, 2 * feature : 2 * feature + 2])
        lo, hi = plot_range[feature]
        edges = np.linspace(lo, hi, bins + 1)
        for domain in ("source", "target"):
            for split in ("train", "test"):
                name = f"{domain}_{split}"
                density, _ = np.histogram(
                    arrays[name][:, feature], bins=edges, density=True
                )
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(
                    centers,
                    density,
                    color=colors[domain],
                    linestyle=styles[split],
                    linewidth=1.8,
                    label=f"{domain} {split}",
                )
        ax.set_title(f"Feature {feature} marginal")
        ax.set_xlabel(f"feature {feature}")
        ax.set_ylabel("density")
        ax.set_xlim(lo, hi)
        ax.legend(ncol=2)
        ax.grid(alpha=0.2)

    fig.suptitle("Optimal-transport toy data: unpaired source and target samples")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=HERE / "data" / "toy_ot_data.npz",
        help="Input NPZ file (default: exercise toy data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "figures" / "toy_ot_data.png",
        help="Output image (default: figures/toy_ot_data.png)",
    )
    parser.add_argument("--bins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins < 10:
        raise ValueError("--bins must be at least 10")
    arrays = load_and_validate(args.data)
    print_summary(arrays)
    make_figure(arrays, args.output, bins=args.bins)
    print(f"\nSaved visualisation to {args.output}")


if __name__ == "__main__":
    main()
