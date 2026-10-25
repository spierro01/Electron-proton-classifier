"""Redraw Task-2 PCA projections using density contours.

Uses the exact plotting samples saved by plot_pca_by_class.py.
No PCA refitting, inference or training is performed.

Density estimation:
- shared 2D histogram grid for MC and TB;
- same Gaussian smoothing in both domains;
- each density is normalised separately.

Contours enclose approximately 68%, 95% and 99%
of the estimated probability mass.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/task2_pca"
OUT = HERE / "results/task2_pca_contours"

CLASSES = ["e", "p", "C"]
MODES = ["original_mixture", "matched_energy"]

TITLES = {
    "e": "Electrons",
    "p": "Protons",
    "C": "Carbon",
}

# Same grid and smoothing for the two domains of each class.
N_BINS = 220
SMOOTHING_SIGMA = 1.2  # In histogram-bin units.
MASSES = [0.68, 0.95, 0.99]

# Faint points retain visibility of sparse populations and tails.
SHOW_BACKGROUND_POINTS = True


def load_projection(cls, mode):
    path = SOURCE / f"plot_samples_{cls}_{mode}.npz"

    if not path.is_file():
        raise FileNotFoundError(
            f"{path}\n"
            "Run the original plot_pca_by_class.py first to save projections."
        )

    with np.load(path, allow_pickle=False) as data:
        mc = data["mc_pca"]
        tb = data["tb_pca"]

    for array in [mc, tb]:
        if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
            raise ValueError(f"Invalid PCA projection in {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite projection in {path}")

    return mc, tb


def density_grid(points, x_edges, y_edges):
    histogram, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[x_edges, y_edges],
    )

    if not np.isclose(histogram.sum(), len(points)):
        raise ValueError("Some events lie outside the histogram grid.")

    smoothed = gaussian_filter(
        histogram.astype(float),
        sigma=SMOOTHING_SIGMA,
        mode="constant",
        cval=0.0,
    )

    # Every cell has the same area.
    # Normalising the cell masses is enough to find density thresholds.
    total = smoothed.sum()
    if total <= 0:
        raise ValueError("Empty density estimate.")

    return smoothed / total


def enclosed_mass_levels(probability):
    """Thresholds for highest-density regions on the grid."""

    sorted_values = np.sort(probability.ravel())[::-1]
    cumulative = np.cumsum(sorted_values)

    thresholds = {}

    for mass in MASSES:
        index = np.searchsorted(cumulative, mass, side="left")
        index = min(index, len(sorted_values) - 1)
        level = float(sorted_values[index])

        if not 0 < level < probability.max():
            raise ValueError(
                "Cannot resolve contour levels; inspect binning or data."
            )

        # A discrete grid can give the same threshold for two masses.
        thresholds.setdefault(level, []).append(mass)

    return thresholds


def draw_contours(ax, points, x_edges, y_edges, color, linestyle):
    probability = density_grid(points, x_edges, y_edges)
    thresholds = enclosed_mass_levels(probability)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    levels = sorted(thresholds)

    contours = ax.contour(
        x_centers,
        y_centers,
        probability.T,
        levels=levels,
        colors=color,
        linestyles=linestyle,
        linewidths=1.5,
        zorder=3,
    )

    labels = {
        level: "/".join(
            f"{100 * mass:.0f}%"
            for mass in thresholds[level]
        )
        for level in levels
    }

    ax.clabel(
        contours,
        fmt=labels,
        inline=True,
        fontsize=8,
    )

    # Record the actual discrete-grid mass above each threshold.
    return [
        {
            "requested_masses": thresholds[level],
            "threshold_cell_probability": level,
            "enclosed_grid_mass": float(
                probability[probability >= level].sum()
            ),
        }
        for level in levels
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    figures = {}
    for mode in MODES:
        figures[mode] = plt.subplots(
            1, 3,
            figsize=(17, 6),
            layout="constrained",
        )

    report = {
        "source": str(SOURCE),
        "n_bins_per_axis": N_BINS,
        "smoothing_sigma_bins": SMOOTHING_SIGMA,
        "requested_enclosed_masses": MASSES,
        "background_points": SHOW_BACKGROUND_POINTS,
        "density_method": "Gaussian-smoothed 2D histogram",
        "normalisation": "Unit probability mass separately per domain",
        "panels": [],
        "limitations": [
            "Contour shapes depend on grid resolution and smoothing.",
            "Percentages refer to estimated enclosed probability mass.",
            "Contours are not confidence intervals.",
            "Two PCA coordinates cannot establish closure in 64 dimensions.",
        ],
    }

    for panel, cls in enumerate(CLASSES):
        projections = {
            mode: load_projection(cls, mode)
            for mode in MODES
        }

        with np.load(SOURCE / f"pca_{cls}.npz") as data:
            variance = 100 * data["explained_variance_ratio"]

        # Shared axes/grid across domains AND mixture versions.
        all_points = np.vstack([
            points
            for pair in projections.values()
            for points in pair
        ])

        low = all_points.min(axis=0)
        high = all_points.max(axis=0)
        padding = 0.05 * np.maximum(high - low, 1e-6)

        limits_low = low - padding
        limits_high = high + padding

        x_edges = np.linspace(
            limits_low[0], limits_high[0], N_BINS + 1
        )
        y_edges = np.linspace(
            limits_low[1], limits_high[1], N_BINS + 1
        )

        for mode, (mc, tb) in projections.items():
            _, axes = figures[mode]
            ax = axes[panel]

            if SHOW_BACKGROUND_POINTS:
                for points, color in [
                    (mc, "tab:blue"),
                    (tb, "tab:orange"),
                ]:
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=2,
                        alpha=0.07,
                        color=color,
                        edgecolors="none",
                        rasterized=True,
                        zorder=1,
                    )

            for domain, points, color, linestyle in [
                ("MC", mc, "tab:blue", "solid"),
                ("TB validation", tb, "tab:orange", "dashed"),
            ]:
                levels = draw_contours(
                    ax, points,
                    x_edges, y_edges,
                    color, linestyle,
                )

                report["panels"].append({
                    "cls": cls,
                    "mode": mode,
                    "domain": domain,
                    "n_events": len(points),
                    "levels": levels,
                    "x_limits": [
                        float(limits_low[0]), float(limits_high[0])
                    ],
                    "y_limits": [
                        float(limits_low[1]), float(limits_high[1])
                    ],
                })

            ax.set_title(TITLES[cls])
            ax.set_xlabel(f"PC1 ({variance[0]:.1f}% MC variance)")
            ax.set_ylabel(f"PC2 ({variance[1]:.1f}% MC variance)")
            ax.set_xlim(limits_low[0], limits_high[0])
            ax.set_ylim(limits_low[1], limits_high[1])
            ax.grid(alpha=0.2)

            handles = [
                Line2D(
                    [0], [0],
                    color="tab:blue",
                    linewidth=1.5,
                    label=f"MC (n={len(mc):,})",
                ),
                Line2D(
                    [0], [0],
                    color="tab:orange",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"TB validation (n={len(tb):,})",
                ),
            ]
            ax.legend(handles=handles, fontsize=9)

        print(f"{cls}: contour plots prepared.", flush=True)

    titles = {
        "original_mixture": (
            "PCA fitted on MC only, separately by class\n"
            "Original energy mixtures — 68%, 95%, 99% density contours"
        ),
        "matched_energy": (
            "Same MC-fitted PCA axes; matched energy mixtures\n"
            "68%, 95%, 99% density contours"
        ),
    }

    for mode, (fig, _) in figures.items():
        fig.suptitle(titles[mode])

        for extension in ["png", "pdf", "svg"]:
            path = OUT / f"pca_{mode}_contours.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")

        plt.close(fig)

    (OUT / "contour_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()