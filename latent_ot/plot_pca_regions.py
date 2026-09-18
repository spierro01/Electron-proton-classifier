"""Plot PCA density contours using the original scatter axis limits.

- Reuses saved PCA projections.
- No PCA fitting, model inference or OT training.
- Axis limits match the original scatter:
  min/max across both mixture versions, with 5% padding.
- No automatic zoom.
- Saves contours alone and contours over the original points.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


ROOT = Path(r"C:\Progetti\Electron-proton-classifier\latent_ot")
SOURCE = ROOT / "results/task2_pca"

# New directory to avoid confusing these figures with the zoomed versions.
OUT = ROOT / "results/pca_regions_same_scale"

MASS = 0.95
BINS = 160
SIGMA = 2.5

CLASSES = ["e", "p", "C"]
MODES = ["original_mixture", "matched_energy"]

TITLES = {
    "e": "Electrons",
    "p": "Protons",
    "C": "Carbon",
}


def load_projections(cls):
    projections = {}

    for mode in MODES:
        path = SOURCE / f"plot_samples_{cls}_{mode}.npz"

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            mc = data["mc_pca"]
            tb = data["tb_pca"]

        for points in [mc, tb]:
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or len(points) == 0
            ):
                raise ValueError(f"Invalid projection shape in {path}")

            if not np.isfinite(points).all():
                raise ValueError(f"Non-finite coordinates in {path}")

        projections[mode] = [mc, tb]

    return projections


def estimate_density(points, x_edges, y_edges):
    histogram, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[x_edges, y_edges],
    )

    if not np.isclose(histogram.sum(), len(points)):
        raise ValueError("Some events are outside the density grid.")

    probability = gaussian_filter(
        histogram,
        sigma=SIGMA,
        mode="constant",
        cval=0.0,
    )

    total = probability.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid density estimate.")

    probability /= total

    ordered = np.sort(probability.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]

    index = np.searchsorted(cumulative, MASS, side="left")
    index = min(index, len(ordered) - 1)
    level = float(ordered[index])

    if not 0 < level < probability.max():
        raise ValueError("Unresolved density threshold.")

    return probability, level


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    report = {
        "mass": MASS,
        "bins": BINS,
        "sigma_bins": SIGMA,
        "method": "Gaussian-smoothed 2D histogram",
        "axis_limits": (
            "Original scatter limits: min/max over both mixture "
            "versions, with 5% padding. No zoom."
        ),
        "note": (
            "Contours describe estimated probability mass, "
            "not confidence intervals or the boundary of all events. "
            "Smoothing still affects contour shapes."
        ),
        "panels": [],
    }

    figures = {}

    for mode in MODES:
        for view in ["contours", "points_contours"]:
            figures[(mode, view)] = plt.subplots(
                1,
                3,
                figsize=(16, 5.5),
                layout="constrained",
            )

    for column, cls in enumerate(CLASSES):
        projections = load_projections(cls)

        all_points = np.vstack([
            points
            for pair in projections.values()
            for points in pair
        ])

        low = all_points.min(axis=0)
        high = all_points.max(axis=0)
        span = np.maximum(high - low, 1e-6)

        # EXACTLY the same rule as the original scatter script.
        axis_padding = 0.05 * span

        x_limits = (
            float(low[0] - axis_padding[0]),
            float(high[0] + axis_padding[0]),
        )
        y_limits = (
            float(low[1] - axis_padding[1]),
            float(high[1] + axis_padding[1]),
        )

        # Keep the previous density-estimation grid unchanged.
        # The estimation grid and the displayed axis limits need not coincide.
        density_padding = 0.12 * span

        x_edges = np.linspace(
            low[0] - density_padding[0],
            high[0] + density_padding[0],
            BINS + 1,
        )
        y_edges = np.linspace(
            low[1] - density_padding[1],
            high[1] + density_padding[1],
            BINS + 1,
        )

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        with np.load(
            SOURCE / f"pca_{cls}.npz",
            allow_pickle=False,
        ) as data:
            variance = 100 * data["explained_variance_ratio"]

        for mode in MODES:
            mc, tb = projections[mode]

            for points, name, color, style in [
                (mc, "MC", "#2166ac", "solid"),
                (tb, "TB validation", "#d95f02", "dashed"),
            ]:
                probability, level = estimate_density(
                    points, x_edges, y_edges
                )

                report["panels"].append({
                    "mode": mode,
                    "class": cls,
                    "domain": name,
                    "n_events": len(points),
                    "enclosed_grid_mass": float(
                        probability[probability >= level].sum()
                    ),
                    "sigma_PC1": float(
                        SIGMA * (x_edges[1] - x_edges[0])
                    ),
                    "sigma_PC2": float(
                        SIGMA * (y_edges[1] - y_edges[0])
                    ),
                    "x_limits": list(x_limits),
                    "y_limits": list(y_limits),
                })

                for view in ["contours", "points_contours"]:
                    _, axes = figures[(mode, view)]
                    ax = axes[column]

                    if view == "points_contours":
                        ax.scatter(
                            points[:, 0],
                            points[:, 1],
                            s=3,
                            alpha=0.15,
                            color=color,
                            edgecolors="none",
                            rasterized=True,
                            zorder=1,
                        )
                    else:
                        ax.contourf(
                            x_centers,
                            y_centers,
                            probability.T,
                            levels=[
                                level,
                                float(probability.max()) * 1.001,
                            ],
                            colors=[color],
                            alpha=0.12,
                            zorder=1,
                        )

                    ax.contour(
                        x_centers,
                        y_centers,
                        probability.T,
                        levels=[level],
                        colors=[color],
                        linestyles=[style],
                        linewidths=1.8,
                        zorder=3,
                    )

            for view in ["contours", "points_contours"]:
                _, axes = figures[(mode, view)]
                ax = axes[column]

                ax.set_title(TITLES[cls], fontsize=14)
                ax.set_xlabel(
                    f"PC1 ({variance[0]:.1f}% MC variance)"
                )
                ax.set_ylabel(
                    f"PC2 ({variance[1]:.1f}% MC variance)"
                )

                # Same axis limits for every view and mixture of this class.
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits)

                ax.grid(alpha=0.2)
                ax.set_axisbelow(True)

                handles = [
                    Line2D(
                        [0], [0],
                        color="#2166ac",
                        linewidth=1.8,
                        linestyle="-",
                        label=f"MC (n={len(mc):,})",
                    ),
                    Line2D(
                        [0], [0],
                        color="#d95f02",
                        linewidth=1.8,
                        linestyle="--",
                        label=f"TB validation (n={len(tb):,})",
                    ),
                ]
                ax.legend(handles=handles, fontsize=9)

        print(
            f"{cls}: original scatter axis limits preserved.",
            flush=True,
        )

    for (mode, view), (fig, _) in figures.items():
        mixture = (
            "Original energy mixtures"
            if mode == "original_mixture"
            else "Matched energy mixtures"
        )

        representation = (
            "Points and estimated 95% density contours"
            if view == "points_contours"
            else "Estimated 95% density regions"
        )

        fig.suptitle(
            f"{mixture} | PCA fitted on MC only\n"
            f"{representation} — original scatter axis limits",
            fontsize=13,
        )

        for extension in ["png", "pdf", "svg"]:
            path = OUT / f"pca_{mode}_{view}.{extension}"
            fig.savefig(
                path,
                dpi=220,
                bbox_inches="tight",
            )
            print(f"Saved: {path}", flush=True)

        plt.close(fig)

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()