"""Task 2: PCA of z_enc, separately for each particle class.

- Load the Task 1 NPZ files.
- Fit PCA on MC only, at energies shared with TB validation.
- Apply the same MC centering and projection to both domains.
- Save plots with original and matched energy mixtures.
- No classifier inference, training or OT is performed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
MC_PATH = HERE / "derived/latents_MC.npz"
TB_PATH = HERE / "derived/latents_TB.npz"
MASK_PATH = HERE / "derived/tb_is_derivation.npy"
OUT = HERE / "results/task2_pca"

SEED = 42
MAX_PLOT_EVENTS = 20_000
PARTICLE_CLASSES = ["e", "p", "C"]


def load_latents(path, is_tb=False):
    with np.load(path, allow_pickle=False) as data:
        result = {
            "z": data["z_enc"],
            "label": data["label"],
            "energy": data["energy_mev"],
            "classes": data["classes"].tolist(),
            "checkpoint": str(data["checkpoint_sha256"].item()),
            "row": data["parquet_row"],
        }
        if is_tb:
            result["is_derivation"] = data["is_derivation"]

    n = len(result["label"])

    if result["z"].shape != (n, 64):
        raise ValueError(f"Unexpected latent shape in {path}")

    if result["energy"].shape != (n,):
        raise ValueError(f"Unexpected energy shape in {path}")

    if not np.array_equal(result["row"], np.arange(n)):
        raise ValueError(f"Unexpected event order in {path}")

    if not np.issubdtype(result["label"].dtype, np.integer):
        raise ValueError("Labels must be integer class indices.")

    if (
        np.any(result["label"] < 0)
        or np.any(result["label"] >= len(result["classes"]))
    ):
        raise ValueError("Invalid class indices.")

    # Check large arrays in chunks.
    for start in range(0, n, 50_000):
        if not np.isfinite(result["z"][start:start + 50_000]).all():
            raise ValueError(f"Non-finite latents in {path}")

    if not np.isfinite(result["energy"]).all():
        raise ValueError(f"Non-finite energies in {path}")

    if is_tb:
        mask = result["is_derivation"]
        if mask.dtype != np.bool_ or mask.shape != (n,):
            raise ValueError("Invalid TB split in NPZ.")

        original_mask = np.load(MASK_PATH, allow_pickle=False)
        if not np.array_equal(mask, original_mask):
            raise ValueError("NPZ split differs from the original TB split.")

    return result


def random_subset(indices, maximum, rng):
    size = min(maximum, len(indices))
    return np.sort(rng.choice(indices, size=size, replace=False))


def matched_energy_subsets(mc, tb, idx_mc, idx_tb, energies, rng):
    """Equal numbers per energy and per domain, without replacement."""
    mc_groups = [
        idx_mc[mc["energy"][idx_mc] == energy]
        for energy in energies
    ]
    tb_groups = [
        idx_tb[tb["energy"][idx_tb] == energy]
        for energy in energies
    ]

    per_energy = min(
        MAX_PLOT_EVENTS // len(energies),
        min(len(group) for group in mc_groups),
        min(len(group) for group in tb_groups),
    )

    if per_energy < 1:
        raise ValueError("Not enough events for energy matching.")

    selected_mc = np.concatenate([
        rng.choice(group, size=per_energy, replace=False)
        for group in mc_groups
    ])
    selected_tb = np.concatenate([
        rng.choice(group, size=per_energy, replace=False)
        for group in tb_groups
    ])

    return np.sort(selected_mc), np.sort(selected_tb), per_energy


def main():
    for path in [MC_PATH, TB_PATH, MASK_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    mc = load_latents(MC_PATH)
    tb = load_latents(TB_PATH, is_tb=True)

    if mc["classes"] != tb["classes"]:
        raise ValueError("MC and TB use different class orders.")

    if mc["checkpoint"] != tb["checkpoint"]:
        raise ValueError("MC and TB were extracted with different checkpoints.")

    for cls in PARTICLE_CLASSES:
        if cls not in mc["classes"]:
            raise ValueError(f"Missing class: {cls}")

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    figures = {}
    for mode in ["original_mixture", "matched_energy"]:
        fig, axes = plt.subplots(
            1, 3,
            figsize=(16, 5.5),
            layout="constrained",
        )
        figures[mode] = (fig, axes)

    summary = []
    inventory = []

    for panel, cls in enumerate(PARTICLE_CLASSES):
        class_index = mc["classes"].index(cls)

        idx_mc = np.flatnonzero(mc["label"] == class_index)
        idx_tb = np.flatnonzero(
            (tb["label"] == class_index)
            & (~tb["is_derivation"])
        )

        energies_mc = np.unique(mc["energy"][idx_mc])
        energies_tb = np.unique(tb["energy"][idx_tb])
        common = np.intersect1d(energies_mc, energies_tb)

        if len(common) == 0:
            raise ValueError(f"No common energies for {cls}")

        idx_mc = idx_mc[np.isin(mc["energy"][idx_mc], common)]
        idx_tb = idx_tb[np.isin(tb["energy"][idx_tb], common)]

        print(
            f"\n{cls}: PCA fit on {len(idx_mc):,} MC events; "
            f"{len(idx_tb):,} TB validation events available",
            flush=True,
        )
        print(f"Common energies: {common.tolist()}", flush=True)

        # No TB data are used for the PCA fit.
        # No additional per-domain standardisation is performed.
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(mc["z"][idx_mc])

        variance = pca.explained_variance_ratio_ * 100

        print(
            f"PC1: {variance[0]:.2f}% | "
            f"PC2: {variance[1]:.2f}% | "
            f"Total: {variance.sum():.2f}%",
            flush=True,
        )

        np.savez(
            OUT / f"pca_{cls}.npz",
            mean=pca.mean_,
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            mc_fit_rows=idx_mc,
            common_energies=common,
            classes=np.asarray(mc["classes"], dtype=str),
            checkpoint_sha256=np.asarray(mc["checkpoint"]),
        )

        original_mc = random_subset(idx_mc, MAX_PLOT_EVENTS, rng)
        original_tb = random_subset(idx_tb, MAX_PLOT_EVENTS, rng)

        matched_mc, matched_tb, per_energy = matched_energy_subsets(
            mc, tb, idx_mc, idx_tb, common, rng
        )

        selections = {
            "original_mixture": (original_mc, original_tb),
            "matched_energy": (matched_mc, matched_tb),
        }

        projections = {}
        for mode, (rows_mc, rows_tb) in selections.items():
            # transform() applies the MC mean and axes to both domains.
            projected_mc = pca.transform(mc["z"][rows_mc])
            projected_tb = pca.transform(tb["z"][rows_tb])

            projections[mode] = (projected_mc, projected_tb)

            np.savez(
                OUT / f"plot_samples_{cls}_{mode}.npz",
                mc_rows=rows_mc,
                tb_rows=rows_tb,
                mc_pca=projected_mc,
                tb_pca=projected_tb,
                seed=SEED,
            )

            for domain, data, rows in [
                ("MC", mc, rows_mc),
                ("TB_validation", tb, rows_tb),
            ]:
                for energy in common:
                    inventory.append({
                        "cls": cls,
                        "mode": mode,
                        "domain": domain,
                        "energy_mev": energy,
                        "n_plotted": int(
                            (data["energy"][rows] == energy).sum()
                        ),
                    })

        # Use identical axis limits for the two versions of each class.
        all_projected = np.vstack([
            array
            for pair in projections.values()
            for array in pair
        ])
        low = all_projected.min(axis=0)
        high = all_projected.max(axis=0)
        padding = np.maximum(high - low, 1e-6) * 0.05

        for mode, (projected_mc, projected_tb) in projections.items():
            _, axes = figures[mode]
            ax = axes[panel]

            ax.scatter(
                projected_mc[:, 0], projected_mc[:, 1],
                s=3, alpha=0.15, color="tab:blue",
                label=f"MC (n={len(projected_mc):,})",
                rasterized=True,
            )
            ax.scatter(
                projected_tb[:, 0], projected_tb[:, 1],
                s=3, alpha=0.15, color="tab:orange",
                label=f"TB validation (n={len(projected_tb):,})",
                rasterized=True,
            )

            ax.set_title({
                "e": "Electrons",
                "p": "Protons",
                "C": "Carbon",
            }[cls])
            ax.set_xlabel(f"PC1 ({variance[0]:.1f}% MC variance)")
            ax.set_ylabel(f"PC2 ({variance[1]:.1f}% MC variance)")
            ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
            ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
            ax.grid(alpha=0.2)
            ax.legend(markerscale=3, fontsize=9)

        summary.append({
            "cls": cls,
            "n_MC_fit": len(idx_mc),
            "n_TB_validation_available": len(idx_tb),
            "PC1_variance_percent": float(variance[0]),
            "PC2_variance_percent": float(variance[1]),
            "total_variance_percent": float(variance.sum()),
            "matched_events_per_energy_per_domain": per_energy,
            "common_energies": common.tolist(),
        })

    titles = {
        "original_mixture": (
            "PCA fitted on MC only, separately by class\n"
            "Original energy mixtures; random plotting subsets"
        ),
        "matched_energy": (
            "Same MC-fitted PCA axes\n"
            "Equal event counts per common energy and domain"
        ),
    }

    for mode, (fig, _) in figures.items():
        fig.suptitle(titles[mode])
        fig.savefig(OUT / f"pca_{mode}.png", dpi=200)
        fig.savefig(OUT / f"pca_{mode}.svg")
        plt.close(fig)

    pd.DataFrame(inventory).to_csv(
        OUT / "plot_energy_counts.csv", index=False
    )
    pd.DataFrame(summary).drop(columns="common_energies").to_csv(
        OUT / "pca_summary.csv", index=False
    )

    report = {
        "seed": SEED,
        "checkpoint_sha256": mc["checkpoint"],
        "classes": mc["classes"],
        "PCA_fit": "All MC events of each class at common energies.",
        "TB_selection": "Existing validation subset only.",
        "additional_standardisation": False,
        "summary": summary,
        "limitations": [
            "PCA visualises only two linear projections of 64 dimensions.",
            "PCA axes differ between particle classes.",
            "Energy matching does not match bias or other run conditions.",
            "Overlapping projections do not establish distributional closure.",
        ],
    }

    (OUT / "pca_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to: {OUT}", flush=True)


if __name__ == "__main__":
    main()