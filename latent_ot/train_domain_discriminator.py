"""Task 2: per-class MC-vs-TB discriminators on 64-dimensional z_enc.

Domain labels: MC=0, TB=1.
Energy-balanced sampling in train, internal validation and final test.
The existing TB derivation/validation split is never modified.

The final test is evaluated only after selecting the number of iterations.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve


HERE = Path(__file__).resolve().parent
MC_PATH = HERE / "derived/latents_MC.npz"
TB_PATH = HERE / "derived/latents_TB.npz"
MASK_PATH = HERE / "derived/tb_is_derivation.npy"

OUT = HERE / "results/task2_discriminator"

CLASSES = ["e", "p", "C"]
SEED = 42

# Maximum number per energy, per domain, in each subset.
TRAIN_PER_ENERGY = 2000
VALIDATION_PER_ENERGY = 500
TEST_PER_ENERGY = 1000

# Select the number of iterations using internal validation only.
ITERATION_CANDIDATES = [50, 100, 150, 200, 250, 300]


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
        raise ValueError(f"Unexpected latent shape: {path}")

    if result["energy"].shape != (n,):
        raise ValueError("Invalid energy shape.")

    if not np.array_equal(result["row"], np.arange(n)):
        raise ValueError("Unexpected parquet row order.")

    if not np.issubdtype(result["label"].dtype, np.integer):
        raise ValueError("Labels must contain integer indices.")

    if (
        np.any(result["label"] < 0)
        or np.any(result["label"] >= len(result["classes"]))
    ):
        raise ValueError("Invalid class indices.")

    for start in range(0, n, 50_000):
        if not np.isfinite(result["z"][start:start + 50_000]).all():
            raise ValueError("Non-finite latent values.")

    if not np.isfinite(result["energy"]).all():
        raise ValueError("Non-finite energy values.")

    if is_tb:
        mask = result["is_derivation"]
        original = np.load(MASK_PATH, allow_pickle=False)

        if mask.dtype != np.bool_ or mask.shape != (n,):
            raise ValueError("Invalid TB split.")

        if not np.array_equal(mask, original):
            raise ValueError("TB split differs from the original mask.")

    return result


def select_rows(mc, tb, cls, rng):
    """Create disjoint, energy-balanced subsets.

    TB derivation is split into discriminator training and internal validation.
    TB validation is reserved for the final domain-classification test.
    """
    class_index = mc["classes"].index(cls)

    mc_class = np.flatnonzero(mc["label"] == class_index)
    tb_derivation = np.flatnonzero(
        (tb["label"] == class_index) & tb["is_derivation"]
    )
    tb_validation = np.flatnonzero(
        (tb["label"] == class_index) & ~tb["is_derivation"]
    )

    common = np.intersect1d(
        np.unique(mc["energy"][mc_class]),
        np.unique(tb["energy"][tb_derivation]),
    )
    common = np.intersect1d(
        common,
        np.unique(tb["energy"][tb_validation]),
    )

    if len(common) == 0:
        raise ValueError(f"No common energies for {cls}")

    pools = {}

    # Partition before subsampling, independently at each energy.
    for energy in common:
        mc_rows = rng.permutation(
            mc_class[mc["energy"][mc_class] == energy]
        )
        tb_rows = rng.permutation(
            tb_derivation[tb["energy"][tb_derivation] == energy]
        )
        tb_test = rng.permutation(
            tb_validation[tb["energy"][tb_validation] == energy]
        )

        mc_train_end = int(0.6 * len(mc_rows))
        mc_val_end = int(0.8 * len(mc_rows))
        tb_train_end = int(0.8 * len(tb_rows))

        pools[float(energy)] = {
            "mc_train": mc_rows[:mc_train_end],
            "mc_val": mc_rows[mc_train_end:mc_val_end],
            "mc_test": mc_rows[mc_val_end:],
            "tb_train": tb_rows[:tb_train_end],
            "tb_val": tb_rows[tb_train_end:],
            "tb_test": tb_test,
        }

    caps = {
        "train": TRAIN_PER_ENERGY,
        "val": VALIDATION_PER_ENERGY,
        "test": TEST_PER_ENERGY,
    }

    selected = {}
    inventory = []

    for split, cap in caps.items():
        # Same count at every energy and in both domains.
        per_energy = min(
            [cap]
            + [
                len(pool[f"{domain}_{split}"])
                for pool in pools.values()
                for domain in ["mc", "tb"]
            ]
        )

        if per_energy < 1:
            raise ValueError(f"Empty {split} subset for {cls}")

        for domain in ["mc", "tb"]:
            chunks = []

            for energy, pool in pools.items():
                rows = pool[f"{domain}_{split}"][:per_energy]
                chunks.append(rows)

                inventory.append({
                    "cls": cls,
                    "split": split,
                    "domain": domain.upper(),
                    "energy_mev": energy,
                    "n": len(rows),
                })

            selected[f"{domain}_{split}"] = np.concatenate(chunks)

    # Explicit independence checks.
    for domain in ["mc", "tb"]:
        for first, second in [
            ("train", "val"), ("train", "test"), ("val", "test")
        ]:
            overlap = np.intersect1d(
                selected[f"{domain}_{first}"],
                selected[f"{domain}_{second}"],
            )
            if len(overlap):
                raise RuntimeError("Overlap between discriminator subsets.")

    if not tb["is_derivation"][selected["tb_train"]].all():
        raise RuntimeError("TB training contains validation events.")

    if not tb["is_derivation"][selected["tb_val"]].all():
        raise RuntimeError("Internal validation contains final TB events.")

    if tb["is_derivation"][selected["tb_test"]].any():
        raise RuntimeError("Final test contains TB derivation events.")

    return selected, common, inventory


def make_dataset(mc, tb, selected, split):
    mc_rows = selected[f"mc_{split}"]
    tb_rows = selected[f"tb_{split}"]

    X = np.concatenate([
        mc["z"][mc_rows],
        tb["z"][tb_rows],
    ])

    y = np.concatenate([
        np.zeros(len(mc_rows), dtype=np.int64),
        np.ones(len(tb_rows), dtype=np.int64),
    ])

    return X, y


def new_model(seed, iterations=50, warm_start=False):
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=iterations,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        warm_start=warm_start,
        random_state=seed,
    )


def main():
    for path in [MC_PATH, TB_PATH, MASK_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep prior results intact.
    if OUT.exists():
        raise FileExistsError(
            f"{OUT} already exists. Inspect existing results before rerunning."
        )

    mc = load_latents(MC_PATH)
    tb = load_latents(TB_PATH, is_tb=True)

    if mc["classes"] != tb["classes"]:
        raise ValueError("Different class order between domains.")

    if mc["checkpoint"] != tb["checkpoint"]:
        raise ValueError("Different PID checkpoints between domains.")

    for cls in CLASSES:
        if cls not in mc["classes"]:
            raise ValueError(f"Missing class {cls}")

    OUT.mkdir(parents=True)

    summaries = []
    inventories = []
    histories = []

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.8), layout="constrained"
    )

    for class_number, cls in enumerate(CLASSES):
        seed = SEED + class_number
        rng = np.random.default_rng(seed)

        selected, energies, inventory = select_rows(
            mc, tb, cls, rng
        )
        inventories.extend(inventory)

        np.savez(
            OUT / f"split_{cls}.npz",
            **selected,
            common_energies=energies,
            classes=np.asarray(mc["classes"], dtype=str),
            checkpoint_sha256=np.asarray(mc["checkpoint"]),
            seed=seed,
        )

        X_train, y_train = make_dataset(
            mc, tb, selected, "train"
        )
        X_val, y_val = make_dataset(
            mc, tb, selected, "val"
        )
        X_test, y_test = make_dataset(
            mc, tb, selected, "test"
        )

        # Shuffle training once and preserve order across warm-start fits.
        order = rng.permutation(len(y_train))
        X_train = X_train[order]
        y_train = y_train[order]

        print(
            f"\n{cls}: train={len(y_train):,}, "
            f"internal validation={len(y_val):,}, "
            f"final test={len(y_test):,}",
            flush=True,
        )
        print(f"Common energies: {energies.tolist()}", flush=True)

        candidate = new_model(seed, warm_start=True)
        best_auc = -np.inf
        best_iterations = None

        # Only internal validation is used for model selection.
        for iterations in ITERATION_CANDIDATES:
            candidate.set_params(max_iter=iterations)
            candidate.fit(X_train, y_train)

            scores = candidate.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, scores)

            histories.append({
                "cls": cls,
                "iterations": iterations,
                "internal_validation_auc": float(auc),
            })

            print(
                f"  iterations={iterations}: "
                f"internal validation AUC={auc:.6f}",
                flush=True,
            )

            # Keep the simpler model if the scores are exactly equal.
            if auc > best_auc:
                best_auc = float(auc)
                best_iterations = iterations

        # Refit the selected configuration on the same training subset.
        # Final-test events have not been used for selection.
        model = new_model(seed, iterations=best_iterations)
        model.fit(X_train, y_train)

        final_scores = model.predict_proba(X_test)[:, 1]
        final_auc = float(roc_auc_score(y_test, final_scores))
        fpr, tpr, thresholds = roc_curve(y_test, final_scores)

        model_path = OUT / f"discriminator_{cls}.joblib"
        joblib.dump(model, model_path)

        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(
            reloaded.predict_proba(X_test)[:, 1],
            final_scores,
            rtol=0,
            atol=0,
        )

        np.savez(
            OUT / f"evaluation_{cls}.npz",
            domain_label=y_test,
            score_tb=final_scores,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
        )

        summaries.append({
            "cls": cls,
            "n_train_MC": len(selected["mc_train"]),
            "n_train_TB": len(selected["tb_train"]),
            "n_internal_val_MC": len(selected["mc_val"]),
            "n_internal_val_TB": len(selected["tb_val"]),
            "n_test_MC": len(selected["mc_test"]),
            "n_test_TB": len(selected["tb_test"]),
            "selected_iterations": best_iterations,
            "internal_validation_auc": best_auc,
            "heldout_auc": final_auc,
        })

        ax = axes[class_number]
        ax.plot(
            fpr, tpr,
            label=f"Held-out AUC = {final_auc:.4f}",
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        ax.set_title({
            "e": "Electrons",
            "p": "Protons",
            "C": "Carbon",
        }[cls])
        ax.set_xlabel("False positive rate (MC classified as TB)")
        ax.set_ylabel("True positive rate (TB)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")

        print(
            f"{cls}: FINAL HELD-OUT AUC = {final_auc:.6f}",
            flush=True,
        )

        del X_train, X_val, X_test, candidate, model, reloaded

    fig.suptitle(
        "Before OT: MC-vs-TB discrimination on z_enc\n"
        "Separate species; equal energy mixtures in both domains"
    )
    fig.savefig(OUT / "roc_by_class.png", dpi=200)
    fig.savefig(OUT / "roc_by_class.svg")
    plt.close(fig)

    pd.DataFrame(summaries).to_csv(
        OUT / "auc_summary.csv", index=False
    )
    pd.DataFrame(inventories).to_csv(
        OUT / "sample_counts.csv", index=False
    )
    pd.DataFrame(histories).to_csv(
        OUT / "training_history.csv", index=False
    )

    config = {
        "seed": SEED,
        "checkpoint_sha256": mc["checkpoint"],
        "input": "64-dimensional z_enc; no PCA reduction",
        "domain_labels": {"MC": 0, "TB": 1},
        "model": "HistGradientBoostingClassifier",
        "iteration_candidates": ITERATION_CANDIDATES,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "energy_sampling": (
            "Equal count per common energy and domain within each split."
        ),
        "TB_usage": (
            "Derivation for training/internal validation; "
            "validation for final test only."
        ),
        "limitations": [
            "AUC depends on discriminator capacity and available statistics.",
            "AUC near 0.5 is not a proof of distributional equality.",
            "Energy balancing does not also balance bias or run conditions.",
            "Random event splits do not test generalisation to unseen runs.",
            "Final AUC is a point estimate; uncertainty is not computed here.",
        ],
    }

    (OUT / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nFinal results:", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSaved to: {OUT}", flush=True)


if __name__ == "__main__":
    main()