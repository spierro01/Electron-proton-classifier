"""Compare fixed-energy MC and TB validation predictions.

Run from latent_ot. No model training or inference is performed.
CSV rows must correspond to the original, unchanged parquet files.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MC_PARQUET = DATA / "dumpMC.parquet"
TB_PARQUET = DATA / "dumpTB.parquet"

MC_CSV = ROOT / "transformer/predictions_MC_fixed.csv"
TB_CSV = ROOT / "transformer/predictions_TB_all.csv"

MASK_FILE = ROOT / "latent_ot/derived/tb_is_derivation.npy"
OUT = ROOT / "latent_ot/results/baseline_mc_tb"

CLASSES = ["e", "p", "C"]
LABEL_MAP = {
    "electron": "e", "proton": "p", "carbon": "C", "helium": "He",
    "e": "e", "p": "p", "C": "C", "He": "He",
}


def load_events(parquet_path, csv_path):
    """Attach original parquet metadata after checking row alignment."""
    available = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    required = ["particle_type", "energy_mev"]

    if not set(required).issubset(available):
        raise ValueError(f"Missing label or energy in {parquet_path}")

    optional = [
        "source_file", "angle_deg", "config", "bias_v",
        "boot_nr", "run_nr", "event_index",
    ]
    columns = required + [c for c in optional if c in available]
    meta = pd.read_parquet(parquet_path, columns=columns)

    pred = pd.read_csv(csv_path)
    if len(pred) != len(meta):
        raise ValueError(
            f"{csv_path.name}: {len(pred)} predictions for "
            f"{len(meta)} parquet rows."
        )

    if "parquet_row" in pred:
        rows = pd.to_numeric(pred["parquet_row"], errors="raise").to_numpy()
        if not np.isfinite(rows).all() or not np.equal(rows, rows.astype(int)).all():
            raise ValueError("Invalid parquet_row identifiers.")

        rows = rows.astype(np.int64)
        if not np.array_equal(np.sort(rows), np.arange(len(meta))):
            raise ValueError("Missing, duplicate or invalid parquet rows.")

        pred = pred.assign(parquet_row=rows).sort_values(
            "parquet_row"
        ).reset_index(drop=True)
    else:
        print(
            f"{csv_path.name}: assuming original row order "
            "(full inference without --limit)."
        )

    labels = meta["particle_type"].map(LABEL_MAP)
    if labels.isna().any():
        raise ValueError(f"Unknown particle labels in {parquet_path}")

    saved_label = next(
        (c for c in ["label_canonical", "truth", "particle_type"] if c in pred),
        None,
    )
    if saved_label is None:
        raise ValueError(f"No reference labels in {csv_path}")

    csv_labels = pred[saved_label].map(LABEL_MAP)
    if not np.array_equal(labels.to_numpy(), csv_labels.to_numpy()):
        raise ValueError(f"Label alignment failed for {csv_path}")

    if not pred["pred"].isin(CLASSES).all():
        raise ValueError(f"Unexpected predictions in {csv_path}")

    meta = meta.copy()
    meta["cls"] = labels
    meta["pred"] = pred["pred"].to_numpy()
    meta["energy_mev"] = pd.to_numeric(meta["energy_mev"], errors="raise")

    if not np.isfinite(meta["energy_mev"]).all():
        raise ValueError("Missing or invalid energies.")

    return meta


def wilson(k, n):
    """Wilson binomial interval with z=1 (approximately 68% coverage)."""
    p = k / n
    denominator = 1 + 1 / n
    center = (p + 1 / (2 * n)) / denominator
    radius = np.sqrt(p * (1 - p) / n + 1 / (4 * n**2)) / denominator
    return center - radius, center + radius


def efficiency_table(df, domain):
    rows = []
    for (cls, energy), group in df.groupby(["cls", "energy_mev"]):
        n = len(group)
        k = int((group["pred"] == cls).sum())
        low, high = wilson(k, n)
        rows.append({
            "cls": cls,
            "energy_mev": energy,
            "domain": domain,
            "n": n,
            "correct": k,
            "efficiency": k / n,
            "low": low,
            "high": high,
        })
    return pd.DataFrame(rows)


def confusion(df):
    return pd.crosstab(df["cls"], df["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    ).to_numpy()


def main():
    for path in [MC_PARQUET, TB_PARQUET, MC_CSV, TB_CSV, MASK_FILE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)

    mc = load_events(MC_PARQUET, MC_CSV)
    tb = load_events(TB_PARQUET, TB_CSV)

    mask = np.load(MASK_FILE, allow_pickle=False)
    if mask.dtype != np.bool_ or mask.shape != (len(tb),):
        raise ValueError("TB split must be a boolean mask matching the parquet.")

    print(f"TB derivation excluded: {int(mask.sum()):,}")
    tb = tb.loc[~mask].copy()
    print(f"TB validation: {len(tb):,}")

    mc = mc[mc["cls"].isin(CLASSES)].copy()
    tb = tb[tb["cls"].isin(CLASSES)].copy()

    # Inventory BEFORE restricting to common energies.
    inventory_mc = mc.groupby(["cls", "energy_mev"]).size().rename("n_MC")
    inventory_tb = tb.groupby(["cls", "energy_mev"]).size().rename("n_TB_validation")
    inventory = pd.concat([inventory_mc, inventory_tb], axis=1).fillna(0)
    inventory.to_csv(OUT / "inventory_validation.csv")

    keys_mc = set(zip(mc["cls"], mc["energy_mev"]))
    keys_tb = set(zip(tb["cls"], tb["energy_mev"]))
    common = keys_mc & keys_tb

    excluded = sorted(keys_mc ^ keys_tb)
    print("Class-energy combinations without a counterpart:", excluded)

    mc = mc.loc[
        [(c, e) in common for c, e in zip(mc["cls"], mc["energy_mev"])]
    ].copy()
    tb = tb.loc[
        [(c, e) in common for c, e in zip(tb["cls"], tb["energy_mev"])]
    ].copy()

    # Save inventories of available experimental conditions.
    # Matching energy alone does NOT establish full physical compatibility.
    for domain, df in [("MC", mc), ("TB_validation", tb)]:
        conditions = ["cls", "energy_mev"] + [
            c for c in ["angle_deg", "config", "bias_v"] if c in df
        ]
        df.groupby(conditions, dropna=False).size().rename("n").reset_index().to_csv(
            OUT / f"conditions_{domain}.csv", index=False
        )

        if "source_file" in df:
            run_columns = ["source_file", "cls", "energy_mev"] + [
                c for c in ["angle_deg", "config", "bias_v"] if c in df
            ]
            run_summary = df.assign(correct=df["pred"] == df["cls"])
            run_summary.groupby(run_columns, dropna=False).agg(
                n=("correct", "size"),
                correct=("correct", "sum"),
                efficiency=("correct", "mean"),
            ).reset_index().to_csv(
                OUT / f"efficiency_by_file_{domain}.csv", index=False
            )

    table_mc = efficiency_table(mc, "MC")
    table_tb = efficiency_table(tb, "TB validation")
    long_table = pd.concat([table_mc, table_tb], ignore_index=True)
    long_table.to_csv(OUT / "efficiency_long.csv", index=False)

    comparison = table_mc.drop(columns="domain").merge(
        table_tb.drop(columns="domain"),
        on=["cls", "energy_mev"],
        suffixes=("_MC", "_TB"),
        validate="one_to_one",
    )
    comparison["delta_percentage_points"] = 100 * (
        comparison["efficiency_MC"] - comparison["efficiency_TB"]
    )
    comparison.to_csv(OUT / "comparison_by_energy.csv", index=False)

    # Efficiency curves: one panel per class.
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, cls in zip(axes, CLASSES):
        for table, label, color, marker in [
            (table_mc, "MC", "#2166ac", "o"),
            (table_tb, "TB validation", "#d6604d", "s"),
        ]:
            part = table[table["cls"] == cls].sort_values("energy_mev")
            y = part["efficiency"].to_numpy()
            lower = np.maximum(0, y - part["low"].to_numpy())
            upper = np.maximum(0, part["high"].to_numpy() - y)

            ax.errorbar(
                part["energy_mev"], y,
                yerr=np.vstack([lower, upper]),
                marker=marker, color=color, label=label,
                capsize=3, linewidth=1.2,
            )

        ax.set_title({"e": "Electrons", "p": "Protons", "C": "Carbon"}[cls])
        ax.set_xlabel("energy_mev (file value)")
        ax.set_ylim(0, 1.025)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("PID efficiency")
    fig.suptitle("Before OT: common energies; conditions still to be verified")
    fig.tight_layout()
    fig.savefig(OUT / "efficiency_vs_energy.png", dpi=200)
    fig.savefig(OUT / "efficiency_vs_energy.svg")
    plt.close(fig)

    # Row-normalised confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, df, name in zip(axes, [mc, tb], ["MC", "TB validation"]):
        matrix = confusion(df)
        row_totals = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(
            matrix, row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        im = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, f"{matrix[i, j]:,}\n{100*fractions[i, j]:.2f}%",
                    ha="center", va="center",
                    color="white" if fractions[i, j] > 0.5 else "black",
                )
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("File label")
        ax.set_title(name)
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
            OUT / f"confusion_{name.replace(' ', '_')}.csv"
        )

    fig.colorbar(im, ax=axes, label="Fraction within file-label class")
    fig.savefig(OUT / "confusion_matrices.png", dpi=200)
    fig.savefig(OUT / "confusion_matrices.svg")
    plt.close(fig)

    notes = {
        "mc_events_selected": len(mc),
        "tb_validation_events_selected": len(tb),
        "excluded_class_energy_combinations": excluded,
        "interval": "Wilson, z=1, approximately 68% binomial coverage",
        "limitations": [
            "Energy matching is exact on stored numeric values.",
            "Angles, configurations, selections and energy units require verification.",
            "TB labels are file labels, not guaranteed event-level truth.",
            "Confusion matrices pool energies with different domain mixtures.",
            "Intervals exclude systematic uncertainty and possible run correlations.",
            "CSV without parquet_row relies on unchanged original row order.",
        ],
    }
    (OUT / "analysis_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )

    print("\nEfficiency comparison:")
    print(comparison[
        ["cls", "energy_mev", "n_MC", "n_TB",
         "efficiency_MC", "efficiency_TB", "delta_percentage_points"]
    ].to_string(index=False))
    print(f"\nResults saved in: {OUT}")


if __name__ == "__main__":
    main()