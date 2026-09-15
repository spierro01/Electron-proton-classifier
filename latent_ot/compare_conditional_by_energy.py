"""Task 6: compare conditional and unconditional OT by beam energy."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_ot_by_energy import efficiency


HERE = Path(__file__).resolve().parent
OUT = HERE / "results/task6_energy"


def main():
    if OUT.exists():
        raise FileExistsError(
            f"{OUT} esiste già. Non sovrascrivo i risultati."
        )

    previous = pd.read_csv(
        HERE / "results/task5_energy/efficiency_by_energy.csv"
    )
    integrated = pd.read_csv(
        HERE / "results/task6_efficiency/efficiency_comparison.csv"
    ).set_index("cls")

    records = []

    for cls in ["e", "p", "C"]:
        new_path = (
            HERE / "results/task6_efficiency"
            / f"calibrated_MC_{cls}.npz"
        )
        old_path = (
            HERE / "results/task4_efficiency"
            / f"calibrated_MC_{cls}.npz"
        )

        with np.load(new_path, allow_pickle=False) as data:
            classes = data["classes"].tolist()
            labels = data["label"]
            energies = data["energy_mev"]
            rows = data["parquet_row"]
            logits = data["logits"]
            checkpoint = str(data["checkpoint_sha256"].item())

        # The two calibrations must refer to exactly the same MC events.
        with np.load(old_path, allow_pickle=False) as old:
            if old["classes"].tolist() != classes:
                raise ValueError("Ordine delle classi differente.")
            if str(old["checkpoint_sha256"].item()) != checkpoint:
                raise ValueError("Checkpoint PID differente.")

            np.testing.assert_array_equal(rows, old["parquet_row"])
            np.testing.assert_array_equal(labels, old["label"])
            np.testing.assert_array_equal(energies, old["energy_mev"])

        idx = classes.index(cls)
        if not np.all(labels == idx):
            raise ValueError("Specie inattesa nel file calibrato.")
        if logits.shape != (len(labels), len(classes)):
            raise ValueError("Forma dei logits non valida.")
        if not np.isfinite(logits).all():
            raise ValueError("Logits non finiti.")

        pred = logits.argmax(axis=1)

        part = previous[previous["cls"] == cls].sort_values("energy_mev")
        if part["energy_mev"].duplicated().any():
            raise ValueError("Energie duplicate nella tabella precedente.")

        np.testing.assert_array_equal(
            np.unique(energies),
            part["energy_mev"].to_numpy(),
        )

        class_records = []

        for _, old_row in part.iterrows():
            energy = float(old_row["energy_mev"])
            selected = energies == energy
            n = int(selected.sum())

            if n != int(old_row["n_MC"]):
                raise ValueError(f"Conteggio MC diverso per {cls}, {energy}.")

            correct = int((pred[selected] == idx).sum())
            value, low, high = efficiency(correct, n)

            tb = float(old_row["TB_validation"])
            old_residual = float(old_row["gap_after_pp"])
            new_residual = 100 * abs(value - tb)

            # Preserve Task-5 columns and append the new results.
            record = old_row.to_dict()
            record.update({
                "correct_conditional_MC": correct,
                "conditional_MC": value,
                "conditional_low": low,
                "conditional_high": high,
                "conditional_gap_pp": new_residual,
                "improvement_vs_unconditional_pp": (
                    old_residual - new_residual
                ),
            })
            class_records.append(record)

        total_correct = sum(
            r["correct_conditional_MC"] for r in class_records
        )
        total_events = sum(int(r["n_MC"]) for r in class_records)

        if total_correct != int(integrated.loc[cls, "correct_conditional_MC"]):
            raise ValueError("Conteggi corretti diversi dal risultato integrato.")
        if total_events != int(integrated.loc[cls, "n_MC"]):
            raise ValueError("Numero di eventi diverso dal risultato integrato.")

        for key in ["n_TB_validation", "correct_TB", "correct_raw_MC"]:
            total = sum(int(r[key]) for r in class_records)
            if total != int(integrated.loc[cls, key]):
                raise ValueError(f"{cls}: {key} diverso dal risultato integrato.")

        records.extend(class_records)
        print(f"PASS {cls}: allineamento e conteggi integrati coerenti.")

    df = pd.DataFrame(records)
    OUT.mkdir(parents=True)
    df.to_csv(OUT / "efficiency_by_energy.csv", index=False)

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5), layout="constrained"
    )

    curves = [
        ("raw_MC", "raw_low", "raw_high",
         "MC originale", "tab:blue", "o"),
        ("calibrated_MC", "cal_low", "cal_high",
         "OT senza energia", "tab:green", "s"),
        ("conditional_MC", "conditional_low", "conditional_high",
         "OT con energia", "tab:red", "^"),
        ("TB_validation", "TB_low", "TB_high",
         "TB validation", "black", "D"),
    ]

    for ax, cls, title in zip(
        axes, ["e", "p", "C"], ["Elettroni", "Protoni", "Carbonio"]
    ):
        part = df[df["cls"] == cls].sort_values("energy_mev")
        x = part["energy_mev"].to_numpy()

        for value, low, high, label, color, marker in curves:
            y = 100 * part[value].to_numpy()
            lower = 100 * part[low].to_numpy()
            upper = 100 * part[high].to_numpy()

            ax.errorbar(
                x, y,
                yerr=np.maximum(
                    0, np.vstack([y - lower, upper - y])
                ),
                label=label,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.2,
                capsize=3,
            )

        ax.set_title(title)
        ax.set_xlabel("Energia del fascio [MeV]")
        ax.set_ylabel("Efficienza [%]")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Confronto OT senza e con condizionamento energetico\n"
        "Intervalli di Wilson con z=1; scale verticali indipendenti"
    )

    for extension in ["png", "pdf"]:
        fig.savefig(
            OUT / f"efficiency_vs_energy.{extension}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)

    columns = [
        "cls", "energy_mev", "raw_MC", "calibrated_MC",
        "conditional_MC", "TB_validation",
        "gap_after_pp", "conditional_gap_pp",
        "improvement_vs_unconditional_pp",
    ]

    print("\nConfronto per energia:")
    print(df[columns].to_string(index=False))
    print(f"\nSalvato in: {OUT}")


if __name__ == "__main__":
    main()