"""Task 5: PID efficiency versus energy, before and after OT."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "results/task5_energy"


def load_original(path, is_tb=False):
    with np.load(path, allow_pickle=False) as data:
        result = {
            "logits": data["logits"],
            "label": data["label"],
            "energy": data["energy_mev"],
            "row": data["parquet_row"],
            "classes": data["classes"].tolist(),
            "checkpoint": str(data["checkpoint_sha256"].item()),
        }
        if is_tb:
            result["is_derivation"] = data["is_derivation"]

    n = len(result["label"])
    if not np.array_equal(result["row"], np.arange(n)):
        raise ValueError("Ordine degli eventi inatteso.")
    if result["energy"].shape != (n,):
        raise ValueError("Forma delle energie non valida.")
    if result["logits"].shape != (n, len(result["classes"])):
        raise ValueError("Forma dei logits non valida.")
    if not np.isfinite(result["logits"]).all():
        raise ValueError("Logits non finiti.")
    if not np.isfinite(result["energy"]).all():
        raise ValueError("Energie non finite.")

    return result


def efficiency(correct, n):
    """Efficiency and Wilson interval with z=1 (approximately 68%)."""
    if n <= 0:
        raise ValueError("Campione vuoto.")

    p = correct / n
    z = 1.0
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half = (
        z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        / denominator
    )
    return p, center - half, center + half


def main():
    if OUT.exists():
        raise FileExistsError(
            f"{OUT} esiste già. Conserva i risultati prima di rieseguire."
        )

    mc = load_original(HERE / "derived/latents_MC.npz")
    tb = load_original(HERE / "derived/latents_TB.npz", is_tb=True)

    if mc["classes"] != tb["classes"]:
        raise ValueError("Classi MC e TB differenti.")
    if mc["checkpoint"] != tb["checkpoint"]:
        raise ValueError("Checkpoint MC e TB differenti.")

    mask = np.load(
        HERE / "derived/tb_is_derivation.npy", allow_pickle=False
    )
    if mask.dtype != np.bool_ or mask.shape != tb["label"].shape:
        raise ValueError("Maschera TB non valida.")
    np.testing.assert_array_equal(mask, tb["is_derivation"])

    # Reproduce Task-4 integrated counts as an alignment check.
    task4 = pd.read_csv(
        HERE / "results/task4_efficiency/efficiency_comparison.csv"
    ).set_index("cls")

    records = []

    for cls in ["e", "p", "C"]:
        idx = mc["classes"].index(cls)
        path = (
            HERE / "results/task4_efficiency"
            / f"calibrated_MC_{cls}.npz"
        )

        with np.load(path, allow_pickle=False) as data:
            if data["classes"].tolist() != mc["classes"]:
                raise ValueError("Classi del MC calibrato differenti.")
            if str(data["checkpoint_sha256"].item()) != mc["checkpoint"]:
                raise ValueError("Checkpoint del MC calibrato differente.")

            rows = data["parquet_row"]
            cal_logits = data["logits"]
            cal_label = data["label"]
            cal_energy = data["energy_mev"]

        if (
            np.any(rows < 0)
            or np.any(rows >= len(mc["label"]))
            or len(np.unique(rows)) != len(rows)
        ):
            raise ValueError("Indici MC calibrati non validi.")

        np.testing.assert_array_equal(cal_label, mc["label"][rows])
        np.testing.assert_array_equal(cal_energy, mc["energy"][rows])

        if not np.all(cal_label == idx):
            raise ValueError("Specie inattesa nel file calibrato.")
        if cal_logits.shape != (len(rows), len(mc["classes"])):
            raise ValueError("Forma dei logits calibrati non valida.")
        if not np.isfinite(cal_logits).all():
            raise ValueError("Logits calibrati non finiti.")

        energies = np.unique(cal_energy)

        # Every original MC event at the selected energies must be present.
        expected_rows = np.flatnonzero(
            (mc["label"] == idx) & np.isin(mc["energy"], energies)
        )
        np.testing.assert_array_equal(
            np.sort(rows), expected_rows
        )

        raw_pred = mc["logits"][rows].argmax(axis=1)
        cal_pred = cal_logits.argmax(axis=1)

        class_records = []
        for energy in energies:
            m_mc = cal_energy == energy
            m_tb = (
                (tb["label"] == idx)
                & ~mask
                & (tb["energy"] == energy)
            )

            n_mc = int(m_mc.sum())
            n_tb = int(m_tb.sum())

            correct_raw = int((raw_pred[m_mc] == idx).sum())
            correct_cal = int((cal_pred[m_mc] == idx).sum())
            correct_tb = int(
                (tb["logits"][m_tb].argmax(axis=1) == idx).sum()
            )

            raw, raw_low, raw_high = efficiency(correct_raw, n_mc)
            cal, cal_low, cal_high = efficiency(correct_cal, n_mc)
            real, tb_low, tb_high = efficiency(correct_tb, n_tb)

            gap_before = 100 * abs(raw - real)
            gap_after = 100 * abs(cal - real)

            class_records.append({
                "cls": cls,
                "energy_mev": float(energy),
                "n_MC": n_mc,
                "n_TB_validation": n_tb,
                "correct_raw_MC": correct_raw,
                "correct_calibrated_MC": correct_cal,
                "correct_TB": correct_tb,
                "raw_MC": raw,
                "raw_low": raw_low,
                "raw_high": raw_high,
                "calibrated_MC": cal,
                "cal_low": cal_low,
                "cal_high": cal_high,
                "TB_validation": real,
                "TB_low": tb_low,
                "TB_high": tb_high,
                "gap_before_pp": gap_before,
                "gap_after_pp": gap_after,
                "improvement_pp": gap_before - gap_after,
            })

        # Aggregated counts must reproduce the preceding Task-4 evaluation.
        for key in [
            "n_MC", "n_TB_validation", "correct_raw_MC",
            "correct_calibrated_MC", "correct_TB",
        ]:
            total = sum(record[key] for record in class_records)
            if total != int(task4.loc[cls, key]):
                raise ValueError(
                    f"{cls}: {key} non riproduce il Task 4."
                )

        records.extend(class_records)
        print(f"PASS {cls}: conteggi integrati coerenti con Task 4.")

    df = pd.DataFrame(records)
    OUT.mkdir(parents=True)
    df.to_csv(OUT / "efficiency_by_energy.csv", index=False)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), layout="constrained"
    )

    curves = [
        ("raw_MC", "raw_low", "raw_high", "MC originale", "tab:blue"),
        (
            "calibrated_MC", "cal_low", "cal_high",
            "MC calibrato", "tab:green"
        ),
        (
            "TB_validation", "TB_low", "TB_high",
            "TB validation", "tab:orange"
        ),
    ]

    for ax, cls, title in zip(
        axes, ["e", "p", "C"], ["Elettroni", "Protoni", "Carbonio"]
    ):
        part = df[df["cls"] == cls].sort_values("energy_mev")
        x = part["energy_mev"].to_numpy()

        for value, low, high, label, color in curves:
            y = 100 * part[value].to_numpy()
            lower = 100 * part[low].to_numpy()
            upper = 100 * part[high].to_numpy()

            ax.errorbar(
                x, y,
                yerr=np.maximum(
                    0, np.vstack([y - lower, upper - y])
                ),
                marker="o",
                markersize=4,
                capsize=3,
                linewidth=1.3,
                color=color,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlabel("Energia del fascio [MeV]")
        ax.set_ylabel("Efficienza [%]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.suptitle(
        "Efficienza per energia prima e dopo OT\n"
        "Intervalli di Wilson con z=1; scale verticali indipendenti"
    )

    for extension in ["png", "pdf"]:
        fig.savefig(
            OUT / f"efficiency_vs_energy.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    columns = [
        "cls", "energy_mev", "raw_MC", "calibrated_MC",
        "TB_validation", "gap_before_pp", "gap_after_pp",
        "improvement_pp",
    ]
    print("\nConfronto per energia:")
    print(df[columns].to_string(index=False))
    print(f"\nSalvato in: {OUT}")


if __name__ == "__main__":
    main()