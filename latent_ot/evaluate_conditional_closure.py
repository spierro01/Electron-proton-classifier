"""Task 6: evaluate conditional OT closure and compare all three cases."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate_ot_closure as closure


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / "results/task4_closure"
CURRENT = HERE / "results/task6_closure"


def main():
    # Verify that previous results exist before starting.
    required = [PREVIOUS / "auc_summary.csv"]

    for cls in ["e", "p", "C"]:
        for representation in ["z_enc", "logits"]:
            for status in ["raw", "calibrated"]:
                required.append(
                    PREVIOUS
                    / f"{cls}_{representation}_{status}_evaluation.npz"
                )

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Reuse the Task-4 procedure with the conditional latent files.
    # The original module and previous results remain unchanged.
    closure.CALIBRATED = HERE / "results/task6_efficiency"
    closure.OUT = CURRENT

    print(
        "Evaluating energy-conditioned OT.\n"
        "Using the same discriminator splits as Task 4.",
        flush=True,
    )
    closure.main()

    previous = pd.read_csv(PREVIOUS / "auc_summary.csv")
    current = pd.read_csv(CURRENT / "auc_summary.csv")

    def get_auc(frame, cls, representation, status):
        selected = frame[
            (frame["cls"] == cls)
            & (frame["representation"] == representation)
            & (frame["status"] == status)
        ]
        if len(selected) != 1:
            raise ValueError("Missing or duplicate AUC result.")
        return float(selected.iloc[0]["heldout_auc"])

    records = []

    for representation in ["z_enc", "logits"]:
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 5), layout="constrained"
        )

        for ax, cls in zip(axes, ["e", "p", "C"]):
            raw_previous = get_auc(previous, cls, representation, "raw")
            raw_current = get_auc(current, cls, representation, "raw")
            unconditional = get_auc(
                previous, cls, representation, "calibrated"
            )
            conditional = get_auc(
                current, cls, representation, "calibrated"
            )

            # The unchanged raw baseline should reproduce the old result.
            np.testing.assert_allclose(
                raw_current, raw_previous, rtol=0, atol=1e-8
            )

            records.append({
                "cls": cls,
                "representation": representation,
                "raw_auc": raw_current,
                "unconditional_auc": unconditional,
                "conditional_auc": conditional,
                "auc_unconditional_minus_conditional": (
                    unconditional - conditional
                ),
            })

            curves = [
                (
                    CURRENT, "raw", "MC originale",
                    "tab:blue", raw_current
                ),
                (
                    PREVIOUS, "calibrated", "OT senza energia",
                    "tab:green", unconditional
                ),
                (
                    CURRENT, "calibrated", "OT con energia",
                    "tab:red", conditional
                ),
            ]

            reference_rows = None

            for folder, status, label, color, auc in curves:
                path = (
                    folder
                    / f"{cls}_{representation}_{status}_evaluation.npz"
                )
                with np.load(path, allow_pickle=False) as data:
                    test_rows = (
                        data["mc_test_rows"],
                        data["tb_test_rows"],
                    )

                    if reference_rows is None:
                        reference_rows = test_rows
                    else:
                        for actual, expected in zip(
                            test_rows, reference_rows
                        ):
                            np.testing.assert_array_equal(actual, expected)

                    ax.plot(
                        data["fpr"], data["tpr"],
                        color=color,
                        label=f"{label}: AUC={auc:.4f}",
                    )

            ax.plot(
                [0, 1], [0, 1], "--", color="gray",
                label="Riferimento casuale",
            )
            ax.set_title({
                "e": "Elettroni",
                "p": "Protoni",
                "C": "Carbonio",
            }[cls])
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="lower right")

        fig.suptitle(
            f"Confronto MC-vs-TB su {representation}\n"
            "Stessi eventi di test e miscele energetiche bilanciate"
        )

        for extension in ["png", "pdf"]:
            fig.savefig(
                CURRENT / f"roc_three_way_{representation}.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

    comparison = pd.DataFrame(records)
    comparison.to_csv(
        CURRENT / "auc_three_way_comparison.csv", index=False
    )

    print("\nPASS: raw baseline reproduced.")
    print("PASS: identical final-test event indices.")
    print("\nConfronto delle tre configurazioni:")
    print(comparison.to_string(index=False))
    print(f"\nSaved to: {CURRENT}")


if __name__ == "__main__":
    main()