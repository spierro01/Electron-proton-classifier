"""Disegna le curve di addestramento OT dal log salvato."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    log_path = args.run / "training_log.csv"
    df = pd.read_csv(log_path)

    columns = [
        ("loss_g", "Loss di g", "Loss"),
        ("loss_f", "Loss di f", "Loss"),
        (
            "mean_l2_standardised",
            "Spostamento medio nello spazio standardizzato",
            r"$\langle\|T(z)-z\|_2\rangle$",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (column, title, ylabel) in zip(axes, columns):
        ax.plot(df["outer_step"], df[column], marker="o")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Passo esterno OT")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Addestramento OT — {args.run.name}")
    fig.tight_layout()

    for extension in ("png", "pdf"):
        output = args.run / f"training_curves.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Salvato: {output.resolve()}")

    plt.close(fig)


if __name__ == "__main__":
    main()