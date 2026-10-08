"""Summarize between-map bootstrap dispersion on fixed evaluation events."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    frames = []
    for job in manifest:
        df = pd.read_csv(Path(job["evaluation"]) / "efficiency_by_energy.csv", dtype={"energy": str})
        assert (df["cls"] == job["cls"]).all()
        assert (df["bootstrap_seed"] == job["bootstrap_seed"]).all()
        df["replica"] = job["replica"]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["cls", "energy", "replica"]).any():
        raise ValueError("Duplicate replica measurements.")
    rows = []
    for (cls, energy), group in data.groupby(["cls", "energy"], sort=False):
        if len(group) < 2:
            raise ValueError("At least two completed replicas needed.")
        for key in ["raw_MC", "nominal_MC", "TB_validation", "n_MC", "n_TB_validation"]:
            if group[key].nunique() != 1:
                raise ValueError("Evaluation sample mismatch: " + key)
        nominal, tb = float(group["nominal_MC"].iloc[0]), float(group["TB_validation"].iloc[0])
        rows.append(dict(cls=cls, energy=energy, replicas=len(group),
                         raw_MC=float(group["raw_MC"].iloc[0]), nominal_MC=nominal,
                         TB_validation=tb, bootstrap_mean=float(group["bootstrap_MC"].mean()),
                         bootstrap_sd_pp=100 * float(group["bootstrap_MC"].std(ddof=1)),
                         nominal_minus_TB_pp=100 * (nominal - tb),
                         bootstrap_mean_minus_nominal_pp=100 * (group["bootstrap_MC"].mean() - nominal)))
    summary = pd.DataFrame(rows)
    out = args.manifest.parent / ("summary_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir()
    data.to_csv(out / "replica_efficiencies.csv", index=False)
    summary.to_csv(out / "bootstrap_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    for ax, cls, title in zip(axes, ["e", "p", "C"], ["Elettroni", "Protoni", "Carbonio"]):
        frame = summary[(summary.cls == cls) & (summary.energy != "all")].copy()
        frame["energy"] = frame.energy.astype(float)
        frame = frame.sort_values("energy")
        x = frame.energy.to_numpy()
        for column, label, color in [("raw_MC", "MC originale", "#2166ac"),
                                      ("nominal_MC", "OT nominale", "#2ca02c"),
                                      ("TB_validation", "TB validation", "black")]:
            ax.plot(x, 100 * frame[column], "o-", color=color, label=label, markersize=4)
        mean = 100 * frame.bootstrap_mean.to_numpy()
        sd = frame.bootstrap_sd_pp.to_numpy()
        ax.errorbar(x, mean, yerr=sd, fmt="s--", color="#d95f02", capsize=3,
                    label="Media bootstrap ± 1 SD", markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Energia del fascio [MeV/n]" if cls == "C" else "Energia del fascio [MeV]")
        ax.set_ylabel("Efficienza [%]")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Bootstrap TB derivation — scale verticali indipendenti\nSD tra mappe: non è l’errore sulla media né l’incertezza totale del confronto col TB")
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"bootstrap_efficiency.{ext}", dpi=220)
    plt.close(fig)
    print(summary.to_string(index=False))
    print("Summary saved:", out)


if __name__ == "__main__":
    main()
