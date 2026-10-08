"""Fit per-class/per-energy 1D quantile maps; compare fixed-head PID efficiencies."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformer"))
from predict import load_model
from train_ot import sha256_file
from quantile_model import QuantileMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results/quantile_1d_energy")
    parser.add_argument("--knots", type=int, default=1001)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: choose a new output directory.")
    if args.knots < 3:
        raise ValueError("At least three quantile knots required.")
    start_time = time.perf_counter()
    ckpt = HERE.parent / "transformer/checkpoints/pid_transformer/best.pt"
    mask_path = HERE / "derived/tb_is_derivation.npy"
    mc_path, tb_path = HERE / "derived/latents_MC.npz", HERE / "derived/latents_TB.npz"
    fingerprints = {p.name: sha256_file(p) for p in [ckpt, mask_path, mc_path, tb_path]}
    model, classes, _, _, _ = load_model(ckpt, args.device)
    classes = list(classes)
    model.eval()
    model.requires_grad_(False)
    domains = {}
    for domain, path in [("MC", mc_path), ("TB", tb_path)]:
        with np.load(path, allow_pickle=False) as data:
            if data["classes"].tolist() != classes or str(data["checkpoint_sha256"].item()) != fingerprints[ckpt.name]:
                raise ValueError("Latent provenance mismatch.")
            keys = ["z_enc", "logits", "label", "energy_mev", "parquet_row"]
            if domain == "TB":
                keys.append("is_derivation")
            domains[domain] = {key: data[key] for key in keys}
        d = domains[domain]
        np.testing.assert_array_equal(d["parquet_row"], np.arange(len(d["label"])))
        if d["z_enc"].shape != (len(d["label"]), 64) or d["logits"].shape != (len(d["label"]), len(classes)):
            raise ValueError("Invalid latent/logit dimensions.")
        if not np.isfinite(d["z_enc"]).all() or not np.isfinite(d["logits"]).all():
            raise ValueError("Non-finite data.")
    mc, tb = domains["MC"], domains["TB"]
    split = np.load(mask_path, allow_pickle=False)
    if split.dtype != np.bool_ or split.shape != tb["label"].shape:
        raise ValueError("Invalid split.")
    np.testing.assert_array_equal(split, tb["is_derivation"])
    # Same head/normalisation as stored latents. No classifier retraining.
    check_rows = np.linspace(0, len(mc["label"]) - 1, min(512, len(mc["label"])), dtype=int)
    with torch.no_grad():
        checked = model.head_from_latent(torch.from_numpy(mc["z_enc"][check_rows]).to(args.device)).cpu().numpy()
    np.testing.assert_allclose(checked, mc["logits"][check_rows], rtol=1e-4, atol=1e-5)
    print("PASS: frozen head matches raw logits.", flush=True)
    nominal_root = HERE / "results/task6_nominal_mod4_efficiency"
    for cls in ["e", "p", "C"]:
        if not (nominal_root / f"calibrated_MC_{cls}.npz").is_file():
            raise FileNotFoundError(nominal_root / f"calibrated_MC_{cls}.npz")
    args.out.mkdir(parents=True)
    records, details = [], []
    for cls in ["e", "p", "C"]:
        idx = classes.index(cls)
        common = np.intersect1d(np.unique(mc["energy_mev"][mc["label"] == idx]),
                                np.unique(tb["energy_mev"][(tb["label"] == idx) & split]))
        nominal_dir = HERE / f"results/task6_spark_{cls}_mod4"
        config = json.loads((nominal_dir / "config.json").read_text())
        np.testing.assert_array_equal(common, config["common_energies"])
        if config["checkpoint_sha256"] != fingerprints[ckpt.name] or config["tb_split_sha256"] != fingerprints[mask_path.name]:
            raise ValueError("Nominal provenance mismatch.")
        mc_rows = np.flatnonzero((mc["label"] == idx) & np.isin(mc["energy_mev"], common))
        train_rows = np.flatnonzero((tb["label"] == idx) & split & np.isin(tb["energy_mev"], common))
        val_rows = np.flatnonzero((tb["label"] == idx) & ~split & np.isin(tb["energy_mev"], common))
        if not len(mc_rows) or not len(train_rows) or not len(val_rows):
            raise ValueError("Empty sample.")
        with np.load(nominal_dir / "selection.npz", allow_pickle=False) as selection:
            np.testing.assert_array_equal(selection["mc_rows"], mc_rows)
            np.testing.assert_array_equal(selection["tb_derivation_rows"], train_rows)
        with np.load(nominal_root / f"calibrated_MC_{cls}.npz", allow_pickle=False) as previous:
            for key, expected in [("parquet_row", mc_rows), ("label", mc["label"][mc_rows]),
                                   ("energy_mev", mc["energy_mev"][mc_rows])]:
                np.testing.assert_array_equal(previous[key], expected)
            if previous["classes"].tolist() != classes or str(previous["checkpoint_sha256"].item()) != fingerprints[ckpt.name]:
                raise ValueError("Nominal predictions mismatch.")
            if str(previous["map_sha256"].item()) != sha256_file(nominal_dir / "map.pt"):
                raise ValueError("Nominal predictions from a different map.")
            nominal_logits = previous["logits"]
        print(f"\n{cls}: MC={len(mc_rows):,}, TB derivation={len(train_rows):,}, validation={len(val_rows):,}", flush=True)
        source, energies = mc["z_enc"][mc_rows], mc["energy_mev"][mc_rows]
        mapping = QuantileMap.fit(source, energies, tb["z_enc"][train_rows], tb["energy_mev"][train_rows], args.knots)
        map_path = args.out / f"quantile_map_{cls}.npz"
        mapping.save(map_path)
        check = np.linspace(0, len(source)-1, min(512, len(source)), dtype=int)
        np.testing.assert_array_equal(mapping.transform(source[check], energies[check]),
                                      QuantileMap.load(map_path).transform(source[check], energies[check]))
        mapped = mapping.transform(source, energies)
        logits = np.empty((len(mc_rows), len(classes)), dtype=np.float32)
        with torch.no_grad():
            for first in range(0, len(mapped), 4096):
                last = min(first + 4096, len(mapped))
                logits[first:last] = model.head_from_latent(torch.from_numpy(mapped[first:last]).to(args.device)).cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite quantile logits.")
        np.savez(args.out / f"calibrated_MC_{cls}.npz", z_enc=mapped, logits=logits,
                 label=mc["label"][mc_rows], energy_mev=energies, parquet_row=mc_rows,
                 classes=np.asarray(classes), checkpoint_sha256=np.asarray(fingerprints[ckpt.name]),
                 map_sha256=np.asarray(sha256_file(map_path)), method=np.asarray("quantile_1d_per_class_per_energy"))
        np.savez_compressed(args.out / f"selection_{cls}.npz", mc_rows=mc_rows,
                            tb_derivation_rows=train_rows, tb_validation_rows=val_rows, common_energies=common)
        for energy in [None] + list(common):
            im = np.ones(len(mc_rows), dtype=bool) if energy is None else energies == energy
            it = np.ones(len(val_rows), dtype=bool) if energy is None else tb["energy_mev"][val_rows] == energy
            if not im.any() or not it.any():
                raise ValueError("No evaluation events at one energy.")
            row = dict(cls=cls, energy="all" if energy is None else float(energy), n_MC=int(im.sum()), n_TB_validation=int(it.sum()))
            for name, values in [("raw_MC", mc["logits"][mc_rows][im]), ("nominal_OT", nominal_logits[im]),
                                 ("quantile_MC", logits[im]), ("TB_validation", tb["logits"][val_rows][it])]:
                row["correct_" + name] = int((values.argmax(1) == idx).sum())
                row[name] = row["correct_" + name] / len(values)
            row["quantile_minus_TB_pp"] = 100 * (row["quantile_MC"] - row["TB_validation"])
            row["OT_minus_TB_pp"] = 100 * (row["nominal_OT"] - row["TB_validation"])
            records.append(row)
        details.append(dict(cls=cls, n_MC=len(mc_rows), n_TB_derivation=len(train_rows),
                            common_energies=common.tolist(), reload_invariance=True,
                            constant_source_coordinate_energy_pairs=int(np.sum(mapping.x[:,0,:] == mapping.x[:,-1,:]))))
        pd.DataFrame(records).to_csv(args.out / "efficiency_comparison.csv", index=False)
        del mapped, logits, nominal_logits, source, mapping
    if sha256_file(mask_path) != fingerprints[mask_path.name]:
        raise RuntimeError("TB mask changed.")
    table = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), layout="constrained")
    for ax, cls, title in zip(axes, ["e", "p", "C"], ["Elettroni", "Protoni", "Carbonio"]):
        rows = table[(table.cls == cls) & (table.energy != "all")]
        for key, label, color in [("raw_MC", "MC originale", "#2166ac"), ("nominal_OT", "OT nominale mod4", "#2ca02c"),
                                  ("quantile_MC", "Quantili 1-D per energia", "#d95f02"), ("TB_validation", "TB validation", "black")]:
            n = rows["n_TB_validation"].to_numpy() if key == "TB_validation" else rows["n_MC"].to_numpy()
            p = rows[key].to_numpy()
            center = (p + .5/n) / (1 + 1/n)
            half = np.sqrt(p*(1-p)/n + .25/n**2) / (1 + 1/n)
            ax.errorbar(rows.energy.astype(float), 100*p,
                        yerr=100*np.maximum(0., np.array([p-(center-half), center+half-p])),
                        fmt="o-", color=color, capsize=2, markersize=4, label=label)
        ax.set_title(title)
        ax.set_xlabel("Energia [MeV/n]" if cls == "C" else "Energia [MeV]")
        ax.set_ylabel("Efficienza [%]")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Baseline a quantili 1-D vs OT condizionato\nWilson z=1 sugli eventi; scale verticali indipendenti")
    for ext in ["png", "pdf"]:
        fig.savefig(args.out / f"efficiency_vs_energy.{ext}", dpi=220)
    plt.close(fig)
    report = dict(method="Independent quantile matching of 64 latents per class AND beam energy",
                  knots=args.knots, quantile_method="linear", input_sha256=fingerprints,
                  tb_validation_used_for_fitting=False, classes=details,
                  tie_policy="Midpoint probability of a source grid plateau; constant source maps to target median",
                  tail_policy="Clip to mapped endpoints; no extrapolation in energy",
                  integrated_mixtures="Original MC and TB fractions; not energy matched",
                  uncertainty="Wilson intervals count events only; map uncertainty not estimated here",
                  elapsed_seconds=time.perf_counter()-start_time, completed=True)
    (args.out / "completion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table[table.energy == "all"].to_string(index=False))
    print("PASS: baseline fitted, reloaded and evaluated. Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
