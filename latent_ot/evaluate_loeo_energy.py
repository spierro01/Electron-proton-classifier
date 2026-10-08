"""Evaluate nominal and LOEO maps on the same MC events at the held-out energy.

Place this file in latent_ot. TB reference always uses validation events only.
Efficiencies and Wilson z=1 intervals describe event counting, not map uncertainty.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformer"))
from predict import load_model
from train_conditional_ot import ConditionalTransport
from train_ot import sha256_file


def wilson(correct, total):
    p = correct / total
    denominator = 1 + 1 / total
    center = (p + 0.5 / total) / denominator
    half = math.sqrt(p * (1 - p) / total + 0.25 / total**2) / denominator
    return max(0., center - half), min(1., center + half)


def same_energy(values, energy):
    return np.isclose(values, energy, rtol=0, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", choices=["e", "p", "C"], required=True)
    parser.add_argument("--energy", type=float, required=True)
    parser.add_argument("--loeo-dir", type=Path, required=True)
    parser.add_argument("--nominal-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out}: choose a new output directory.")
    if not math.isfinite(args.energy):
        raise ValueError("Non-finite energy.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu if intended.")

    ckpt = HERE.parent / "transformer/checkpoints/pid_transformer/best.pt"
    mask_path = HERE / "derived/tb_is_derivation.npy"
    ckpt_hash, mask_hash = sha256_file(ckpt), sha256_file(mask_path)
    model, classes, _, _, _ = load_model(ckpt, args.device)
    classes = list(classes)
    model.eval()
    model.requires_grad_(False)
    idx = classes.index(args.class_name)
    datasets = {}
    for domain in ["MC", "TB"]:
        with np.load(HERE / f"derived/latents_{domain}.npz", allow_pickle=False) as data:
            if data["classes"].tolist() != classes or str(data["checkpoint_sha256"].item()) != ckpt_hash:
                raise ValueError(f"{domain}: class order or PID checkpoint mismatch.")
            keys = ["label", "energy_mev", "parquet_row", "logits"]
            keys += ["z_enc"] if domain == "MC" else ["is_derivation"]
            datasets[domain] = {key: data[key] for key in keys}
        d = datasets[domain]
        np.testing.assert_array_equal(d["parquet_row"], np.arange(len(d["label"])))
        if d["logits"].shape != (len(d["label"]), len(classes)) or not np.isfinite(d["logits"]).all():
            raise ValueError(f"{domain}: invalid logits.")
    mc, tb = datasets["MC"], datasets["TB"]
    split = tb["is_derivation"]
    if split.dtype != np.bool_ or split.shape != tb["label"].shape:
        raise ValueError("Invalid TB split.")
    np.testing.assert_array_equal(split, np.load(mask_path, allow_pickle=False))
    mc_rows = np.flatnonzero((mc["label"] == idx) & same_energy(mc["energy_mev"], args.energy))
    tb_rows = np.flatnonzero((tb["label"] == idx) & ~split & same_energy(tb["energy_mev"], args.energy))
    if not len(mc_rows) or not len(tb_rows):
        raise ValueError("No MC or TB validation events at requested energy.")
    for d, rows in [(mc, mc_rows), (tb, tb_rows)]:
        if len(np.unique(d["energy_mev"][rows])) != 1:
            raise ValueError("Energy tolerance matches multiple settings.")

    configs, map_hashes = {}, {}
    for name, folder in [("nominal", args.nominal_dir), ("loeo", args.loeo_dir)]:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        verification = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
        if not verification.get("reload_invariance_passed") or verification.get("TB_validation_used") is not False:
            raise ValueError(f"{name}: missing successful validation-free training verification.")
        for key, expected in {"class_name": args.class_name, "classes": classes,
                              "checkpoint_sha256": ckpt_hash, "tb_split_sha256": mask_hash,
                              "TB_validation_used": False}.items():
            if config.get(key) != expected:
                raise ValueError(f"{name}: config mismatch for {key}.")
        common = np.asarray(config["common_energies"])
        if name == "loeo":
            if not same_energy(config.get("excluded_energy", float("nan")), args.energy):
                raise ValueError("Wrong excluded energy in LOEO config.")
            if same_energy(common, args.energy).any():
                raise ValueError("Held-out energy in LOEO training settings.")
        elif not same_energy(common, args.energy).any():
            raise ValueError("Nominal map did not include requested energy.")

        with np.load(folder / "selection.npz", allow_pickle=False) as selected:
            for domain, rows_key, energy_key in [("MC", "mc_rows", "mc_energy"),
                                                  ("TB", "tb_derivation_rows", "tb_energy")]:
                d = datasets[domain]
                expected = (d["label"] == idx) & np.isin(d["energy_mev"], common)
                if domain == "TB":
                    expected &= split
                rows = selected[rows_key]
                np.testing.assert_array_equal(rows, np.flatnonzero(expected))
                np.testing.assert_array_equal(selected[energy_key], d["energy_mev"][rows])
                if name == "loeo" and same_energy(selected[energy_key], args.energy).any():
                    raise ValueError("Held-out energy leaked into training selection.")
        # Read provenance from the map itself, not only the sidecar JSON.
        saved = torch.load(folder / "map.pt", map_location="cpu", weights_only=True)
        if saved["metadata"] != config or saved["architecture"] != config["architecture"]:
            raise ValueError(f"{name}: map/config mismatch.")
        del saved
        configs[name] = config
        map_hashes[name] = sha256_file(folder / "map.pt")
    for key in ["architecture", "batch_size", "outer_steps", "identity_steps", "g_updates",
                "f_updates", "lr", "min_lr", "betas", "weight_decay", "grad_clip", "seed"]:
        if configs["nominal"][key] != configs["loeo"][key]:
            raise ValueError(f"Nominal/LOEO training settings differ: {key}.")
    nominal_energies = np.asarray(configs["nominal"]["common_energies"])
    np.testing.assert_array_equal(np.asarray(configs["loeo"]["common_energies"]),
                                  nominal_energies[~same_energy(nominal_energies, args.energy)])
    kind = "interpolation" if nominal_energies.min() < args.energy < nominal_energies.max() else "extrapolation"
    print("PASS: map provenance, identical settings, training selections and TB split.", flush=True)

    # Check the raw head on ALL evaluation MC events, using bounded batches.
    for start in range(0, len(mc_rows), 1024):
        rows = mc_rows[start:start + 1024]
        with torch.no_grad():
            logits = model.head_from_latent(torch.from_numpy(mc["z_enc"][rows]).to(args.device)).cpu().numpy()
        np.testing.assert_allclose(logits, mc["logits"][rows], rtol=1e-4, atol=1e-5)
    print("PASS: frozen head reproduces raw logits at held-out energy.", flush=True)

    calibrated = {}
    for name, folder in [("nominal", args.nominal_dir), ("loeo", args.loeo_dir)]:
        transport = ConditionalTransport.load(folder / "map.pt", device=args.device)
        output = np.empty((len(mc_rows), len(classes)), dtype=np.float32)
        for start in range(0, len(mc_rows), 1024):
            rows = mc_rows[start:start + 1024]
            # Do not use no_grad here: OT needs the gradient of the potential.
            mapped = transport.transform(mc["z_enc"][rows], mc["energy_mev"][rows], batch_size=256)
            with torch.no_grad():
                logits = model.head_from_latent(torch.from_numpy(mapped).to(args.device)).cpu().numpy()
            if not np.isfinite(logits).all():
                raise ValueError("Non-finite calibrated logits.")
            output[start:start + len(rows)] = logits
        calibrated[name] = output
        del transport
        print(f"Evaluated {name}: {len(mc_rows):,} MC events.", flush=True)

    records = []
    for name, logits in [("raw_MC", mc["logits"][mc_rows]), ("nominal_MC", calibrated["nominal"]),
                         ("LOEO_MC", calibrated["loeo"]), ("TB_validation", tb["logits"][tb_rows])]:
        n = len(logits)
        correct = int((logits.argmax(1) == idx).sum())
        low, high = wilson(correct, n)
        records.append(dict(cls=args.class_name, energy=args.energy, sample=name, n=n,
                            correct=correct, efficiency=correct / n, wilson_low=low, wilson_high=high))
    raw, nominal, loeo, measured = [row["efficiency"] for row in records]
    report = dict(cls=args.class_name, energy=args.energy, kind=kind,
                  loeo_minus_nominal_pp=100 * (loeo - nominal),
                  abs_loeo_minus_nominal_pp=100 * abs(loeo - nominal),
                  loeo_minus_TB_pp=100 * (loeo - measured),
                  abs_loeo_minus_TB_pp=100 * abs(loeo - measured),
                  abs_nominal_minus_TB_pp=100 * abs(nominal - measured),
                  abs_raw_minus_TB_pp=100 * abs(raw - measured),
                  nominal_dir=str(args.nominal_dir.resolve()), loeo_dir=str(args.loeo_dir.resolve()),
                  map_sha256=map_hashes, checkpoint_sha256=ckpt_hash, tb_split_sha256=mask_hash,
                  note="Wilson z=1: event-count uncertainty only. LOEO shift also includes training variability.")
    if sha256_file(mask_path) != mask_hash:
        raise RuntimeError("TB mask changed during evaluation.")
    args.out.mkdir(parents=True)
    with (args.out / "efficiency_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "predictions.npz", mc_rows=mc_rows, tb_validation_rows=tb_rows,
                        classes=np.asarray(classes), label_mc=mc["label"][mc_rows],
                        energy_mc=mc["energy_mev"][mc_rows], logits_raw=mc["logits"][mc_rows],
                        logits_nominal=calibrated["nominal"], logits_loeo=calibrated["loeo"],
                        logits_TB_validation=tb["logits"][tb_rows])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for position, (record, color) in enumerate(zip(records, ["#2166ac", "#2ca02c", "#d62728", "black"])):
        efficiency = 100 * record["efficiency"]
        ax.errorbar(position, efficiency,
                    yerr=[[max(0., efficiency - 100 * record["wilson_low"])],
                          [max(0., 100 * record["wilson_high"] - efficiency)]],
                    fmt="o", color=color, capsize=5)
    ax.set_xticks(range(4), ["MC originale", "OT nominale", "OT LOEO", "TB validation"])
    ax.set_ylabel("Efficienza [%]")
    ax.set_title(f"{args.class_name}: energia esclusa {args.energy:g}\nLOEO ({kind}); intervalli di Wilson z=1")
    ax.grid(axis="y", alpha=.25)
    for ext in ["png", "pdf"]:
        fig.savefig(args.out / f"efficiency_comparison.{ext}", dpi=200)
    plt.close(fig)
    for row in records:
        print(f"{row['sample']:16s}: {100 * row['efficiency']:.4f}% ({row['correct']}/{row['n']})")
    print(f"LOEO - nominale: {report['loeo_minus_nominal_pp']:+.4f} punti percentuali")
    print(f"LOEO - TB:       {report['loeo_minus_TB_pp']:+.4f} punti percentuali")
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
