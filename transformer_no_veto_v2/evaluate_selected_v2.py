"""Selected fixed-energy MC-vs-TB baseline for the no-veto v2 classifier.

Run beside pid_data.py, pid_model.py and predict.py in transformer_no_veto_v2.
TB split is loaded on original parquet rows, never regenerated or written.
Integrated comparisons use common selected energies, with each domain's own
energy mixture. CSV efficiencies are fractions; differences are percentage points.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TR1 = [f"RAW_TR1_HG_{i}_{s}" for i in range(5) for s in range(2)]
TR2 = [f"RAW_TR2_HG_{i}_{s}" for i in range(4) for s in range(2)]
VETO = [f"RAW_VETO_HG_{i}_{s}" for i in range(5) for s in range(2)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selection_stages(df, labels, domain, classes):
    energy = df["energy_mev"].to_numpy(dtype=float)
    if not np.isfinite(energy).all():
        raise ValueError("Nonfinite beam energies.")
    veto = df[VETO].to_numpy(dtype=float)
    tr1 = df[TR1].to_numpy(dtype=float)
    tr2 = df[TR2].to_numpy(dtype=float)
    theta = df["theta"].to_numpy(dtype=float)
    boot = "file_boot_nr" if domain == "MC" else "boot_nr"
    run = "file_run_nr" if domain == "MC" else "run_nr"
    p70 = (labels == "p") & np.isclose(energy, 70, rtol=0, atol=1e-4)
    chosen_run = (df[boot].to_numpy() == 94) & (df[run].to_numpy() == 11)
    return [
        ("classes_e_p_C", np.isin(labels, classes)),
        ("containment_all_veto_HG_lt_50", np.isfinite(veto).all(1) & (veto < 50).all(1)),
        ("TR1_max_HG_gt_50", np.isfinite(tr1).all(1) & (tr1.max(1) > 50)),
        ("TR2_max_HG_gt_50", np.isfinite(tr2).all(1) & (tr2.max(1) > 50)),
        ("theta_lt_80_deg", np.isfinite(theta) & (theta < 80)),
        ("p70_boot94_run11", ~p70 | chosen_run),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mc", type=Path, default=Path("../data/dumpMC_v2.parquet"))
    ap.add_argument("--tb", type=Path, default=Path("../data/dumpTB_v2.parquet"))
    ap.add_argument("--mask", type=Path, default=Path("../latent_ot/derived_v2/tb_is_derivation.npy"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()
    import torch
    import pid_data as PD
    from predict import load_model, predict_matrix

    if args.out.exists():
        raise FileExistsError(f"{args.out}: use a new output directory.")
    if args.batch_size < 1:
        raise ValueError("Invalid batch size.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    expected = ["TR1", "TR2"] + [f"RAN{i}" for i in range(12)] + ["EN1", "EN2"]
    assert [name for name, _ in PD.LAYERS] == expected
    assert (PD.N_STEPS, PD.N_FEATURES) == (16, 8)
    assert not any("VETO" in c or c.startswith(("BOT_", "LAT_")) for c in PD.required_columns())

    files = {"MC": args.mc, "TB": args.tb}
    parquets = {d: pq.ParquetFile(p) for d, p in files.items()}
    mask = np.load(args.mask, allow_pickle=False)
    assert mask.dtype == np.bool_ and mask.shape == (1_485_889,)
    assert len(mask) == parquets["TB"].metadata.num_rows
    assert int(mask.sum()) == 743_135 and int((~mask).sum()) == 742_754
    mask_hash = sha256(args.mask)

    columns = {}
    for domain, parquet in parquets.items():
        boot, run = (("file_boot_nr", "file_run_nr") if domain == "MC" else ("boot_nr", "run_nr"))
        columns[domain] = list(dict.fromkeys(PD.required_columns() + TR1 + TR2 + VETO
                                          + ["theta", "particle_type", "energy_mev", boot, run]))
        missing = set(columns[domain]) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{domain}: missing columns {sorted(missing)}")

    model, classes, mean, std, checkpoint = load_model(args.ckpt, args.device)
    assert classes == ["e", "p", "C"]
    assert mean.shape == std.shape == (16, 8)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    model.requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=False)
    print("Checkpoint:", args.ckpt.resolve(), flush=True)
    print("Device:", args.device, "| Classes:", classes, flush=True)
    print("Hashing input files for provenance...", flush=True)
    provenance = {
        "checkpoint": str(args.ckpt.resolve()), "checkpoint_sha256": sha256(args.ckpt),
        "mask": str(args.mask.resolve()), "mask_sha256": mask_hash,
        "source_files": {d: {"path": str(p.resolve()), "sha256": sha256(p),
                             "rows": parquets[d].metadata.num_rows} for d, p in files.items()},
        "classes": classes, "layers": expected, "TB_metrics": "validation only",
        "selection": {"containment": "all 10 RAW_VETO_HG < 50",
                      "TR1": "max over all 10 HG values > 50",
                      "TR2": "max over all 8 HG values > 50", "theta": "theta < 80 degrees",
                      "p70": "keep only boot 94 run 11; MC uses file_boot_nr/file_run_nr",
                      "invalid_values": "reject nonfinite values in cut variables"},
        "integrated_metrics": "common selected energies; original domain energy fractions",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    cutflow, inventory, confusion = Counter(), Counter(), {}
    metrics = Counter()
    for domain, parquet in parquets.items():
        offset = 0
        wrote_header = False
        confusion[domain] = Counter()
        for batch in parquet.iter_batches(batch_size=50_000, columns=columns[domain]):
            df = batch.to_pandas()
            labels_series = df["particle_type"].map(PD.LABEL_MAP)
            if labels_series.isna().any():
                raise ValueError("Unknown particle labels.")
            labels = labels_series.to_numpy()
            energies = df["energy_mev"].to_numpy(dtype=float)
            rows = np.arange(offset, offset + len(df), dtype=np.int64)
            keep = np.ones(len(df), dtype=bool)

            def record(stage):
                counts = Counter(zip(labels[keep], energies[keep]))
                for (cls, energy), count in counts.items():
                    cutflow[domain, stage, str(cls), float(energy)] += count

            record("input")
            boot = "file_boot_nr" if domain == "MC" else "boot_nr"
            run = "file_run_nr" if domain == "MC" else "run_nr"
            p70 = (labels == "p") & np.isclose(energies, 70, rtol=0, atol=1e-4)
            for (b, r), count in Counter(zip(df.loc[p70, boot], df.loc[p70, run])).items():
                inventory[domain, str(b), str(r)] += count
            for stage, cut in selection_stages(df, labels, domain, classes):
                keep &= cut
                record(stage)
            if domain == "TB":
                keep &= ~mask[rows]
            record("evaluation_sample")

            if keep.any():
                selected = df.loc[keep]
                x = PD.build_matrix(selected)
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite classifier input.")
                probabilities, pred = predict_matrix(model, x, mean, std, args.device, args.batch_size)
                if not np.isfinite(probabilities).all():
                    raise ValueError("Nonfinite probabilities.")
                truth = labels[keep]
                energy = energies[keep]
                y = np.array([classes.index(c) for c in truth])
                frame = pd.DataFrame({"parquet_row": rows[keep], "truth": truth,
                                      "energy_mev": energy, "pred": np.array(classes)[pred],
                                      "boot": selected[boot].to_numpy(), "run": selected[run].to_numpy()})
                for i, cls in enumerate(classes):
                    frame[f"p_{cls}"] = probabilities[:, i]
                if domain == "TB":
                    frame["is_derivation"] = False
                frame.to_csv(args.out / f"predictions_{domain}_selected.csv", mode="a", index=False,
                             header=not wrote_header)
                wrote_header = True
                for cls, e, actual, predicted in zip(truth, energy, y, pred):
                    metrics[domain, str(cls), float(e), "n"] += 1
                    metrics[domain, str(cls), float(e), "correct"] += int(actual == predicted)
                    confusion[domain][float(e), int(actual), int(predicted)] += 1
            offset += len(df)
            print(f"{domain}: processed {offset:,}/{parquet.metadata.num_rows:,}", flush=True)
        assert offset == parquet.metadata.num_rows

    pd.DataFrame([dict(domain=d, stage=s, cls=c, energy_mev=e, n=n)
                  for (d, s, c, e), n in cutflow.items()]).to_csv(args.out / "cutflow.csv", index=False)
    pd.DataFrame([dict(domain=d, boot=b, run=r, n=n) for (d, b, r), n in inventory.items()]).to_csv(
        args.out / "p70_run_inventory.csv", index=False)
    summaries, integrated, conf_report = [], [], {}
    for domain in files:
        conf_report[domain] = np.zeros((3, 3), dtype=np.int64)
    for cls in classes:
        i = classes.index(cls)
        energy_sets = {d: {e for (dom, c, e, status), n in metrics.items()
                          if dom == d and c == cls and status == "n" and n > 0} for d in files}
        common = energy_sets["MC"] & energy_sets["TB"]
        if not common:
            raise ValueError(f"No common selected energies for {cls}.")

        def summary(energy, values):
            row = {"cls": cls, "energy_mev": energy}
            for d in files:
                n = sum(metrics[d, cls, e, "n"] for e in values)
                correct = sum(metrics[d, cls, e, "correct"] for e in values)
                row[f"n_{d}"] = n
                row[f"correct_{d}"] = correct
                row[f"efficiency_{d}"] = correct / n if n else np.nan
            row["MC_minus_TB_pp"] = 100 * (row["efficiency_MC"] - row["efficiency_TB"])
            return row

        for e in sorted(energy_sets["MC"] | energy_sets["TB"]):
            row = summary(e, [e])
            row["common_energy"] = e in common
            summaries.append(row)
        integrated.append(summary("all_common", sorted(common)))
        for d in files:
            for e in common:
                for j in range(3):
                    conf_report[d][i, j] += confusion[d][e, i, j]
    by_energy = pd.DataFrame(summaries)
    by_energy.to_csv(args.out / "efficiency_by_energy.csv", index=False)
    integrated_df = pd.DataFrame(integrated)
    integrated_df.to_csv(args.out / "efficiency_integrated.csv", index=False)
    for domain, matrix in conf_report.items():
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(args.out / f"confusion_{domain}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, cls in zip(axes, classes):
        table = by_energy[(by_energy.cls == cls) & by_energy.common_energy].sort_values("energy_mev")
        for domain, color in [("MC", "tab:blue"), ("TB", "black")]:
            p = table[f"efficiency_{domain}"].to_numpy()
            n = table[f"n_{domain}"].to_numpy()
            center = (p + 0.5 / n) / (1 + 1 / n)
            half = np.sqrt(p * (1 - p) / n + 0.25 / n**2) / (1 + 1 / n)
            errors = np.maximum(0, np.vstack([p - center + half, center + half - p]))
            ax.errorbar(table.energy_mev, 100 * p, yerr=100 * errors, marker="o", color=color,
                        label="MC v2" if domain == "MC" else "TB v2 validation")
        ax.set(title={"e": "Elettroni", "p": "Protoni", "C": "Carbonio"}[cls],
               xlabel="Energia [MeV/n]" if cls == "C" else "Energia [MeV]", ylabel="Efficienza [%]")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Transformer senza veto, campioni v2 selezionati — Wilson z=1; scale indipendenti")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(args.out / f"efficiency_vs_energy.{ext}", dpi=180)
    plt.close(fig)
    assert sha256(args.mask) == mask_hash, "TB split changed during evaluation."
    (args.out / "completion.json").write_text(json.dumps({
        "completed": True, "TB_validation_only": True, "mask_unchanged": True,
        "classes": classes,
    }, indent=2), encoding="utf-8")
    print("\nEFFICIENZE INTEGRATE (frazioni, energie comuni):", flush=True)
    print(integrated_df.to_string(index=False), flush=True)
    print("\nPER ENERGIA:", flush=True)
    print(by_energy.to_string(index=False), flush=True)
    print("\nSaved:", args.out.resolve(), flush=True)


if __name__ == "__main__":
    main()
