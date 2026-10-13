"""Convert the spectra parquet into memory-mapped training tensors (N, 16, 8).

Reads the raw per-strip columns from the parquet, applies the hit-strip
aggregation rule (see pid_data.py and the README), and writes flat arrays that
train.py can memory-map — so training never needs the parquet or a big RAM
load again.

Usage:
    python preprocess.py --input /path/to/dumpMC_spectra.parquet
    python preprocess.py --input data.parquet --classes e p C He --out derived/pid

Outputs (in --out):
    X.npy        float32 (N, 16, 8)   detector matrices
    y.npy        int64   (N,)         labels, index into classes.npy
    classes.npy  class names; position in this array = label index
    meta.json    provenance
"""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import pid_data as PD

CHUNK = 200_000  # rows converted per pass; keeps peak memory bounded


def main():
    ap = argparse.ArgumentParser(
        description="Build (N, 16, 8) training tensors from the spectra parquet.")
    ap.add_argument("--input", type=Path, required=True,
                    help="path to the spectra parquet (e.g. dumpMC_spectra.parquet)")
    ap.add_argument("--out", type=Path, default=Path("derived/pid"),
                    help="output directory (default: derived/pid)")
    ap.add_argument("--classes", nargs="+", default=["e", "p", "C"],
                    help="classes to keep, canonical names e/p/C/He. "
                         "Order here defines the label index. Default: e p C "
                         "(the three particles with test-beam data; He is "
                         "simulation-only)")
    args = ap.parse_args()
    expected_layers = (
        ["TR1", "TR2"] + [f"RAN{i}" for i in range(12)] + ["EN1", "EN2"]
    )
    if [name for name, _ in PD.LAYERS] != expected_layers:
        raise ValueError("Ordine dei layer inatteso o veto ancora presente.")
    if (PD.N_STEPS, PD.N_FEATURES) != (16, 8):
        raise ValueError("Input atteso: 16 x 8.")
    if args.classes != ["e", "p", "C"]:
        raise ValueError("Per questo training usare le classi e p C.")
    if any("VETO" in c or c.startswith(("BOT_", "LAT_"))
           for c in PD.required_columns()):
        raise ValueError("Colonne veto ancora presenti negli input.")
    args.out.mkdir(parents=True, exist_ok=False)

    # only read the columns we actually need — the parquet has ~285 of them
    tr1 = [f"RAW_TR1_HG_{i}_{s}" for i in range(5) for s in range(2)]
    tr2 = [f"RAW_TR2_HG_{i}_{s}" for i in range(4) for s in range(2)]
    cols = list(dict.fromkeys(
        PD.required_columns() + ["particle_type", "theta"] + tr1 + tr2
    ))
    print(f"Loading {args.input} …")
    df = pd.read_parquet(args.input, columns=cols)

    canon = df["particle_type"].map(PD.LABEL_MAP)
    if canon.isna().any():
        raise ValueError("Etichette di particella non riconosciute.")

    n_input = len(df)
    original_rows = np.arange(n_input, dtype=np.int64)
    theta = df["theta"].to_numpy(dtype=np.float64)
    signals_tr1 = df[tr1].to_numpy(dtype=np.float64)
    signals_tr2 = df[tr2].to_numpy(dtype=np.float64)

    # Maximum over all strips and both sides, separately for TR1 and TR2.
    cuts = [
        ("classes_e_p_C", canon.isin(args.classes).to_numpy()),
        ("TR1_max_HG_gt_50",
         np.isfinite(signals_tr1).all(axis=1)
         & (signals_tr1.max(axis=1) > 50)),
        ("TR2_max_HG_gt_50",
         np.isfinite(signals_tr2).all(axis=1)
         & (signals_tr2.max(axis=1) > 50)),
        ("theta_lt_80_deg", np.isfinite(theta) & (theta < 80)),
    ]

    keep = np.ones(n_input, dtype=bool)
    cutflow = []

    def record(stage):
        entry = {"stage": stage, "total": int(keep.sum())}
        for cls in args.classes:
            entry[cls] = int((keep & (canon == cls).to_numpy()).sum())
        cutflow.append(entry)

    record("input")
    for stage, cut in cuts:
        keep &= cut
        record(stage)

    print("\\nSelezioni cumulative:")
    print(pd.DataFrame(cutflow).to_string(index=False))
    pd.DataFrame(cutflow).to_csv(args.out / "cutflow.csv", index=False)
    np.save(args.out / "parquet_row.npy", original_rows[keep])

    # No veto containment cut on spectra v2.
    df = df.loc[keep].reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise SystemExit(
            f"No rows left after selecting classes {args.classes}. "
            f"particle_type values present: "
            f"{sorted(canon.dropna().unique().tolist())}")
    print(f"  kept {n:,} rows, classes {args.classes}")
    print(f"  class counts:\n{canon[keep].value_counts().to_string()}")

    X_mm = np.lib.format.open_memmap(
        args.out / "X.npy", mode="w+", dtype=np.float32,
        shape=(n, PD.N_STEPS, PD.N_FEATURES))
    y = PD.encode_labels(df["particle_type"], args.classes)
    np.save(args.out / "y.npy", y)
    np.save(args.out / "classes.npy", np.array(args.classes))

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        X_mm[start:stop] = PD.build_matrix(df.iloc[start:stop])
        print(f"  … {stop:>9,} / {n:,}")
    X_mm.flush()

    (args.out / "meta.json").write_text(json.dumps({
        "source": str(args.input.resolve()),
        "source_size_bytes": args.input.stat().st_size,
        "n_input_rows": n_input,
        "selection": {
            "TR1": "max HG over 5 strips and 2 sides > 50",
            "TR2": "max HG over 4 strips and 2 sides > 50",
            "theta": "finite theta < 80 degrees",
            "trigger_values": "finite",
            "veto_containment": False,
            "veto_inputs": False,
        },
        "layers": [name for name, _ in PD.LAYERS],
        "original_row_indices": "parquet_row.npy",
        "cutflow": cutflow,
        "n_rows": n,
        "classes": args.classes,
        "shape": [PD.N_STEPS, PD.N_FEATURES],
        "hit_strip_rule": "posx != -999 sentinel; fallback strip 0 (pedestal), pos=-999",
        "created": date.today().isoformat(),
    }, indent=2))
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
