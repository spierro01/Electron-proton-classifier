"""Convert the spectra parquet into memory-mapped training tensors (N, 21, 8).

Reads the raw per-strip columns from the parquet, applies the hit-strip
aggregation rule (see pid_data.py and the README), and writes flat arrays that
train.py can memory-map — so training never needs the parquet or a big RAM
load again.

Usage:
    python preprocess.py --input /path/to/dumpMC_spectra.parquet
    python preprocess.py --input data.parquet --classes e p C He --out derived/pid

Outputs (in --out):
    X.npy        float32 (N, 21, 8)   detector matrices
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
        description="Build (N, 21, 8) training tensors from the spectra parquet.")
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
    args.out.mkdir(parents=True, exist_ok=True)

    # only read the columns we actually need — the parquet has ~285 of them
    cols = PD.required_columns() + ["particle_type"]
    print(f"Loading {args.input} …")
    df = pd.read_parquet(args.input, columns=cols)

    canon = df["particle_type"].map(PD.LABEL_MAP)
    keep = canon.isin(args.classes).to_numpy()
    df = df[keep].reset_index(drop=True)
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
        "source": str(args.input),
        "n_rows": n,
        "classes": args.classes,
        "shape": [PD.N_STEPS, PD.N_FEATURES],
        "hit_strip_rule": "posx != -999 sentinel; fallback strip 0 (pedestal), pos=-999",
        "created": date.today().isoformat(),
    }, indent=2))
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()
