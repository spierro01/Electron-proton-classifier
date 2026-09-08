"""Check that everything this exercise needs is in place, and show you what the
two samples actually contain. Run this before task 0.

It does four things:

  1. checks that the MC and TB parquets exist and carry every detector column
     `pid_data.required_columns()` asks for;
  2. prints the class / beam-energy inventory of both files side by side, and
     flags energies that exist in one domain but not the other;
  3. loads your classifier checkpoint and reports its class list;
  4. pushes a sample of events from both domains through the model and prints
     the latent shapes plus the raw accuracy on each — i.e. the MC→TB
     degradation this exercise is about, on a small sample.

Nothing here is part of the exercise: it is a setup check and a look at the
data. The extraction, the OT map and the evaluation are yours to write.

Usage:
    python check_setup.py --mc /path/to/dumpMC.parquet \
                          --tb /path/to/dumpTB.parquet \
                          --ckpt ../transformer/pretrained/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformer"))   # reuse the classifier code

import pid_data as PD                                   # noqa: E402
from pid_model import ParticleTransformer               # noqa: E402

TB_CLASSES = ["e", "p", "C"]
BATCH = 50_000          # parquet read granularity
PER_BATCH = 500         # events kept per batch for the sanity inference


# --------------------------------------------------------------------------
# 1. schema
# --------------------------------------------------------------------------
def check_schema(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Not found: {path}\n(the fixed-energy dumps are in the "
                         f"shared Google Drive folder — raw data is never "
                         f"committed)")
    have = set(pq.ParquetFile(path).schema.names)
    missing = [c for c in PD.required_columns() if c not in have]
    if missing:
        raise SystemExit(
            f"{path.name} is missing {len(missing)} detector columns, e.g. "
            f"{missing[:4]}.\nThis exercise expects the parquet schema "
            f"(RAW_TR1_HG_0_0, TR1_posx_0, …), not the older .pkl naming.")
    for c in ("particle_type", "energy_mev"):
        if c not in have:
            raise SystemExit(f"{path.name} has no `{c}` column — wrong file? "
                             f"You want the fixed-energy dumps, not the spectra "
                             f"file the classifier was trained on.")


# --------------------------------------------------------------------------
# 2. inventory
# --------------------------------------------------------------------------
def inventory(path: Path) -> pd.DataFrame:
    """(class, energy) -> number of events. Only two columns are read."""
    df = pd.read_parquet(path, columns=["particle_type", "energy_mev"])
    df["cls"] = df["particle_type"].map(PD.LABEL_MAP)
    return df.groupby(["cls", "energy_mev"]).size().rename("n").reset_index()


def print_inventory(mc: pd.DataFrame, tb: pd.DataFrame) -> None:
    mc_i = {(r.cls, r.energy_mev): r.n for r in mc.itertuples()}
    tb_i = {(r.cls, r.energy_mev): r.n for r in tb.itertuples()}
    keys = sorted(set(mc_i) | set(tb_i), key=lambda k: (TB_CLASSES.index(k[0])
                  if k[0] in TB_CLASSES else 99, k[1]))
    print(f"\n{'class':<6}{'energy [MeV]':>14}{'MC events':>12}{'TB events':>12}   note")
    print("-" * 68)
    for cls, e in keys:
        n_mc, n_tb = mc_i.get((cls, e), 0), tb_i.get((cls, e), 0)
        note = ""
        if n_mc == 0:
            note = "TB only — no MC to transport from, skip it"
        elif n_tb == 0:
            note = "MC only — nothing to transport to"
        print(f"{cls:<6}{e:>14g}{n_mc:>12,}{n_tb:>12,}   {note}")
    print("-" * 68)
    for cls in TB_CLASSES:
        n_mc = sum(v for (c, _), v in mc_i.items() if c == cls)
        n_tb = sum(v for (c, _), v in tb_i.items() if c == cls)
        print(f"{cls:<6}{'total':>14}{n_mc:>12,}{n_tb:>12,}")


# --------------------------------------------------------------------------
# 3 + 4. model and a look at the latents
# --------------------------------------------------------------------------
def load_model(ckpt_path: Path, device: str):
    if not ckpt_path.is_file():
        raise SystemExit(f"Not found: {ckpt_path}\nEither train your own "
                         f"classifier (../transformer) or point --ckpt at "
                         f"../transformer/pretrained/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    a = ckpt["args"]
    model = ParticleTransformer(
        num_classes=len(ckpt["classes"]), d_model=a["d_model"],
        n_heads=a["n_heads"], n_layers=a["n_layers"], ffn_dim=a["ffn_dim"],
        dropout=a["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return model, ckpt["classes"], mean, std


@torch.no_grad()
def sample_latents(path: Path, model, classes, mean, std, device: str) -> dict:
    """Stream the file, keep a slice of every batch, return latents + labels.

    The slice-per-batch is only so the sample spans the whole file: rows are
    ordered by class, so the first N rows would be one single class.
    """
    cols = PD.required_columns() + ["particle_type", "energy_mev"]
    z, lg, lab, en = [], [], [], []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=cols):
        df = batch.to_pandas().iloc[:PER_BATCH]
        X = (PD.build_matrix(df) - mean) / std
        out = model.latents(torch.from_numpy(X.astype(np.float32)).to(device))
        z.append(out["z_enc"].cpu().numpy())
        lg.append(out["logits"].cpu().numpy())
        lab.append(PD.encode_labels(df["particle_type"], classes))
        en.append(df["energy_mev"].to_numpy())
    return {"z_enc": np.concatenate(z), "logits": np.concatenate(lg),
            "label": np.concatenate(lab), "energy_mev": np.concatenate(en)}


def report_accuracy(tag: str, lat: dict, classes: list[str]) -> None:
    pred = lat["logits"].argmax(1)
    print(f"\n  {tag}:  z_enc {lat['z_enc'].shape}   "
          f"logits {lat['logits'].shape}   "
          f"|z| mean {np.abs(lat['z_enc']).mean():.3f}")
    for cls in TB_CLASSES:
        if cls not in classes:
            continue
        i = classes.index(cls)
        m = lat["label"] == i
        if m.any():
            print(f"      {cls:>2}: accuracy {float((pred[m] == i).mean()):.4f}"
                  f"   (n={int(m.sum()):,})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mc", type=Path, required=True, help="dumpMC.parquet (fixed energy)")
    ap.add_argument("--tb", type=Path, required=True, help="dumpTB.parquet")
    ap.add_argument("--ckpt", type=Path,
                    default=HERE.parent / "transformer" / "pretrained" / "best.pt")
    ap.add_argument("--skip_inference", action="store_true",
                    help="schema + inventory only (faster)")
    args = ap.parse_args()

    print("=" * 68)
    print("1. schema")
    for p in (args.mc, args.tb):
        check_schema(p)
        print(f"   OK  {p}  ({pq.ParquetFile(p).metadata.num_rows:,} rows)")

    print("\n" + "=" * 68)
    print("2. what is in the two samples")
    print_inventory(inventory(args.mc), inventory(args.tb))

    print("\n" + "=" * 68)
    print("3. classifier")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, mean, std = load_model(args.ckpt, device)
    print(f"   {args.ckpt}")
    print(f"   classes : {classes}   (label index = position in this list)")
    print(f"   params  : {model.num_parameters:,}   device: {device}")
    missing = [c for c in TB_CLASSES if c not in classes]
    if missing:
        raise SystemExit(f"   This checkpoint cannot classify {missing} — you "
                         f"need a model trained on at least e, p and C.")

    if args.skip_inference:
        return

    print("\n" + "=" * 68)
    print("4. latents, and the degradation you are about to fix")
    print(f"   (sample of {PER_BATCH} events per {BATCH:,}-row batch, both files)")
    for tag, path in (("MC", args.mc), ("TB", args.tb)):
        report_accuracy(tag, sample_latents(path, model, classes, mean, std,
                                            device), classes)
    print("\n   The MC→TB drop above is the whole problem. Task 0 measures it "
          "properly;\n   tasks 3–6 are about closing it.")
    print("   (These are sampled events, not the full files — and the TB carbon "
          "number\n    includes the three energies with no MC, so it reads lower "
          "than it should.)")


if __name__ == "__main__":
    main()
