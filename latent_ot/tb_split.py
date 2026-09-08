"""The test-beam derivation / validation split — **fixed, shared, and sacred**.

The OT maps are *fitted* on one half of the test-beam sample and *validated* on
the other. If you fit and validate on the same events you are measuring how well
the map memorised them, not whether the calibration works.

The split is a deterministic function of the seed and the TB row order, so
everybody who runs this gets the same one — which is what makes your numbers
comparable with the reference numbers quoted in the README. Cached to disk.

    Rule: fit maps on `is_derivation == True` only.
          Report every validation number on `is_derivation == False` only.

Usage:
    python tb_split.py --tb /path/to/dumpTB.parquet      # create + report

or from your own code:
    from tb_split import get_tb_split
    is_derivation = get_tb_split(n_tb_rows)              # bool mask, len == n_rows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MASK = HERE / "derived" / "tb_is_derivation.npy"

SPLIT_SEED = 42          # do not change — it is what makes the split shared
DERIVATION_FRACTION = 0.5


def get_tb_split(n_rows: int, mask_file: Path = DEFAULT_MASK) -> np.ndarray:
    """Boolean mask over TB rows: True = derivation half (fit the map here).

    Created once from the seed, then cached. If the cache exists it is reused
    and checked against `n_rows`, so a changed TB file is caught rather than
    silently producing a different split.
    """
    mask_file = Path(mask_file)
    if mask_file.is_file():
        mask = np.load(mask_file)
        if len(mask) != n_rows:
            raise ValueError(
                f"{mask_file} was built for {len(mask):,} TB rows but the file "
                f"has {n_rows:,}. Either you are pointing at a different TB "
                f"parquet, or the file changed. Do not just delete the mask — "
                f"work out which TB file is the right one first.")
        return mask

    rng = np.random.default_rng(SPLIT_SEED)
    mask = rng.random(n_rows) < DERIVATION_FRACTION
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_file, mask)
    print(f"Created TB split (seed={SPLIT_SEED}): {mask.sum():,} derivation / "
          f"{(~mask).sum():,} validation  →  {mask_file}")
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tb", type=Path, required=True, help="dumpTB.parquet")
    ap.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    n_rows = pq.ParquetFile(args.tb).metadata.num_rows
    mask = get_tb_split(n_rows, args.mask)
    print(f"TB rows       : {n_rows:,}")
    print(f"  derivation  : {mask.sum():,}   ({mask.mean():.1%})  ← fit maps here")
    print(f"  validation  : {(~mask).sum():,}   ({1 - mask.mean():.1%})  ← report here")


if __name__ == "__main__":
    main()
