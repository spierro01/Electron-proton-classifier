"""Build (N_STEPS, N_FEATURES) = (21, 8) detector matrices from the parquet files.

Replicates the input layout of the hepd2_transformer_pid project
(one token per detector layer, 8 features per token):

    pos_x, pos_y, pos_z, RAW_HG_side0, RAW_HG_side1, RAW_LG_side0, RAW_LG_side1, EDEP_HG

Layer order: TR1, TR2, RAN x 12, EN1, EN2, BOT, LAT x 4  -> 21 tokens.

Our parquet stores multi-strip layers (TR1: 5 strips, TR2: 4, EN1/EN2: 3)
as separate per-strip columns. The hit strip is identified by its position
not being the -999 sentinel (exactly one strip per layer carries the hit).
Aggregation rule per multi-strip layer:
  * hit strip present  -> take pos/RAW/EDEP from that strip
  * no hit             -> pos = -999, RAW/EDEP from strip 0 (pedestal)
RAN, BOT (VETO ch 4) and LAT (VETO ch 0-3) layers map 1:1 to parquet columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_STEPS = 21
N_FEATURES = 8
POS_SENTINEL = -999.0

# canonical class names shared between spectra and fixed-energy files
LABEL_MAP = {
    "electron": "e", "proton": "p", "carbon": "C", "helium": "He",
    "e": "e", "p": "p", "C": "C", "He": "He",
}


def _strip_cols(det: str, n_strips: int, pos_prefix: str | None = None) -> dict:
    """Column names for a multi-strip detector, one list per feature."""
    pp = pos_prefix or det
    return {
        "posx": [f"{pp}_posx_{i}" for i in range(n_strips)],
        "posy": [f"{pp}_posy_{i}" for i in range(n_strips)],
        "posz": [f"{pp}_posz_{i}" for i in range(n_strips)],
        "hg0": [f"RAW_{det}_HG_{i}_0" for i in range(n_strips)],
        "hg1": [f"RAW_{det}_HG_{i}_1" for i in range(n_strips)],
        "lg0": [f"RAW_{det}_LG_{i}_0" for i in range(n_strips)],
        "lg1": [f"RAW_{det}_LG_{i}_1" for i in range(n_strips)],
        "edep": [f"EDEP_{det}_HG_{i}" for i in range(n_strips)],
    }


# (layer_name, spec) — spec is either
#   ("multi", strip_cols_dict)                     -> hit-strip aggregation
#   ("direct", [8 column names in feature order])  -> 1:1 mapping
LAYERS: list[tuple[str, tuple]] = [
    ("TR1", ("multi", _strip_cols("TR1", 5))),
    ("TR2", ("multi", _strip_cols("TR2", 4))),
    *[(f"RAN{i}", ("direct", [
        f"RAN_posx_{i}", f"RAN_posy_{i}", f"RAN_posz_{i}",
        f"RAW_RAN_HG_{i}_0", f"RAW_RAN_HG_{i}_1",
        f"RAW_RAN_LG_{i}_0", f"RAW_RAN_LG_{i}_1",
        f"EDEP_RAN_HG_{i}"])) for i in range(12)],
    ("EN1", ("multi", _strip_cols("EN1", 3))),
    ("EN2", ("multi", _strip_cols("EN2", 3))),
    ("BOT", ("direct", [
        "BOT_posx_0", "BOT_posy_0", "BOT_posz_0",
        "RAW_VETO_HG_4_0", "RAW_VETO_HG_4_1",
        "RAW_VETO_LG_4_0", "RAW_VETO_LG_4_1",
        "EDEP_VETO_HG_4"])),
    *[(f"LAT{i}", ("direct", [
        f"LAT_posx_{i}", f"LAT_posy_{i}", f"LAT_posz_{i}",
        f"RAW_VETO_HG_{i}_0", f"RAW_VETO_HG_{i}_1",
        f"RAW_VETO_LG_{i}_0", f"RAW_VETO_LG_{i}_1",
        f"EDEP_VETO_HG_{i}"])) for i in range(4)],
]
assert len(LAYERS) == N_STEPS


def required_columns() -> list[str]:
    cols: list[str] = []
    for _, (kind, spec) in LAYERS:
        if kind == "direct":
            cols.extend(spec)
        else:
            for lst in spec.values():
                cols.extend(lst)
    return cols


def _pick(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Row-wise selection: values (N, n_strips), idx (N,) -> (N,)."""
    return values[np.arange(len(idx)), idx]


def build_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert a dataframe slice to (len(df), 21, 8) float32."""
    n = len(df)
    X = np.empty((n, N_STEPS, N_FEATURES), dtype=np.float32)
    for step, (_, (kind, spec)) in enumerate(LAYERS):
        if kind == "direct":
            X[:, step, :] = df[spec].to_numpy(dtype=np.float32)
            continue
        posx = df[spec["posx"]].to_numpy(dtype=np.float32)
        valid = posx > (POS_SENTINEL + 1.0)
        has_hit = valid.any(axis=1)
        hit = np.where(has_hit, valid.argmax(axis=1), 0)
        for j, feat in enumerate(["posx", "posy", "posz", "hg0", "hg1", "lg0", "lg1", "edep"]):
            vals = _pick(df[spec[feat]].to_numpy(dtype=np.float32), hit)
            if feat in ("posx", "posy", "posz"):
                vals = np.where(has_hit, vals, POS_SENTINEL)
            X[:, step, j] = vals
    return X


def encode_labels(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """Map particle_type strings (either naming scheme) to indices in `classes`."""
    canon = labels.map(LABEL_MAP)
    if canon.isna().any():
        bad = labels[canon.isna()].unique()
        raise ValueError(f"unknown particle_type values: {bad}")
    lut = {c: i for i, c in enumerate(classes)}
    return canon.map(lut).to_numpy(dtype=np.int64)
