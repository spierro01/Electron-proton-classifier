"""Calibrated charge Z in the trigger planes TR1/TR2 (shared by
scripts/tb_charge_audit.py and the carbon charge selection, src/data.py).

Per plane: S = sum over segments of LG (both sides) above 50 ADC; per plane and
domain a two-anchor power law S = a * D**alpha, D = Z**2 * B(beta), anchors =
proton runs >= 100 MeV (Z = 1) and carbon runs (Z = 6); Z of an event at a
carbon beam energy assumes the carbon velocity. See tb_charge_audit.py for the
full description and CLAUDE.md status item 19.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import ENERGIES

PLANES = {"TR1": 5, "TR2": 4}
THRESHOLD_ADC = 50.0          # per segment, sum of both sides
M_P, M_U = 938.272, 931.494   # proton mass, atomic mass unit [MeV]
I_PLASTIC = 64.7e-6           # mean excitation energy, PVT [MeV]
P_ANCHORS = [e for e in ENERGIES["p"] if e >= 100]   # p@70 = six mixed runs
BOUNDS = [("Z=1", 0.0, 1.5), ("Z=2", 1.5, 2.5), ("Z=3-5", 2.5, 5.0),
          ("C", 5.0, np.inf)]
CONTAM = ("nessun segnale", "Z=1", "Z=2")


def beta(t_per_nucleon: float, mass: float) -> float:
    g = 1.0 + t_per_nucleon / mass
    return float(np.sqrt(1.0 - 1.0 / g ** 2))


def bethe(b: float) -> float:
    """dE/dx / Z^2 up to a constant (no density/shell corrections)."""
    g = 1.0 / np.sqrt(1.0 - b * b)
    return (np.log(2 * 0.511 * b * b * g * g / I_PLASTIC) - b * b) / (b * b)


def plane_signals(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    cols = ["particle_type", "energy_mev"] + [
        f"RAW_{p}_LG_{i}_{s}" for p, n in PLANES.items()
        for i in range(n) for s in (0, 1)]
    df = pq.read_table(path, columns=cols).to_pandas()
    sig = {}
    for p, n in PLANES.items():
        s = np.zeros(len(df))
        for i in range(n):
            x = df[f"RAW_{p}_LG_{i}_0"].to_numpy() + df[f"RAW_{p}_LG_{i}_1"].to_numpy()
            s += np.where(x > THRESHOLD_ADC, x, 0.0)
        sig[p] = s
    return sig, df["particle_type"].to_numpy(), df["energy_mev"].to_numpy()


def calibrate(sig: dict, ptype: np.ndarray, energy: np.ndarray) -> dict:
    """Two-anchor power law S = a * D^alpha per plane (fit in log space)."""
    cal = {}
    for p in PLANES:
        d, s, tag = [], [], []
        for e in P_ANCHORS:
            m = (ptype == "p") & np.isclose(energy, e) & (sig[p] > 0)
            d.append(bethe(beta(e, M_P))); s.append(np.median(sig[p][m])); tag.append(f"p{e:g}")
        for e in ENERGIES["C"]:
            m = (ptype == "C") & np.isclose(energy, e) & (sig[p] > 0)
            x = sig[p][m]
            x = x[x > 0.5 * np.median(x)]           # carbon peak, not its low tail
            d.append(36 * bethe(beta(e, M_U))); s.append(np.median(x)); tag.append(f"C{e:g}")
        alpha, ln_a = np.polyfit(np.log(d), np.log(s), 1)
        cal[p] = {"a": float(np.exp(ln_a)), "alpha": float(alpha),
                  "anchors": {t: {"D": float(di), "S": float(si)}
                              for t, di, si in zip(tag, d, s)}}
    return cal


def charge(sig: dict, cal: dict, idx: np.ndarray, t_per_n: float) -> np.ndarray:
    """Calibrated Z at the velocity of a carbon of t_per_n MeV/n."""
    bc = bethe(beta(t_per_n, M_U))
    z = {}
    for p in PLANES:
        s = sig[p][idx]
        z[p] = np.where(s > 0, np.sqrt((s / cal[p]["a"]) ** (1 / cal[p]["alpha"]) / bc), 0.0)
    both = (z["TR1"] > 0) & (z["TR2"] > 0)
    return np.where(both, np.sqrt(z["TR1"] * z["TR2"]), np.maximum(z["TR1"], z["TR2"])), z


def classes_of(z: np.ndarray) -> np.ndarray:
    c = np.full(len(z), "nessun segnale", dtype=object)
    for name, lo, hi in BOUNDS:
        c[(z > 0) & (z >= lo) & (z < hi)] = name
    return c


def carbon_charge(path: Path) -> np.ndarray:
    """Calibrated Z for every row of a fixed-energy parquet (NaN for non-carbon
    rows), with the calibration of that file's own domain."""
    sig, ptype, energy = plane_signals(path)
    cal = calibrate(sig, ptype, energy)
    z = np.full(len(ptype), np.nan)
    for e in ENERGIES["C"]:
        idx = np.flatnonzero((ptype == "C") & np.isclose(energy, e))
        if len(idx):
            z[idx] = charge(sig, cal, idx, e)[0]
    return z
