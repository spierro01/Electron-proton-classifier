"""Independent monotone empirical-quantile maps, conditional on beam setting.

Uses a fixed 1001-point probability grid by default. Ties in source quantiles
map to the midpoint probability of their grid plateau. Out-of-range values
are clipped to endpoint mapped values. No cross-energy interpolation.
"""
from pathlib import Path
import numpy as np


class QuantileMap:
    def __init__(self, energies, probabilities, source_knots, target_knots):
        self.energies = np.asarray(energies)
        self.q = np.asarray(probabilities)
        self.x = np.asarray(source_knots)
        self.y = np.asarray(target_knots)
        if self.x.shape != self.y.shape or self.x.ndim != 3:
            raise ValueError("Invalid knot arrays.")
        if self.x.shape[:2] != (len(self.energies), len(self.q)):
            raise ValueError("Invalid energy/probability dimensions.")
        if not all(np.isfinite(v).all() for v in [self.energies, self.q, self.x, self.y]):
            raise ValueError("Non-finite map.")
        if not np.all(np.diff(self.q) > 0) or self.q[0] != 0 or self.q[-1] != 1:
            raise ValueError("Invalid probability grid.")
        if (np.diff(self.x, axis=1) < 0).any() or (np.diff(self.y, axis=1) < 0).any():
            raise ValueError("Nonmonotone knots.")
        if len(np.unique(self.energies)) != len(self.energies):
            raise ValueError("Duplicate energies.")

    @classmethod
    def fit(cls, source, source_e, target, target_e, knots=1001):
        source, target = np.asarray(source), np.asarray(target)
        source_e, target_e = np.asarray(source_e), np.asarray(target_e)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("Invalid feature dimensions.")
        if source_e.shape != (len(source),) or target_e.shape != (len(target),):
            raise ValueError("Energy alignment error.")
        if not all(np.isfinite(v).all() for v in [source, target, source_e, target_e]):
            raise ValueError("Non-finite fitting data.")
        energies = np.unique(source_e)
        if not len(energies) or not np.array_equal(energies, np.unique(target_e)) or knots < 3:
            raise ValueError("Expected nonempty identical energy sets and >=3 knots.")
        q = np.linspace(0., 1., knots)
        x, y = [], []
        for energy in energies:
            print(f"  Quantiles at E={energy:g}", flush=True)
            x.append(np.quantile(source[source_e == energy], q, axis=0, method="linear"))
            y.append(np.quantile(target[target_e == energy], q, axis=0, method="linear"))
        return cls(energies, q, x, y)

    def transform(self, values, energy):
        values, energy = np.asarray(values), np.asarray(energy)
        if values.ndim != 2 or values.shape[1] != self.x.shape[2] or energy.shape != (len(values),):
            raise ValueError("Invalid transform shape.")
        if not np.isfinite(values).all() or not np.isfinite(energy).all():
            raise ValueError("Non-finite input.")
        if not np.isin(energy, self.energies).all():
            raise ValueError("Unseen energy: this baseline does not interpolate in energy.")
        output = np.empty(values.shape, dtype=np.float32)
        for k, setting in enumerate(self.energies):
            rows = np.flatnonzero(energy == setting)
            if not len(rows):
                continue
            for j in range(values.shape[1]):
                unique, first, count = np.unique(self.x[k, :, j], return_index=True, return_counts=True)
                midpoint_q = (self.q[first] + self.q[first + count - 1]) / 2
                mapped_knots = np.interp(midpoint_q, self.q, self.y[k, :, j])
                output[rows, j] = np.interp(values[rows, j], unique, mapped_knots)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite output.")
        return output

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, energies=self.energies, probabilities=self.q,
                            source_knots=self.x, target_knots=self.y)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["energies"], data["probabilities"], data["source_knots"], data["target_knots"])
