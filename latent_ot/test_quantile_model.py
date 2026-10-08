"""Fast tests using synthetic arrays; does not read or change experiment data."""
from pathlib import Path
import tempfile
import numpy as np
from quantile_model import QuantileMap


def main():
    z = np.linspace(-3, 3, 201).reshape(-1, 1)
    source = np.vstack([np.hstack([z, 2*z]), np.hstack([z, 2*z])]).astype(np.float32)
    energy = np.repeat([6., 9.], 201)
    target = source.copy()
    target[energy == 6] = 2 * source[energy == 6] + 3
    target[energy == 9] = .5 * source[energy == 9] - 4
    original = source.copy()
    mapping = QuantileMap.fit(source, energy, target, energy, 101)
    mapped = mapping.transform(source, energy)
    np.testing.assert_allclose(mapped, target, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(source, original)
    assert (np.diff(mapped[energy == 6], axis=0) >= 0).all()
    assert (np.diff(mapped[energy == 9], axis=0) >= 0).all()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "map.npz"
        mapping.save(path)
        np.testing.assert_array_equal(QuantileMap.load(path).transform(source, energy), mapped)
    constant = QuantileMap.fit(np.zeros((201, 2)), np.ones(201), np.hstack([z, z]), np.ones(201), 101)
    np.testing.assert_allclose(constant.transform(np.zeros((3,2)), np.ones(3)), 0., atol=1e-7)
    tied = QuantileMap.fit(np.repeat([[0.,0.],[1.,1.],[2.,2.]], 20, axis=0), np.ones(60),
                          np.repeat([[4.,4.],[8.,8.],[9.,9.]], 20, axis=0), np.ones(60), 101)
    test = tied.transform(np.array([[-1.,-1.],[0.,0.],[1.,1.],[2.,2.],[3.,3.]]), np.ones(5))
    assert np.isfinite(test).all() and (np.diff(test, axis=0) >= 0).all()
    np.testing.assert_array_equal(test[0], test[1])
    np.testing.assert_array_equal(test[-1], test[-2])
    for bad_values, bad_energy in [(source[:1], np.array([120.])), (np.full((1,2), np.nan), np.array([6.]))]:
        try:
            mapping.transform(bad_values, bad_energy)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted.")
    print("PASS: QUANTILE TESTS — conditional mapping, monotonicity, ties, constant coordinates, reload and input validation.")


if __name__ == "__main__":
    main()
