# Neural optimal transport: toy exercise

## Goal

Learn a map that transports samples from the **source** distribution to the
**target** distribution. The samples are two-dimensional so that the result is
easy to visualise, but the target is nonlinear: matching only the mean and
covariance is not enough.

You are expected to implement the complete optimal-transport part yourself.
The repository does not contain the generating transformation or a reference
OT implementation.

## Data

The file [`data/toy_ot_data.npz`](data/toy_ot_data.npz) contains four NumPy
arrays:

| Array | Shape | Use |
| --- | ---: | --- |
| `source_train` | `(20000, 2)` | Fit the source side of the map |
| `target_train` | `(20000, 2)` | Fit the target side of the map |
| `source_test` | `(5000, 2)` | Apply the fitted map to unseen source samples |
| `target_test` | `(5000, 2)` | Evaluate distributional closure |

Load it with:

```python
import numpy as np

data = np.load("data/toy_ot_data.npz")
source_train = data["source_train"]
target_train = data["target_train"]
```

The samples are **unpaired**. In particular, row `i` in a source array does not
correspond to row `i` in a target array. An event-by-event regression loss is
therefore not appropriate. Do not use either test array during training or
hyperparameter selection.

### Quick visual check

From this directory, run:

```bash
python3 visualize_data.py
```

This validates the arrays, prints their mean and standard deviation, and writes
`figures/toy_ot_data.png`. The figure compares train and test samples using
common axis ranges and also shows both one-dimensional marginals. You can pass
different paths with `--data` and `--output`.

## Required implementation

Implement a neural approximation to the quadratic-cost (2-Wasserstein)
optimal transport map using PyTorch and automatic differentiation.

1. Explore the train distributions with marginal histograms and a 2D scatter
   or density plot.
2. Standardise the inputs using statistics computed from the training arrays
   only. Store all statistics needed to return transformed values in the
   original target coordinates.
3. Implement input-convex neural networks (ICNNs) for two scalar convex
   potentials, `f` and `g`. Enforce the non-negative weights required for
   convexity rather than relying on unconstrained dense layers.
4. Train the potentials with alternating optimisation of the empirical
   minimax objective

   ```text
   max_f min_g  E_source[f(grad g(z)) - <z, grad g(z)>]
                - E_target[f(y)].
   ```

   Clearly document how many `g` and `f` updates are made per outer step and
   how convexity is maintained during optimisation.
5. Define the learned transport as `T(z) = grad g(z)`. Implement batched
   `fit(...)` and `transform(...)`/`apply(...)` operations, including the
   inverse standardisation to target coordinates.
6. Save and reload the model, its configuration, and its standardisation
   statistics. Verify that reloading does not change transformed samples.

PyTorch building blocks and autograd are allowed. To make this an implementation
exercise, do not use a ready-made OT solver, a pre-built ICNN, or another
project's OT calibration implementation for the core solution. Libraries such
as scikit-learn may be used for evaluation.

## Validation

Apply the final model to `source_test` and compare the result only with
`target_test`. Report at least:

- plots of source, target, and transported source in the same axis ranges;
- per-coordinate mean and standard deviation before and after transport;
- a two-sample classifier ROC AUC before and after transport, using a held-out
  classifier split (`0.5` means that the distributions are indistinguishable).

As a practical target, the post-transport AUC should be below `0.62`, should be
substantially lower than the pre-transport AUC, and the visual curvature of the
target should be reproduced. A low AUC on the training arrays alone is not
evidence of closure.

## Deliverables

- the ICNN and OT-map source code;
- a reproducible training entry point with a fixed random seed;
- a saved model checkpoint;
- an evaluation script or notebook that runs on the test arrays;
- a short report containing the plots, metrics, design choices, and any
  observed training instabilities.

CPU execution should be supported. CUDA acceleration may be used when
available.
