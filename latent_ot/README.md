# Latent-space optimal transport: calibrating the classifier on real data

The classifier from [`../transformer`](../transformer/README.md) is trained on
simulation and degrades on real test-beam data. Here you correct that — without
retraining it — by learning an OT map that moves the network's internal
representation of a simulated event onto the real one.

```
                MC (simulation)                     TB (real beam)
                      │                                   │
  event ──► [21×8] ──►│──► transformer encoder ──► z_enc (64-d) ──► head ──► e/p/C
                      │                              │  ▲                 (frozen)
                      └──────────────────────────────┘  │
                                    T = ∇g  ────────────┘
                          the map you fit, one per class
```

You have both halves already: the ICNN and the Makkuva dual from the
[toy exercise](../optimal_transport_toy/README.md), and the classifier.

---

## What you need

- `dumpMC.parquet` and `dumpTB.parquet` — in the shared Google Drive folder.
- A classifier checkpoint: yours, or `../transformer/pretrained/best.pt`.
- Your ICNN / OT code from the toy exercise, generalised past 2-D.
- `pip install -r requirements.txt`

> ⚠️ **Not `dumpMC_spectra.parquet`.** That is the continuous-spectrum sample the
> classifier was *trained* on — different energies, different angles, and it
> contains helium, which the test beam does not. Calibration uses the
> **fixed-energy** dumps.

> ⚠️ Read the class list from the checkpoint and index by it. `pretrained/best.pt`
> has 4 classes (`['C','He','e','p']`), yours has 3. Use one checkpoint
> throughout.

## Rules

1. **The TB split is sacred.** Run `python tb_split.py --tb dumpTB.parquet` once.
   Fit maps on the derivation half **only**; report every number on the
   validation half **only**. Never refit the split.
2. **The classifier is frozen.** No transformer weight changes in this exercise.
3. **One map per class.** e, p and C are fitted separately.

## Start here

```bash
python check_setup.py --mc dumpMC.parquet --tb dumpTB.parquet
```

Checks the files, prints the class × energy inventory of both, loads the
checkpoint, and shows the MC→TB drop on a sample. A few seconds.

---

## Task 0 — quantify the problem

Before correcting anything, measure what you are correcting.

- Accuracy per class, MC vs TB, **and per beam energy**.
- Confusion matrix for each domain: when electrons are misclassified, what do
  they become?
- One plot: accuracy vs beam energy, MC and TB, one panel per class.

*Sanity check:* electrons must come out as the worst class, and their drop must
be strongly energy-dependent — not a flat offset. If you get neither, something
is wrong upstream.

This table is the "before" column for everything that follows, and the opening
of your thesis chapter (task 7).

## Task 1 — extract the latents

Push both parquets through the frozen classifier once and store, per event:
`z_enc` (N, 64), `logits`, `label`, `energy_mev`, `is_derivation` (TB only, from
`tb_split.get_tb_split`), and the class list. One `.npz` per domain.

- Stream with `pq.ParquetFile(...).iter_batches(...)` — do not read the whole
  parquet into memory.
- Normalise with the `mean`/`std` **stored in the checkpoint**.
- `model.eval()` + `torch.no_grad()`.

*Check:* recompute accuracy from the stored logits — it must match
`../transformer/predict.py` on the same file.

> Your `../transformer/extract_latents.py` is nearly there. Four changes:
> point it at `dumpMC.parquet`; store `label` / `energy_mev` / `is_derivation`
> next to `z_enc` (without them no per-class or per-energy map is possible);
> stream instead of `pd.read_parquet` on the whole file (it peaks at 12 GB);
> and request `particle_type` and `energy_mev` explicitly — they are not in
> `required_columns()`, so your `try/except KeyError` never fires.

## Task 2 — look at the latent space

- PCA of `z_enc`, **one class at a time**, fitted on MC and applied to both
  domains. Overlay MC and TB.
- Train a MC-vs-TB discriminator on `z_enc`, per class; report the ROC AUC on
  held-out events. This is your **closure metric** — after calibration it should
  move towards 0.5. Keep the number: it is also evidence for task 7.

> Redo `plot_pca_shift.py` per class. Pooled over all classes it is dominated by
> the class structure and by the different class fractions in the two files, not
> by the MC/TB shift — I measured this: pooled, simulation-vs-simulation
> separates as well as simulation-vs-data.

## Task 3 — fit the class-conditional OT maps

Per class `c ∈ {e, p, C}`:

- source = MC `z_enc` of class `c`; target = TB `z_enc` of class `c`,
  **derivation half only**, dropping the carbon energies with no MC.
- Standardise each domain separately; store the stats (you need the inverse).
- Fit `f`, `g` as ICNNs, `T(z) = ∇g(z)` — the same dual as the toy, at `d = 64`.
- Save map + config + stats per class; verify reload invariance.

Samples are **unpaired**: no event-by-event loss.

Known-working starting point: hidden 2048, batch 1024, 10 000 outer steps,
10 `g`-updates then 4 `f`-updates each, Adam lr 5e-4 → 5e-6 cosine,
`weight_decay = 0`, grad clip 10, plus ~1000 MSE steps pre-training `∇g ≈ id`.
~10–20 min per class on GPU. Log both losses and `‖T(z) − z‖`.

## Task 4 — does the calibration work? (all energies together)

The classifier's accuracy *is* the PID selection efficiency you would quote in a
measurement. Raw MC will tell you the classifier is nearly perfect; the test
beam will tell you it is not. The question is whether **calibrated** MC
reproduces what the test beam says.

For each class, take every MC event of that class and:

1. take its latent `z_enc`;
2. replace it with `T(z_enc)`, the map from task 3;
3. push that through `head_from_latent` and take the argmax;
4. compare with the MC truth label — **unchanged**, because the map corrects the
   detector response, not the identity of the particle.

The fraction classified correctly is the **calibrated MC accuracy**. Put it next
to the two numbers you already have from task 0 — raw MC, and TB on the
validation half — in a table of this shape:

| class | raw MC | calibrated | TB |
|---|---:|---:|---:|
| X | 0.95 | **0.88** | 0.87 |
| Y | 0.99 | **0.97** | 0.96 |

> Invented numbers, to show what the result looks like: the calibrated column
> lands next to **TB**, not next to raw MC. Yours will be different — measure
> them, do not tune towards anything.

*Pass criterion, in terms of your own task-0 numbers:* calibrated must be much
closer to TB than to raw MC, i.e. the residual `|calibrated − TB|` should be
small compared with the MC→TB drop you measured in task 0. Quote both.

Then the closure check: re-run the task-2 MC-vs-TB discriminator with the
calibrated latents in place of the MC ones — on `z_enc`, and also on the
`logits`. The logits AUC should fall clearly towards 0.5. The `z_enc` AUC will
stay high; that is expected, see the troubleshooting table.

## Task 5 — the same result, energy by energy

Exactly the task-4 accuracy, but **split by beam energy**: one point per energy,
three curves (raw MC / calibrated / TB), one panel per class.

Why separately: the maps of task 3 pool all energies, so they can only apply one
*average* correction. Task 4 asks whether the average is right; this asks whether
it is right locally. Expect it not to be — a map can be excellent integrated and
badly wrong at the energies where the shift is strongest, with a local
discrepancy far larger than the integrated one. Schematically:

```
acc │ ─────────────────────  raw MC        (flat, too optimistic)
    │ ─── ─── ─── ─── ─── ─  calibrated    (flat: one average correction)
    │        ╲   ╱           TB            (has structure the map misses)
    │         ╲_╱
    └──────────────────────► beam energy
```

Deliverable: the plot with your own numbers, plus a paragraph on which classes
suffer and which do not.

## Task 6 — make the map energy-dependent, then redo tasks 4 and 5

Refit the three maps with the beam energy as a **conditioning variable**,
`T(z | E)`: a *partially* input-convex network — convex in `z`, as the map
requires, unconstrained in `E`. Then rerun tasks 4 and 5 with these maps and
compare the two sets of curves.

Two traps:

- **Stratified batches.** MC and TB have different energy fractions, so a random
  batch from each side compares different energy mixtures and the map learns the
  mixture difference instead of the physics. Draw the same energy multiset on
  both sides.
- **Bounded gates** on the conditioning path. Unbounded ones compound
  multiplicatively across layers and the losses reach ~1e14.

*Target:* the calibrated curve of task 5 now follows TB energy by energy, dips
included, and the per-energy discrepancies shrink to roughly the level of the
integrated one from task 4. Not every class gains equally — find out which one
gains least, and why.

**This is the map that goes in the thesis.**

## Task 7 — write the method chapter

Start the thesis chapter on the method. Use your own numbers from tasks 0–6 as
the evidence. It has to answer:

- Why calibrate the **latent** rather than the input features?
- Why a **multivariate** map rather than one 1-D correction per feature? (Think
  about which of our mismodellings are correlated: `EDEP_HG` is derived from
  `RAW_HG`, the two readout sides are one deposit, the Bragg pattern is a shape
  *across* the RAN layers.)
- Why not **reweight** the simulation? — your task-2 AUCs are the argument.
- Why not **fine-tune** on the beam data?
- Why does the classifier head stay **frozen**, and what does that buy?
- Why **optimal** transport — what does minimal-cost mean here, and why is it
  the right requirement for a correction?

---

## When it goes wrong

| symptom | look at |
|---|---|
| losses run away to ~1e7, map collapses to identity | Is the quadratic term in `g` fixed? A fixed `½‖z‖²` forces `∇T ⪰ I` — the map can only expand. Make it trainable. |
| losses ~1e14, conditional model only | Unbounded gates (task 6). |
| pre-training MSE ~1e8 at width 2048, fine at 128 | Positive-weight init not scaled by fan-in: the hidden→hidden weight *sum* grows with width. |
| oscillates, never settles | Adam momentum spirals a min–max — try `betas = (0, 0.9)`; check `g` gets more updates than `f`. |
| `T(z) ≈ z`, RMS displacement ≈ 0 | No identity pre-training, or `∇g` computed inside `torch.no_grad()`. Use `torch.autograd.grad`, then `.detach()`. |
| conditional map ignores `E` | Batches not stratified. |
| **`z_enc` AUC stays ~1.0 after calibration** | **Expected — don't chase it.** A freshly-trained discriminator always finds a residual direction in 64-d (the paper sees the same, its Fig. 9b). Judge closure on the logits and the accuracy. |
| calibrated accuracy suspiciously perfect | You are evaluating on the derivation half. |

## Deliverables

- extraction script + the two `.npz` files;
- ICNN / PICNN and OT-map code, training entry point with a fixed seed;
- 3 class-conditional + 3 energy-conditional maps;
- evaluation script producing, on the TB validation half: before/after accuracy
  tables and per-energy plots, closure AUCs on `z_enc` and logits, PCA overlays
  of MC / MC-calibrated / TB;
- a short log of the instabilities you hit and what fixed them;
- the draft method chapter (task 7).

## References

- Algren, Golling, Di Bello, Pollard, *Mind the Gap*,
  [arXiv:2507.08867](https://arxiv.org/abs/2507.08867) — §2.1 and §4.
- Makkuva et al., *OT mapping via input convex neural networks*,
  [arXiv:1908.10962](https://arxiv.org/abs/1908.10962)
- Amos, Xu, Kolter, *Input Convex Neural Networks*,
  [arXiv:1609.07152](https://arxiv.org/abs/1609.07152)
- ATLAS, OT calibration of flavour tagging, EPJC 85 (2025) 1272
