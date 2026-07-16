# Transformer particle-ID classifier

A transformer that identifies the incident particle (electron / proton, and
optionally carbon / helium) from the raw detector response of a single event.

This is **step 1** of the thesis: get a working classifier trained on Monte
Carlo. Later steps deal with the fact that the classifier degrades on real
test-beam data — but ignore that for now. **The goal here is just to train this
and understand what goes into it.**

```
detector signals ──► [ 21 × 8 matrix ] ──► Transformer ──► e / p
   (per layer)         one row = one          encoder
                       detector layer
```

---

## 1. Contents

| File | What it is |
|---|---|
| `pid_data.py` | **The feature encoding.** Turns raw parquet columns into the (21, 8) matrix. Read this one first. |
| `pid_model.py` | The `ParticleTransformer` architecture. |
| `preprocess.py` | Runs the encoding over the whole parquet once → `X.npy`, `y.npy`. |
| `train.py` | Training loop, evaluation, checkpointing. |
| `predict.py` | **Inference** — run a trained model on events. |
| `pretrained/best.pt` | A **trained model you can use straight away** (see §5). |
| `requirements.txt` | Python dependencies. |

---

## 2. The data you need

You need the **spectra parquet** (e.g. `dumpMC_spectra.parquet`): simulated
events over a continuous energy and angle spectrum, with truth labels. Ask
Andrea for it — it is not in this repo (it's ~500 MB and raw data is never
committed).

> ⚠️ **Important — this is a different file from the `.pkl` files in your Colab
> notebooks.** The code here expects the parquet column naming
> (`RAW_TR1_HG_0_0`, `TR1_posx_0`, `EDEP_RAN_HG_3`, …), **not** the
> `TR1_PMT_ADC_HG0` naming you used before. Same detector, different dump
> format. If you point this at the old `.pkl` files it will fail with missing
> columns — that's expected, use the parquet.

Relevant columns: `particle_type` (the label: `electron` / `proton` /
`carbon` / `helium`) plus the ~168 detector columns listed by
`pid_data.required_columns()`.

---

## 3. The feature encoding — the important part

### 3.1 The idea

The detector is a **stack of layers**. A particle enters, deposits energy layer
by layer, and *how* that energy is distributed along the stack is what
distinguishes an electron from a proton (an electron showers early; a proton
punches through and dumps its energy at the end, at the Bragg peak).

So we encode each event as a **sequence of layers**:

- **one token = one detector layer** (21 of them),
- **8 numbers describe each layer** (position + the signals it recorded).

The transformer then attends *across* layers, which is exactly the structure
the physics has — the pattern along the stack is the signal. This is why a
transformer rather than a flat MLP on all columns: the layer sequence is
explicit.

**Every event becomes a `(21, 8)` float matrix.**

### 3.2 The 21 layers (rows), in order

| Row(s) | Layer | Note |
|---|---|---|
| 0 | `TR1` | tracker 1 — 5 strips in the raw file, collapsed to 1 row (see 3.4) |
| 1 | `TR2` | tracker 2 — 4 strips, collapsed |
| 2–13 | `RAN0` … `RAN11` | the 12 range/calorimeter layers — the core of the measurement |
| 14 | `EN1` | 3 strips, collapsed |
| 15 | `EN2` | 3 strips, collapsed |
| 16 | `BOT` | bottom veto (VETO channel 4) |
| 17–20 | `LAT0` … `LAT3` | lateral vetoes (VETO channels 0–3) |

The order is fixed and defined by `LAYERS` in `pid_data.py`. It is physical:
trackers first, then the range stack in depth order, then the vetoes.

### 3.3 The 8 features (columns) per layer

| Col | Feature | Meaning |
|---|---|---|
| 0 | `pos_x` | hit position x |
| 1 | `pos_y` | hit position y |
| 2 | `pos_z` | hit position z (depth) |
| 3 | `RAW_HG_side0` | raw ADC, **high gain**, readout side 0 |
| 4 | `RAW_HG_side1` | raw ADC, **high gain**, readout side 1 |
| 5 | `RAW_LG_side0` | raw ADC, **low gain**, readout side 0 |
| 6 | `RAW_LG_side1` | raw ADC, **low gain**, readout side 1 |
| 7 | `EDEP_HG` | calibrated energy deposit (derived from HG) |

**Why two gains (HG / LG)?** Each channel is read out twice with different
amplification, to cover a huge dynamic range:

- **Electrons and protons** deposit *little* energy → the signal lives in
  **HG**; LG is basically pedestal (noise around zero).
- **Carbon** deposits *a lot* → **HG saturates**, and the real measurement is in
  **LG**. (`EDEP_HG` is HG-derived, so it is unreliable for carbon.)

For an electron/proton classifier this mostly doesn't bite you, but it's why
both gains are kept as separate features rather than merged: the network can
learn which one to trust.

**Why two sides?** Each layer is read out at both ends; comparing the two sides
carries position/sharing information.

### 3.4 The two rules you must understand

**(a) The `-999` sentinel = "no hit".**
Positions are set to `-999` when the layer recorded nothing. It is *not* a
coordinate — it's a flag. It survives into the tensor as a literal `-999`, and
the network learns it as "this layer was empty". (Don't "clean" it to 0 — 0 is
a legitimate position, `-999` is deliberately far away and unambiguous.)

**(b) Multi-strip layers are collapsed to the hit strip.**
`TR1` (5 strips), `TR2` (4), `EN1`/`EN2` (3) exist as *separate columns per
strip* in the parquet, but we want **one row per layer**. The rule (in
`build_matrix`):

- exactly one strip carries the hit → identify it as the strip whose `posx` is
  **not** `-999`, and take all 8 features from **that strip**;
- no strip was hit → set `pos = -999`, and take the RAW/EDEP values from
  **strip 0** (which is then just pedestal).

`RAN` ×12, `BOT` and `LAT` ×4 have a single channel each, so they map **1:1** —
no collapsing needed.

This is the one piece of real logic in the encoding. If you read one function
in this repo, read `pid_data.build_matrix`.

### 3.5 Normalisation

Each of the 21×8 = 168 (layer, feature) slots is standardised independently:

```
x_normalised = (x - mean) / std
```

with `mean`/`std` computed **on the training split only** (never on val/test —
that would leak), and constant channels (`std < 1e-6`) left untouched. The
stats are computed in `train.py::compute_stats` and **saved inside the
checkpoint**, so at inference you normalise exactly the same way without
needing the training data.

---

## 4. The model

`ParticleTransformer` in `pid_model.py`:

```
(B, 21, 8)
   │  input_proj: Linear(8 → 64)              each layer → a 64-d token
   ▼
(B, 21, 64)  ──prepend a learned CLS token──► (B, 22, 64)
   │  TransformerEncoder: 3 layers, 4 heads, FFN 128, pre-norm
   ▼
take the CLS token  ──► z_enc (B, 64)         ← the event summary
   │  head: LayerNorm → Linear(64→64) → GELU → Dropout → Linear(64→n_classes)
   ▼
logits (B, n_classes)
```

- The **CLS token** is a learned vector prepended to the sequence; after
  attention it has "read" all 21 layers, so it becomes a summary of the whole
  event. The classifier head sees only this vector. (Same trick as BERT/ViT.)
- Defaults: `d_model=64`, `n_layers=3`, `n_heads=4`, `ffn_dim=128`, `dropout=0.1`
  → **~106K parameters**. Small; trains fast; don't reach for something bigger
  until this is understood.
- The head is deliberately split into separate modules (`head_norm`,
  `head_fc1`, …) rather than an `nn.Sequential`. That's so the internal
  representations (`z_enc`, `z_head`) can be pulled out later — the optimal-
  transport calibration in the next phase operates on `z_enc`. Ignore
  `latents()` / `head_from_latent()` for now; they matter later.

---

## 5. Try it now — inference with the pretrained model

`pretrained/best.pt` is **an already-trained classifier**. You can run it
immediately — no training, and not even `preprocess.py`, because the checkpoint
is self-contained: it carries the weights, the normalisation `mean`/`std`, the
class list, and the architecture hyperparameters.

```bash
pip install -r requirements.txt
python predict.py --input /path/to/dumpMC_spectra.parquet --limit 20000
```

Output looks like this (this is a real run):

```
Checkpoint : pretrained/best.pt
  classes  : ['C', 'He', 'e', 'p']
  val macro-F1 (at training time): 0.9113
  params   : 105,732   device: cuda

Predicted composition:
    C:    1,808  ( 9.04%)
   He:    2,151  (10.76%)
    e:    8,296  (41.48%)
    p:    7,745  (38.73%)

Accuracy on 20,000 labelled events: 0.9282
Per-class accuracy:
    C: 0.9116   He: 0.9361   e: 0.9267   p: 0.9315
```

> ⚠️ **This pretrained model is the 4-class one** — `['C', 'He', 'e', 'p']`, in
> that order. It is the full thesis classifier (macro-F1 0.911). If you train
> your own with `--classes e p` you get a *2-class* model, and the two are not
> interchangeable: the label indices and the output layer differ. `predict.py`
> always reads the class list **from the checkpoint**, so it does the right
> thing either way.

Useful flags: `--ckpt` (use your own checkpoint instead), `--limit` (quick try),
`--out predictions.csv` (save predictions + per-class probabilities).

If the parquet has a `particle_type` column, accuracy and a confusion matrix are
printed too. Real test-beam data has no truth labels — predictions still work,
you just get the composition without the metrics.

Reading `predict.py` is also the shortest way to see how the pieces fit:
*parquet → `build_matrix` → normalise with the checkpoint's stats → model →
softmax*.

---

## 6. How to train it

### Setup

```bash
pip install -r requirements.txt
```

(On Colab, torch/numpy/pandas/sklearn are already there; you may only need
`pyarrow` for parquet: `pip install pyarrow`.)

### Step 1 — preprocess (once)

Applies the encoding to the whole parquet and writes memory-mapped arrays, so
training never touches the parquet again:

```bash
python preprocess.py --input /path/to/dumpMC_spectra.parquet --classes e p
```

Writes to `derived/pid/`:

| File | |
|---|---|
| `X.npy` | float32 `(N, 21, 8)` — the detector matrices |
| `y.npy` | int64 `(N,)` — labels |
| `classes.npy` | class names; **position = label index** |
| `meta.json` | provenance |

Class names are canonical: `e`, `p`, `C`, `He` (the parquet's
`electron`/`proton`/… are mapped for you). `--classes e p` gives you the
2-class electron/proton problem — the right place to start. To reproduce the
full thesis classifier use `--classes C He e p`.

> The **order** you pass to `--classes` defines the label index (`e p` → e=0,
> p=1). Keep it consistent.

### Step 2 — train

```bash
# quick smoke test first — should take a minute or two
python train.py --data derived/pid --epochs 1 --limit 50000

# the real thing
python train.py --data derived/pid --epochs 30 --batch_size 2048
```

Useful flags: `--epochs`, `--batch_size`, `--lr`, `--d_model`, `--n_layers`,
`--dropout`, `--limit` (subset for testing), `--num_workers` (set `2` on Colab).
It uses the GPU automatically if one is available.

Outputs:
- `checkpoints/pid_transformer/best.pt` — best checkpoint by **validation
  macro-F1**, self-contained (weights + `mean`/`std` + `classes` + hyperparams).
- `derived/pid/split_indices.npz` — the exact train/val/test indices (seeded,
  reproducible).
- `derived/pid/test_metrics.json` — final held-out metrics.

### What you should see

For the **4-class** version the reference numbers are macro-F1 **0.911**,
accuracy **0.930**, AUC **0.991**. The **2-class e/p** problem is much easier —
expect high accuracy quickly. If e/p looks near-perfect, that's real: they are
very different in this detector. The hard part comes later, on real data.

---

## 7. Details worth knowing

- **Rows in the parquet are ordered by class.** So `train.py` always permutes
  the *full* dataset before applying `--limit` — otherwise a subset would be
  one single class. Don't remove that.
- **Class weighting.** The loss is weighted by inverse class frequency, so a
  rare class isn't ignored. Printed at startup as `Class weights (inv_freq)`.
- **Why macro-F1 for checkpointing?** It weights every class equally, unlike
  accuracy which a dominant class can carry. With imbalanced classes accuracy
  flatters you.
- **Why memory-map?** `X.npy` is large; `mmap_mode="r"` lets the OS page it in
  on demand instead of loading it all into RAM.
- **Label smoothing 0.05** and **OneCycleLR** are mild regularisation/schedule
  choices from the reference recipe — reasonable defaults, not sacred.
- **Reproducibility:** `--seed` (default 42) fixes the split and init.

## 8. Things to try

0. **Run `predict.py` with the pretrained model first** (§5) — before training
   anything. Check the confusion matrix: which particles does it mix up?
1. Get the smoke test running end-to-end. Confirm you understand the shapes:
   why 21, why 8.
2. Train e/p properly and look at the **confusion matrix** — which events get
   confused, and does it make physical sense?
3. Add `C` and `He` (`--classes C He e p`) and see which pairs are hard.
4. Ablate: what happens if you feed only the `RAN` layers? Only `EDEP`? This
   builds intuition for *which* features carry the discrimination.
5. Change `d_model` / `n_layers` — does bigger actually help, or just overfit?

Once this is comfortable, the next phase is the interesting one: the classifier
is trained on simulation, and on **real test-beam data it degrades** (electrons
drop from 0.989 → 0.916). Fixing that — without labels on real data — is what
optimal-transport calibration is for.
