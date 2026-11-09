# Fixed-threshold tail analysis

Run these commands from `latent_ot` in the existing project environment.
Keep the original no-veto classifier checkpoint and the DQ derivation split.

## Select events and inspect scores

Set DATA to the directory containing dumpMC_v2.parquet, dumpTB_v2_dq.parquet
and tb_is_derivation_v2_dq.npy. Set PREDICTIONS to the original DQ baseline
evaluation directory containing the selected prediction CSVs and provenance.json.

```bash
python tail_analysis/selection_diagnostics.py --data "$DATA"
python tail_analysis/final_selection.py --data "$DATA"
python tail_analysis/raw_tail.py --data "$DATA" --predictions "$PREDICTIONS"
```

The first stage matches the track to the fired bar in both trigger planes.
The second fits charge calibrations by domain and plane and requires Z >= 2.5
for carbon. The output masks follow original parquet row order. The third
stage computes score distributions and signal tail fractions before transport.

## Train maps and compare with TB validation

```bash
python -u tail_analysis/start_nominal.py
python -u tail_analysis/tail_campaign.py evaluate nominal
python -u tail_analysis/tail_campaign.py train loeo
python -u tail_analysis/tail_campaign.py evaluate loeo
python -u tail_analysis/plot_tail_results.py
python -u tail_analysis/tail_campaign.py train bootstrap
python -u tail_analysis/tail_campaign.py evaluate bootstrap
```

The nominal stage filters the verified DQ latent archives using the final masks
and trains three maps. LOEO trains 25 maps and tests each at the omitted energy.
Bootstrap uses five TB derivation replicas per species. Results are written
to results/nominal_tail_charge and results/fixed_tail_campaign.

For each species the primary observable is the fraction with
`D <= cut`, where `D = ln(2*p_signal/(p_other1+p_other2))`.
The cuts are [2, 1, 0, -1] for electrons, [3.5, 2.5, 1.5, 0.5] for protons
and [5, 4, 3, 2] for carbon. Secondary tables record TB acceptance at
MC-defined working points of 60%, 70%, 80% and 90%.
