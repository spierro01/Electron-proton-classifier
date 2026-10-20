"""Evaluate one LOEO map at its excluded energy against fixed nominal results."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import evaluate_nominal_v2_dq as E
import train_conditional_ot_v2_dq as T

HERE = Path(__file__).resolve().parent


def verify_fitting_rows(saved, mc, tb, index, retained, excluded):
    expected_mc = np.flatnonzero((mc['label'] == index) & np.isin(mc['energy'], retained))
    expected_tb = np.flatnonzero((tb['label'] == index) & tb['is_derivation'] & np.isin(tb['energy'], retained))
    for key, value in [('mc_rows', expected_mc), ('tb_derivation_rows', expected_tb),
                       ('mc_parquet_rows', mc['parquet_row'][expected_mc]),
                       ('tb_derivation_parquet_rows', tb['parquet_row'][expected_tb]),
                       ('mc_energy', mc['energy'][expected_mc]), ('tb_energy', tb['energy'][expected_tb])]:
        np.testing.assert_array_equal(saved[key], value)
    for key in ['mc_energy', 'tb_energy']:
        if np.isclose(saved[key], excluded, rtol=0, atol=1e-4).any():
            raise ValueError('Held-out energy present in fitting selection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-name', choices=E.CLASSES, required=True)
    parser.add_argument('--energy', type=float, required=True)
    parser.add_argument('--loeo-dir', type=Path, required=True)
    parser.add_argument('--nominal-root', type=Path, required=True)
    parser.add_argument('--nominal-evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mc', type=Path, default=HERE / 'derived_v2_dq/latents_MC.npz')
    parser.add_argument('--tb', type=Path, default=HERE / 'derived_v2_dq/latents_TB.npz')
    parser.add_argument('--mask', type=Path, default=HERE / 'derived_v2/tb_is_derivation_v2_dq.npy')
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'{args.out}: use a new evaluation directory.')
    if args.batch_size < 1:
        raise ValueError('Invalid batch size.')
    torch.set_num_threads(8)
    index = E.CLASSES.index(args.class_name)
    config = json.loads((args.loeo_dir / 'config.json').read_text())
    verification = json.loads((args.loeo_dir / 'verification.json').read_text())
    nominal = json.loads((args.nominal_root / args.class_name / 'config.json').read_text())
    if verification.get('reload_invariance_passed') is not True or config.get('smoke_test') is not False:
        raise ValueError('A verified full LOEO training run is required.')
    if config.get('analysis') != 'LOEO' or config['class_name'] != args.class_name:
        raise ValueError('Wrong map type or particle class.')
    excluded = config['excluded_energy']
    if not np.isclose(excluded, args.energy, rtol=0, atol=1e-4):
        raise ValueError('Requested energy differs from the LOEO exclusion.')
    for key in ['architecture', 'batch_size', 'outer_steps', 'identity_steps', 'seed', 'checkpoint_sha256',
                'tb_split_sha256', 'latent_archive_sha256', 'g_updates', 'f_updates', 'lr', 'min_lr', 'betas', 'weight_decay', 'grad_clip']:
        if config[key] != nominal[key]:
            raise ValueError(f'LOEO/nominal configuration mismatch: {key}')
    nominal_hash = T.sha256_file(args.nominal_root / args.class_name / 'map.pt')
    if config['nominal_map_sha256'] != nominal_hash:
        raise ValueError('Nominal map fingerprint mismatch.')
    full = np.asarray(nominal['common_energies'])
    retained = full[~np.isclose(full, excluded, rtol=0, atol=1e-4)]
    np.testing.assert_array_equal(config['common_energies'], retained)
    mask_hash = T.sha256_file(args.mask)
    if mask_hash != config['tb_split_sha256']:
        raise ValueError('TB split fingerprint mismatch.')
    hashes = {'MC': T.sha256_file(args.mc), 'TB': T.sha256_file(args.tb)}
    if hashes != config['latent_archive_sha256'] or T.sha256_file(args.ckpt) != config['checkpoint_sha256']:
        raise ValueError('Archive or PID checkpoint mismatch.')
    mask = np.load(args.mask, allow_pickle=False)
    mc = E.load_archive(args.mc, 'MC', mask, mask_hash)
    tb = E.load_archive(args.tb, 'TB', mask, mask_hash)
    with np.load(args.loeo_dir / 'selection.npz', allow_pickle=False) as saved:
        verify_fitting_rows(saved, mc, tb, index, retained, excluded)
    print('PASS: nominal/LOEO settings, excluded fitting energy and TB split.', flush=True)
    mc_rows = np.flatnonzero((mc['label'] == index) & np.isclose(mc['energy'], excluded, rtol=0, atol=1e-4))
    tb_rows = np.flatnonzero((tb['label'] == index) & ~tb['is_derivation'] & np.isclose(tb['energy'], excluded, rtol=0, atol=1e-4))
    if not len(mc_rows) or not len(tb_rows):
        raise ValueError('No held-out MC or TB validation events.')
    efficiency_dir = args.nominal_evaluation / 'efficiency'
    completion = json.loads((efficiency_dir / 'completion.json').read_text())
    provenance = json.loads((efficiency_dir / 'provenance.json').read_text())
    if completion.get('completed') is not True or provenance['latent_archive_sha256'] != hashes:
        raise ValueError('Nominal evaluation is incomplete or uses other archives.')
    if provenance['map_sha256'][args.class_name] != nominal_hash:
        raise ValueError('Nominal evaluation uses another map.')
    calibrated_path = efficiency_dir / f'calibrated_MC_{args.class_name}.npz'
    if T.sha256_file(calibrated_path) != completion['calibrated_archives_sha256'][args.class_name]:
        raise ValueError('Nominal calibrated archive fingerprint mismatch.')
    with np.load(calibrated_path, allow_pickle=False) as saved:
        keep = np.isclose(saved['energy_mev'], excluded, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(saved['latent_row'][keep], mc_rows)
        np.testing.assert_array_equal(saved['parquet_row'][keep], mc['parquet_row'][mc_rows])
        np.testing.assert_array_equal(saved['label'][keep], mc['label'][mc_rows])
        nominal_logits = saved['logits'][keep]
    model = E.load_pid_head(args.ckpt, HERE.parent / 'transformer_no_veto_v2/pid_model.py', args.device)
    raw_logits = mc['logits'][mc_rows]
    actual = E.head_logits(model, mc['z'][mc_rows], args.device, args.batch_size)
    np.testing.assert_allclose(actual, raw_logits, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual.argmax(1), raw_logits.argmax(1))
    transport = T.ConditionalTransport.load(args.loeo_dir / 'map.pt', device=args.device)
    loeo_logits = np.empty((len(mc_rows), 3), dtype=np.float32)
    for start in range(0, len(mc_rows), 20_000):
        stop = min(start + 20_000, len(mc_rows))
        rows = mc_rows[start:stop]
        z = transport.transform(mc['z'][rows], mc['energy'][rows], batch_size=args.batch_size)
        loeo_logits[start:stop] = E.head_logits(model, z, args.device, args.batch_size)
        print(f'LOEO {args.class_name}@{excluded:g}: {stop:,}/{len(mc_rows):,}', flush=True)
    record = {'cls': args.class_name, 'energy_mev': excluded, 'type': config['loeo_type'],
              'n_MC': len(mc_rows), 'n_TB_validation': len(tb_rows)}
    for key, logits in [('raw_MC', raw_logits), ('nominal_MC', nominal_logits), ('LOEO_MC', loeo_logits), ('TB_validation', tb['logits'][tb_rows])]:
        correct = int((logits.argmax(1) == index).sum())
        record[f'correct_{key}'] = correct
        record[key] = correct / len(logits)
    record['LOEO_minus_nominal_pp'] = 100*(record['LOEO_MC'] - record['nominal_MC'])
    record['LOEO_minus_TB_pp'] = 100*(record['LOEO_MC'] - record['TB_validation'])
    baseline = pd.read_csv(efficiency_dir / 'efficiency_by_energy.csv')
    previous = baseline[(baseline.cls == args.class_name) & np.isclose(baseline.energy_mev, excluded, rtol=0, atol=1e-4)]
    if len(previous) != 1:
        raise ValueError('Nominal baseline row missing or ambiguous.')
    for key in ['n_MC', 'n_TB_validation', 'correct_raw_MC', 'correct_nominal_MC', 'correct_TB_validation']:
        if record[key] != int(previous.iloc[0][key]):
            raise ValueError(f'Held-out baseline mismatch: {key}')
    if T.sha256_file(args.mask) != mask_hash:
        raise RuntimeError('TB split changed during evaluation.')
    args.out.mkdir(parents=True)
    pd.DataFrame([record]).to_csv(args.out / 'efficiency_comparison.csv', index=False)
    np.savez_compressed(args.out / 'heldout_predictions.npz',
        mc_latent_row=mc_rows, mc_parquet_row=mc['parquet_row'][mc_rows],
        tb_latent_row=tb_rows, tb_parquet_row=tb['parquet_row'][tb_rows],
        raw_logits=raw_logits, nominal_logits=nominal_logits, loeo_logits=loeo_logits,
        tb_logits=tb['logits'][tb_rows], classes=np.asarray(E.CLASSES))
    E.write_json(args.out / 'completion.json', {'completed': True, 'excluded_energy_verified': True,
        'identical_MC_events': True, 'TB_validation_only': True, 'mask_unchanged': True,
        'loeo_map_sha256': T.sha256_file(args.loeo_dir / 'map.pt'), 'nominal_map_sha256': nominal_hash,
        'latent_archive_sha256': hashes, 'checkpoint_sha256': mc['checkpoint'],
        'TB_full_sample_used_for_fixed_selection_medians': True, 'metrics': record})
    print(pd.DataFrame([record]).to_string(index=False), flush=True)
    print('Saved:', args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
