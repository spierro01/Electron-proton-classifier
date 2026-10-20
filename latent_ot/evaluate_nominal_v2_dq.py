"""Evaluate nominal conditional OT on selected DQ archives.

PID efficiencies use the original selected energy mixtures. Domain closure
uses newly generated, disjoint, energy-balanced discriminator subsets, shared
before and after OT. TB validation never enters discriminator or OT fitting;
the fixed selection medians were defined on the full TB sample.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

import train_conditional_ot_v2_dq as T
import train_domain_discriminator as D

HERE = Path(__file__).resolve().parent
CLASSES = ['e', 'p', 'C']
NAMES = {'e': 'Electrons', 'p': 'Protons', 'C': 'Carbon'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def load_archive(path, domain, mask, mask_hash):
    metadata = T.check_archive_provenance(path, domain, mask_hash)
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in ['z_enc', 'logits', 'label', 'energy_mev', 'parquet_row']}
        out['classes'] = data['classes'].tolist()
        out['checkpoint'] = str(data['checkpoint_sha256'].item())
        if domain == 'TB':
            out['is_derivation'] = data['is_derivation']
    n = len(out['label'])
    if out['classes'] != CLASSES or out['z_enc'].shape != (n, 64) or out['logits'].shape != (n, 3):
        raise ValueError('Invalid class order or latent/logit shape.')
    rows = out['parquet_row']
    if rows.shape != (n,) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('Invalid original row indices.')
    if (rows < 0).any() or (np.diff(rows) <= 0).any():
        raise ValueError('Original rows must be nonnegative, increasing and unique.')
    total = metadata['evaluation_provenance']['source_files'][domain]['rows']
    if n and rows[-1] >= total:
        raise ValueError('Original row index outside parquet.')
    if out['energy_mev'].shape != (n,) or not np.issubdtype(out['label'].dtype, np.integer):
        raise ValueError('Invalid energies or labels.')
    if ((out['label'] < 0) | (out['label'] >= 3)).any():
        raise ValueError('Invalid class indices.')
    for key in ['z_enc', 'logits', 'energy_mev']:
        if not np.isfinite(out[key]).all():
            raise ValueError(f'Nonfinite archive values: {key}')
    if domain == 'TB':
        if mask.shape != (total,) or out['is_derivation'].dtype != np.bool_:
            raise ValueError('Invalid TB split.')
        np.testing.assert_array_equal(out['is_derivation'], mask[rows])
    out['energy'] = out['energy_mev']
    out['z'] = out['z_enc']
    out['metadata'] = metadata
    return out


def load_pid_head(path, model_source, device):
    # Load the explicitly named model source, avoiding imports from old projects.
    spec = importlib.util.spec_from_file_location('dq_frozen_pid_model', model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES or np.asarray(checkpoint['mean']).shape != (16, 8):
        raise ValueError('Expected the selected no-veto three-class PID checkpoint.')
    keys = ['d_model', 'n_heads', 'n_layers', 'ffn_dim', 'dropout', 'n_features']
    kwargs = {key: checkpoint['args'][key] for key in keys if key in checkpoint['args']}
    model = module.ParticleTransformer(num_classes=3, **kwargs).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def head_logits(model, z, device, batch_size):
    result = np.empty((len(z), 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(z), batch_size):
            stop = min(start + batch_size, len(z))
            x = torch.from_numpy(np.asarray(z[start:stop], dtype=np.float32)).to(device)
            result[start:stop] = model.head_from_latent(x).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite frozen-head logits.')
    return result


def check_maps(args, mc, tb, hashes, mask_hash):
    reports = {}
    for index, cls in enumerate(CLASSES):
        folder = args.maps / cls
        report = json.loads((folder / 'verification.json').read_text())
        config = json.loads((folder / 'config.json').read_text())
        if report.get('reload_invariance_passed') is not True or report.get('TB_validation_used_for_fitting') is not False:
            raise ValueError(f'{cls}: incomplete or invalid map verification.')
        if config['class_name'] != cls or config['classes'] != CLASSES:
            raise ValueError(f'{cls}: map class mismatch.')
        if config['checkpoint_sha256'] != mc['checkpoint'] or config['tb_split_sha256'] != mask_hash:
            raise ValueError(f'{cls}: map checkpoint or split mismatch.')
        if config['latent_archive_sha256'] != hashes:
            raise ValueError(f'{cls}: maps were fitted on different latent archives.')
        common = np.asarray(config['common_energies'])
        expected_mc = np.flatnonzero((mc['label'] == index) & np.isin(mc['energy'], common))
        expected_tb = np.flatnonzero((tb['label'] == index) & tb['is_derivation'] & np.isin(tb['energy'], common))
        with np.load(folder / 'selection.npz', allow_pickle=False) as saved:
            for key, expected in [('mc_rows', expected_mc), ('tb_derivation_rows', expected_tb),
                ('mc_parquet_rows', mc['parquet_row'][expected_mc]),
                ('tb_derivation_parquet_rows', tb['parquet_row'][expected_tb]),
                ('mc_energy', mc['energy'][expected_mc]), ('tb_energy', tb['energy'][expected_tb])]:
                np.testing.assert_array_equal(saved[key], expected)
        reports[cls] = {'config': config, 'map_sha256': T.sha256_file(folder / 'map.pt')}
    print('PASS: map provenance, fitting selections and TB split.', flush=True)
    return reports


def efficiency_record(cls, energy, mc_logits, calibrated_logits, tb_logits, index):
    row = {'cls': cls, 'energy_mev': energy, 'n_MC': len(mc_logits), 'n_TB_validation': len(tb_logits)}
    for key, logits in [('raw_MC', mc_logits), ('nominal_MC', calibrated_logits), ('TB_validation', tb_logits)]:
        correct = int((logits.argmax(axis=1) == index).sum())
        row[f'correct_{key}'] = correct
        row[key] = correct / len(logits) if len(logits) else np.nan
    row['raw_minus_TB_pp'] = 100 * (row['raw_MC'] - row['TB_validation'])
    row['nominal_minus_TB_pp'] = 100 * (row['nominal_MC'] - row['TB_validation'])
    return row


def wilson_errors(p, n):
    center = (p + 0.5 / n) / (1 + 1 / n)
    half = np.sqrt(p * (1 - p) / n + 0.25 / n**2) / (1 + 1 / n)
    return np.maximum(0, np.vstack([p - center + half, center + half - p]))


def plot_confusion(matrix, title, path):
    fractions = 100 * matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.7, 5.7), layout='constrained')
    im = ax.imshow(fractions, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(3), [NAMES[c] for c in CLASSES])
    ax.set_yticks(range(3), [NAMES[c] for c in CLASSES])
    ax.set_xlabel('Predicted class', fontweight='bold', labelpad=10)
    ax.set_ylabel('True class', fontweight='bold', labelpad=10)
    ax.set_title(title, fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:,}\n({fractions[i,j]:.3f}%)', ha='center', va='center',
                    color='white' if fractions[i,j] >= 50 else 'black')
    fig.colorbar(im, ax=ax).set_label('Row-normalised fraction [%]', fontweight='bold')
    for ext in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.' + ext), dpi=220, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def evaluate_efficiencies(args, mc, tb, maps, fingerprint):
    out = args.out / 'efficiency'
    if out.exists():
        raise FileExistsError(f'{out}: preserve existing results.')
    out.mkdir(parents=True)
    write_json(out / 'provenance.json', fingerprint)
    model = load_pid_head(args.ckpt, args.pid_model, args.device)
    baseline = pd.read_csv(args.baseline / 'efficiency_by_energy.csv')
    records, integrated = [], []
    matrices = {key: np.zeros((3, 3), dtype=np.int64) for key in ['raw_MC', 'nominal_MC', 'TB_validation']}
    for index, cls in enumerate(CLASSES):
        common = np.asarray(maps[cls]['config']['common_energies'])
        base = baseline[baseline.cls == cls].sort_values('energy_mev')
        np.testing.assert_array_equal(base.energy_mev.to_numpy(), common)
        mc_rows = np.flatnonzero((mc['label'] == index) & np.isin(mc['energy'], common))
        tb_rows = np.flatnonzero((tb['label'] == index) & ~tb['is_derivation'] & np.isin(tb['energy'], common))
        for data, rows in [(mc, mc_rows), (tb, tb_rows)]:
            actual = head_logits(model, data['z'][rows], args.device, args.batch_size)
            np.testing.assert_allclose(actual, data['logits'][rows], rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(actual.argmax(1), data['logits'][rows].argmax(1))
        print(f'{cls}: frozen head verified; calibrating {len(mc_rows):,} MC events...', flush=True)
        transport = T.ConditionalTransport.load(args.maps / cls / 'map.pt', device=args.device)
        calibrated_z = np.empty((len(mc_rows), 64), dtype=np.float32)
        calibrated_logits = np.empty((len(mc_rows), 3), dtype=np.float32)
        for start in range(0, len(mc_rows), 20_000):
            stop = min(start + 20_000, len(mc_rows))
            rows = mc_rows[start:stop]
            z = transport.transform(mc['z'][rows], mc['energy'][rows], batch_size=args.batch_size)
            calibrated_z[start:stop] = z
            calibrated_logits[start:stop] = head_logits(model, z, args.device, args.batch_size)
            print(f'  {cls}: {stop:,}/{len(mc_rows):,}', flush=True)
        del transport
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        path = out / f'calibrated_MC_{cls}.npz'
        np.savez_compressed(path, z_enc=calibrated_z, logits=calibrated_logits,
            label=mc['label'][mc_rows], energy_mev=mc['energy'][mc_rows],
            latent_row=mc_rows, parquet_row=mc['parquet_row'][mc_rows],
            classes=np.asarray(CLASSES), checkpoint_sha256=np.asarray(mc['checkpoint']),
            map_sha256=np.asarray(maps[cls]['map_sha256']))
        for energy in common:
            a = mc['energy'][mc_rows] == energy
            b = tb['energy'][tb_rows] == energy
            row = efficiency_record(cls, float(energy), mc['logits'][mc_rows[a]], calibrated_logits[a], tb['logits'][tb_rows[b]], index)
            previous = base.loc[base.energy_mev == energy].iloc[0]
            for key, old_key in [('n_MC', 'n_MC'), ('n_TB_validation', 'n_TB'),
                                 ('correct_raw_MC', 'correct_MC'), ('correct_TB_validation', 'correct_TB')]:
                if row[key] != int(previous[old_key]):
                    raise ValueError(f'{cls}@{energy}: baseline mismatch for {key}')
            records.append(row)
        integrated.append(efficiency_record(cls, 'all_common', mc['logits'][mc_rows], calibrated_logits, tb['logits'][tb_rows], index))
        for key, logits in [('raw_MC', mc['logits'][mc_rows]), ('nominal_MC', calibrated_logits), ('TB_validation', tb['logits'][tb_rows])]:
            matrices[key][index] = np.bincount(logits.argmax(axis=1), minlength=3)
        del calibrated_z, calibrated_logits
    table = pd.DataFrame(records)
    table.to_csv(out / 'efficiency_by_energy.csv', index=False)
    pd.DataFrame(integrated).to_csv(out / 'efficiency_integrated.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, cls in zip(axes, CLASSES):
        group = table[table.cls == cls].sort_values('energy_mev')
        for key, label, color, marker in [('raw_MC', 'MC original', 'tab:blue', 'o'),
                ('nominal_MC', 'MC calibrated (OT)', 'tab:green', 's'),
                ('TB_validation', 'TB validation', 'black', 'o')]:
            p = group[key].to_numpy()
            n = group['n_TB_validation' if key == 'TB_validation' else 'n_MC'].to_numpy()
            ax.errorbar(group.energy_mev, 100*p, yerr=100*wilson_errors(p, n), label=label, color=color, marker=marker, capsize=3)
        ax.set_title(NAMES[cls], fontweight='bold')
        ax.set_xlabel('Beam energy [MeV/n]' if cls == 'C' else 'Beam energy [MeV]', fontweight='bold')
        ax.set_ylabel('PID efficiency [%]', fontweight='bold')
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Particle identification efficiency before and after OT', fontweight='bold')
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'efficiency_vs_energy.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    for key, matrix in matrices.items():
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(out / f'confusion_{key}.csv')
        plot_confusion(matrix, {'raw_MC': 'MC original', 'nominal_MC': 'MC calibrated (OT)', 'TB_validation': 'TB validation'}[key], out / f'confusion_{key}')
    write_json(out / 'completion.json', {'completed': True, 'raw_counts_match_baseline': True,
        'frozen_head_verified': True, 'TB_metrics_validation_only': True,
        'uncertainty': 'Wilson z=1 counting intervals; OT map uncertainty not included',
        'calibrated_archives_sha256': {c: T.sha256_file(out / f'calibrated_MC_{c}.npz') for c in CLASSES}})
    print('\nIntegrated efficiencies:\n', pd.DataFrame(integrated).to_string(index=False), flush=True)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()


def validate_split(selected, mc, tb, index, common):
    for domain, data in [('mc', mc), ('tb', tb)]:
        for split in ['train', 'val', 'test']:
            rows = selected[f'{domain}_{split}']
            if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer):
                raise ValueError('Invalid discriminator indices.')
            if (rows < 0).any() or (rows >= len(data['label'])).any() or len(np.unique(rows)) != len(rows):
                raise ValueError('Duplicate or out-of-range discriminator indices.')
            if not (data['label'][rows] == index).all() or not np.isin(data['energy'][rows], common).all():
                raise ValueError('Wrong class or energy in discriminator subset.')
        for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if np.intersect1d(selected[f'{domain}_{first}'], selected[f'{domain}_{second}']).size:
                raise ValueError('Overlapping discriminator subsets.')
    for split in ['train', 'val']:
        if not tb['is_derivation'][selected[f'tb_{split}']].all():
            raise ValueError('TB validation used in discriminator fitting/model selection.')
    if tb['is_derivation'][selected['tb_test']].any():
        raise ValueError('TB derivation used in final discriminator test.')
    for split in ['train', 'val', 'test']:
        counts = [int(np.sum(data['energy'][selected[f'{domain}_{split}']] == energy))
                  for domain, data in [('mc', mc), ('tb', tb)] for energy in common]
        if min(counts) < 1 or len(set(counts)) != 1:
            raise ValueError('Discriminator energies/domains are not equally balanced.')


def discriminator_data(mc, tb, cal, lookup, selected, split, representation, calibrated):
    mc_rows, tb_rows = selected[f'mc_{split}'], selected[f'tb_{split}']
    if calibrated:
        positions = lookup[mc_rows]
        if (positions < 0).any():
            raise ValueError('Missing calibrated MC events.')
        source = cal[representation][positions]
    else:
        source = mc[representation][mc_rows]
    target = tb[representation][tb_rows]
    x = np.concatenate([source, target]).astype(np.float32)
    y = np.concatenate([np.zeros(len(source), dtype=np.int64), np.ones(len(target), dtype=np.int64)])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite discriminator input.')
    return x, y


def evaluate_closure(args, mc, tb, maps, fingerprint):
    efficiency = args.out / 'efficiency'
    completion = json.loads((efficiency / 'completion.json').read_text())
    if completion.get('completed') is not True:
        raise ValueError('Efficiency evaluation must complete before closure.')
    if json.loads((efficiency / 'provenance.json').read_text()) != fingerprint:
        raise ValueError('Efficiency inputs differ from the current closure inputs.')
    out = args.out / 'closure'
    if out.exists():
        raise FileExistsError(f'{out}: preserve existing closure results.')
    out.mkdir()
    write_json(out / 'provenance.json', fingerprint)
    shutil.copyfile(D.__file__, out / 'discriminator_source.py.txt')
    summaries, histories, inventories = [], [], []
    figures = {rep: plt.subplots(1, 3, figsize=(15, 5), layout='constrained') for rep in ['z_enc', 'logits']}
    for index, cls in enumerate(CLASSES):
        seed = D.SEED + index
        selected, common, inventory = D.select_rows(mc, tb, cls, np.random.default_rng(seed))
        np.testing.assert_array_equal(common, np.asarray(maps[cls]['config']['common_energies']))
        validate_split(selected, mc, tb, index, common)
        inventories.extend(inventory)
        original_indices = {f'{domain}_{split}_parquet_row': data['parquet_row'][selected[f'{domain}_{split}']]
            for domain, data in [('mc', mc), ('tb', tb)] for split in ['train', 'val', 'test']}
        np.savez(out / f'split_{cls}.npz', **selected, **original_indices, common_energies=common,
                 classes=np.asarray(CLASSES), checkpoint_sha256=np.asarray(mc['checkpoint']), seed=seed)
        path = efficiency / f'calibrated_MC_{cls}.npz'
        if T.sha256_file(path) != completion['calibrated_archives_sha256'][cls]:
            raise ValueError('Calibrated archive changed after efficiency evaluation.')
        with np.load(path, allow_pickle=False) as saved:
            rows = saved['latent_row']
            expected = np.flatnonzero((mc['label'] == index) & np.isin(mc['energy'], common))
            np.testing.assert_array_equal(rows, expected)
            np.testing.assert_array_equal(saved['parquet_row'], mc['parquet_row'][rows])
            np.testing.assert_array_equal(saved['label'], mc['label'][rows])
            np.testing.assert_array_equal(saved['energy_mev'], mc['energy'][rows])
            if saved['classes'].tolist() != CLASSES or saved['checkpoint_sha256'].item() != mc['checkpoint'] or saved['map_sha256'].item() != maps[cls]['map_sha256']:
                raise ValueError('Calibrated archive provenance mismatch.')
            cal = {key: saved[key] for key in ['z_enc', 'logits']}
        lookup = np.full(len(mc['label']), -1, dtype=np.int64)
        lookup[rows] = np.arange(len(rows))
        for rep in ['z_enc', 'logits']:
            for status, calibrated in [('raw', False), ('calibrated', True)]:
                tag = f'{cls}_{rep}_{status}'
                print(f'\n{tag}: fitting domain discriminator...', flush=True)
                x_train, y_train = discriminator_data(mc, tb, cal, lookup, selected, 'train', rep, calibrated)
                x_val, y_val = discriminator_data(mc, tb, cal, lookup, selected, 'val', rep, calibrated)
                order = np.random.default_rng(seed).permutation(len(y_train))
                x_train, y_train = x_train[order], y_train[order]
                candidate = D.new_model(seed, warm_start=True)
                best_auc, best_iterations = -np.inf, None
                for iterations in D.ITERATION_CANDIDATES:
                    candidate.set_params(max_iter=iterations)
                    candidate.fit(x_train, y_train)
                    auc = float(roc_auc_score(y_val, candidate.predict_proba(x_val)[:, 1]))
                    histories.append(dict(cls=cls, representation=rep, status=status, iterations=iterations, internal_validation_auc=auc))
                    print(f'  iterations={iterations}: internal AUC={auc:.6f}', flush=True)
                    if auc > best_auc:
                        best_auc, best_iterations = auc, iterations
                model = D.new_model(seed, iterations=best_iterations)
                model.fit(x_train, y_train)
                x_test, y_test = discriminator_data(mc, tb, cal, lookup, selected, 'test', rep, calibrated)
                scores = model.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, scores))
                fpr, tpr, thresholds = roc_curve(y_test, scores)
                joblib.dump(model, out / f'{tag}.joblib')
                reloaded = joblib.load(out / f'{tag}.joblib')
                np.testing.assert_allclose(reloaded.predict_proba(x_test)[:, 1], scores, rtol=0, atol=0)
                np.savez(out / f'{tag}_evaluation.npz', domain_label=y_test, score_tb=scores,
                         fpr=fpr, tpr=tpr, thresholds=thresholds,
                         mc_test_rows=selected['mc_test'], tb_test_rows=selected['tb_test'],
                         mc_test_parquet_rows=mc['parquet_row'][selected['mc_test']],
                         tb_test_parquet_rows=tb['parquet_row'][selected['tb_test']])
                summaries.append(dict(cls=cls, representation=rep, status=status, selected_iterations=best_iterations,
                    internal_validation_auc=best_auc, heldout_auc=auc, n_test_MC=len(selected['mc_test']), n_test_TB=len(selected['tb_test'])))
                pd.DataFrame(summaries).to_csv(out / 'auc_summary.csv', index=False)
                pd.DataFrame(histories).to_csv(out / 'training_history.csv', index=False)
                label = 'MC calibrated (OT)' if calibrated else 'MC original'
                figures[rep][1][index].plot(fpr, tpr, label=f'{label}: AUC={auc:.4f}', color='tab:green' if calibrated else 'tab:blue')
                print(f'  FINAL TEST AUC={auc:.6f}', flush=True)
                del candidate, model, reloaded, x_train, x_val, x_test
        del cal, lookup
    for rep, (fig, axes) in figures.items():
        for ax, cls in zip(axes, CLASSES):
            ax.plot([0, 1], [0, 1], '--', color='gray', label='Random reference')
            ax.set_title(NAMES[cls], fontweight='bold')
            ax.set_xlabel('False positive rate', fontweight='bold')
            ax.set_ylabel('True positive rate', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend(loc='lower right', fontsize=8)
        fig.suptitle(f'MC–TB discrimination before and after OT: {rep}', fontweight='bold')
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'roc_{rep}.{ext}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    comparison = pd.DataFrame(summaries).pivot(index=['cls', 'representation'], columns='status', values='heldout_auc').reset_index()
    comparison['auc_raw_minus_calibrated'] = comparison['raw'] - comparison['calibrated']
    comparison.to_csv(out / 'auc_comparison.csv', index=False)
    pd.DataFrame(inventories).to_csv(out / 'sample_counts.csv', index=False)
    write_json(out / 'config.json', {
        'seed': D.SEED, 'iteration_candidates': D.ITERATION_CANDIDATES,
        'model_parameters': D.new_model(D.SEED).get_params(),
        'per_energy_caps': {'train': D.TRAIN_PER_ENERGY, 'val': D.VALIDATION_PER_ENERGY, 'test': D.TEST_PER_ENERGY},
        'domain_labels': {'MC': 0, 'TB': 1}, 'row_indices': 'selected NPZ array indices; original parquet indices also saved',
        'TB_usage': 'derivation for discriminator training/internal validation; validation for final test',
        'energy_mixtures': 'equal counts per common energy and domain in each subset',
        'limitations': ['MC test events were available to OT fitting, but not discriminator fitting.',
            'TB final-test events were excluded from OT/discriminator fitting but included in fixed selection medians.',
            'AUC is a point estimate and depends on discriminator capacity and sample size.',
            'AUC near 0.5 is not proof of distributional equality.',
            'Random event splits do not test generalisation to unseen runs.']})
    write_json(out / 'completion.json', {'completed': True, 'n_discriminators': len(summaries), 'identical_before_after_splits': True,
        'TB_final_test_validation_only': True, 'discriminator_reload_verified': True})
    print('\nAUC comparison:\n', comparison.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mc', type=Path, default=HERE / 'derived_v2_dq/latents_MC.npz')
    parser.add_argument('--tb', type=Path, default=HERE / 'derived_v2_dq/latents_TB.npz')
    parser.add_argument('--mask', type=Path, default=HERE / 'derived_v2/tb_is_derivation_v2_dq.npy')
    parser.add_argument('--pid-model', type=Path, default=HERE.parent / 'transformer_no_veto_v2/pid_model.py')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--stage', choices=['all', 'efficiency', 'closure'], default='all')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Batch size and thread count must be positive.')
    if args.stage != 'closure' and args.out.exists():
        raise FileExistsError(f'{args.out}: use a new output directory.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    torch.set_num_threads(args.threads)
    mask = np.load(args.mask, allow_pickle=False)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError('Invalid TB split.')
    mask_hash = T.sha256_file(args.mask)
    hashes = {'MC': T.sha256_file(args.mc), 'TB': T.sha256_file(args.tb)}
    mc = load_archive(args.mc, 'MC', mask, mask_hash)
    tb = load_archive(args.tb, 'TB', mask, mask_hash)
    if mc['checkpoint'] != tb['checkpoint'] or T.sha256_file(args.ckpt) != mc['checkpoint']:
        raise ValueError('PID checkpoint mismatch.')
    if mc['metadata']['evaluation_provenance'] != tb['metadata']['evaluation_provenance']:
        raise ValueError('MC and TB have different selection provenance.')
    if json.loads((args.baseline / 'provenance.json').read_text()) != mc['metadata']['evaluation_provenance']:
        raise ValueError('Baseline evaluation differs from latent extraction.')
    if json.loads((args.baseline / 'completion.json').read_text()).get('completed') is not True:
        raise ValueError('Baseline evaluation is incomplete.')
    maps = check_maps(args, mc, tb, hashes, mask_hash)
    fingerprint = {'checkpoint_sha256': mc['checkpoint'], 'latent_archive_sha256': hashes,
        'mask_sha256': mask_hash, 'map_sha256': {c: maps[c]['map_sha256'] for c in CLASSES},
        'baseline_provenance_sha256': T.sha256_file(args.baseline / 'provenance.json'),
        'baseline_table_sha256': T.sha256_file(args.baseline / 'efficiency_by_energy.csv'),
        'model_source_sha256': T.sha256_file(args.pid_model),
        'discriminator_source_sha256': T.sha256_file(D.__file__),
        'evaluation_script_sha256': T.sha256_file(__file__)}
    if args.stage in ['all', 'efficiency']:
        evaluate_efficiencies(args, mc, tb, maps, fingerprint)
    if args.stage in ['all', 'closure']:
        with threadpool_limits(limits=args.threads):
            evaluate_closure(args, mc, tb, maps, fingerprint)
    if T.sha256_file(args.mask) != mask_hash:
        raise RuntimeError('TB split changed during evaluation.')
    print('\nSaved:', args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
