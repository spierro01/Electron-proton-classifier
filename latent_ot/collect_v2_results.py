"""Collect the v2 analysis reports and generate English summary figures.

Run on the remote host. Source results are read-only. Raw datasets, checkpoints,
event-level predictions and latent arrays are intentionally not duplicated.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['e', 'p', 'C']
NAMES = {'e': 'Electrons', 'p': 'Protons', 'C': 'Carbon'}
SUFFIXES = {'.csv', '.json', '.png', '.pdf', '.svg', '.log', '.txt', '.md'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(path.with_suffix('.' + ext), dpi=220, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def wilson(p, n):
    center = (p + .5/n) / (1 + 1/n)
    half = np.sqrt(p*(1-p)/n + .25/n**2) / (1 + 1/n)
    return 100*np.maximum(0, np.vstack([p-center+half, center+half-p]))


def plot_matrix(matrix, title, path):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (3, 3) or (matrix < 0).any() or (matrix.sum(1) == 0).any():
        raise ValueError(f'Invalid confusion matrix: {title}')
    fraction = matrix / matrix.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.8, 5.7), layout='constrained')
    im = ax.imshow(fraction, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(3), [NAMES[c] for c in CLASSES])
    ax.set_yticks(range(3), [NAMES[c] for c in CLASSES])
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted class', fontweight='bold', labelpad=10)
    ax.set_ylabel('True class', fontweight='bold', labelpad=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:,}\n({fraction[i,j]:.3f}%)', ha='center', va='center',
                    color='white' if fraction[i,j] >= 50 else 'black')
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label('Row-normalised fraction [%]', fontweight='bold')
    save_figure(fig, path)


def plot_efficiency(table, path, calibrated=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    methods = [('efficiency_MC', 'n_MC', 'MC', 'tab:blue'), ('efficiency_TB', 'n_TB', 'TB validation', 'black')]
    if calibrated:
        methods = [('raw_MC', 'n_MC', 'MC original', 'tab:blue'),
                   ('nominal_MC', 'n_MC', 'MC calibrated (OT)', 'tab:green'),
                   ('TB_validation', 'n_TB_validation', 'TB validation', 'black')]
    for ax, cls in zip(axes, CLASSES):
        group = table[table.cls == cls].sort_values('energy_mev')
        for key, count, label, color in methods:
            p, n = group[key].to_numpy(), group[count].to_numpy()
            if (n <= 0).any():
                raise ValueError('Cannot plot an empty energy setting.')
            ax.errorbar(group.energy_mev, 100*p, yerr=wilson(p, n), marker='s' if key == 'nominal_MC' else 'o',
                        color=color, label=label, capsize=3)
        ax.set_title(NAMES[cls], fontweight='bold')
        ax.set_xlabel('Beam energy [MeV/n]' if cls == 'C' else 'Beam energy [MeV]', fontweight='bold')
        ax.set_ylabel('PID efficiency [%]', fontweight='bold')
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(methods), frameon=False)
    fig.suptitle('Particle identification efficiency before and after OT' if calibrated else 'Particle identification efficiency: MC vs TB', fontweight='bold')
    fig.tight_layout(rect=(0, .09, 1, .94))
    save_figure(fig, path)


def display_table(table, path, calibrated=False):
    rows = []
    for cls in CLASSES:
        for _, row in table[table.cls == cls].sort_values('energy_mev').iterrows():
            energy = f'{row.energy_mev:g} ' + ('MeV/n' if cls == 'C' else 'MeV')
            values = [NAMES[cls], energy, f'{int(row.n_MC):,}']
            if calibrated:
                values += [f'{int(row.n_TB_validation):,}', f'{100*row.raw_MC:.4f}',
                           f'{100*row.nominal_MC:.4f}', f'{100*row.TB_validation:.4f}', f'{row.nominal_minus_TB_pp:+.4f}']
            else:
                values += [f'{int(row.n_TB):,}', f'{100*row.efficiency_MC:.4f}',
                           f'{100*row.efficiency_TB:.4f}', f'{row.MC_minus_TB_pp:+.4f}']
            rows.append(values)
    columns = ['Species', 'Beam energy', 'MC events', 'TB val. events', 'MC [%]']
    columns += ['MC OT [%]', 'TB val. [%]', 'OT - TB [pp]'] if calibrated else ['TB val. [%]', 'MC - TB [pp]']
    pd.DataFrame(rows, columns=columns).to_csv(path.with_suffix('.csv'), index=False)
    fig, ax = plt.subplots(figsize=(14, max(4, .34 * (len(rows) + 1) + 1.4)))
    ax.axis('off')
    grid = ax.table(cellText=rows, colLabels=columns, cellLoc='center', bbox=[0, .05, 1, .88])
    grid.auto_set_font_size(False)
    grid.set_fontsize(9)
    grid.scale(1, 1.6)
    for (r, c), cell in grid.get_celld().items():
        if r == 0:
            cell.set_facecolor('#16385b')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#eef3f8' if r % 2 else 'white')
    ax.set_title('Identification efficiencies by beam energy', fontweight='bold', pad=15)
    fig.text(.5, .02, 'TB refers to validation events. Efficiencies are percentages; differences are percentage points.', ha='center', fontsize=9)
    save_figure(fig, path)


def copy_reports(source, destination, missing, skip_figures=False):
    if not source.is_dir():
        missing.append(str(source))
        return
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.startswith('predictions_') or '.backup_' in path.name:
            continue
        allowed = path.suffix.lower() in SUFFIXES
        allowed |= path.name == 'selection.npz' or (path.name.startswith('split_') and path.suffix == '.npz')
        if not allowed or (skip_figures and path.suffix.lower() in {'.png', '.pdf', '.svg'}):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def baseline_figures(folder):
    table = pd.read_csv(folder / 'efficiency_by_energy.csv')
    plot_efficiency(table, folder / 'efficiency_vs_energy')
    display_table(table, folder / 'efficiency_table')
    for domain in ['MC', 'TB']:
        path = folder / f'confusion_{domain}.csv'
        if path.is_file():
            matrix = pd.read_csv(path, index_col=0).loc[CLASSES, CLASSES].to_numpy()
            plot_matrix(matrix, 'MC' if domain == 'MC' else 'TB validation', folder / f'confusion_{domain}')


def summary_figures(folder, summary):
    table = pd.read_csv(folder / 'efficiency/efficiency_by_energy.csv')
    display_table(table, summary / 'efficiency_table_after_OT', calibrated=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, cls in zip(axes, CLASSES):
        group = table[table.cls == cls].sort_values('energy_mev')
        ax.plot(group.energy_mev, group.raw_minus_TB_pp, 'o-', color='tab:blue', label='MC original - TB')
        ax.plot(group.energy_mev, group.nominal_minus_TB_pp, 's-', color='tab:green', label='MC calibrated - TB')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(NAMES[cls], fontweight='bold')
        ax.set_xlabel('Beam energy [MeV/n]' if cls == 'C' else 'Beam energy [MeV]', fontweight='bold')
        ax.set_ylabel('Efficiency difference [pp]', fontweight='bold')
        ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('MC–TB efficiency differences before and after OT', fontweight='bold')
    fig.tight_layout(rect=(0, .09, 1, .94))
    save_figure(fig, summary / 'efficiency_difference_vs_energy')
    auc = pd.read_csv(folder / 'closure/auc_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, rep, title in zip(axes, ['logits', 'z_enc'], ['Classifier logits', 'Encoder latents']):
        group = auc[auc.representation == rep].set_index('cls').loc[CLASSES]
        x = np.arange(3)
        ax.bar(x-.18, group.raw, .36, label='MC original', color='tab:blue')
        ax.bar(x+.18, group.calibrated, .36, label='MC calibrated (OT)', color='tab:green')
        for i, (_, row) in enumerate(group.iterrows()):
            ax.text(i-.18, row.raw+.012, f'{row.raw:.4f}', ha='center', fontsize=8)
            ax.text(i+.18, row.calibrated+.012, f'{row.calibrated:.4f}', ha='center', fontsize=8)
        ax.axhline(.5, color='gray', linestyle='--', label='Random reference')
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x, [NAMES[c] for c in CLASSES])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Domain-discriminator AUC', fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('MC–TB discrimination before and after OT', fontweight='bold')
    fig.tight_layout(rect=(0, .10, 1, .94))
    save_figure(fig, summary / 'auc_summary')
    shutil.copy2(folder / 'efficiency/efficiency_integrated.csv', summary / 'efficiency_integrated.csv')
    shutil.copy2(folder / 'closure/auc_comparison.csv', summary / 'auc_comparison.csv')


def plot_training(maps, destination):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax, cls in zip(axes, CLASSES):
        frame = pd.read_csv(maps / cls / 'training_log.csv')
        ax.plot(frame.outer_step, frame.mean_l2_standardised)
        ax.set_title(NAMES[cls], fontweight='bold')
        ax.set_xlabel('OT iteration', fontweight='bold')
        ax.set_ylabel('Mean displacement (standardised latents)', fontweight='bold')
        ax.grid(alpha=.25)
    fig.suptitle('Nominal OT training: displacement', fontweight='bold')
    save_figure(fig, destination / 'training_displacement')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout='constrained')
    for i, cls in enumerate(CLASSES):
        frame = pd.read_csv(maps / cls / 'training_log.csv')
        for j, key in enumerate(['loss_g', 'loss_f']):
            axes[j, i].plot(frame.outer_step, frame[key])
            axes[j, i].set_title(f'{NAMES[cls]}: {key}', fontweight='bold')
            axes[j, i].set_xlabel('OT iteration', fontweight='bold')
            axes[j, i].set_ylabel('Logged objective', fontweight='bold')
            axes[j, i].grid(alpha=.25)
    fig.suptitle('Nominal OT training objectives', fontweight='bold')
    save_figure(fig, destination / 'training_objectives')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    repo = args.repo.resolve()
    transformer = repo / 'transformer_no_veto_v2'
    latent = repo / 'latent_ot'
    evaluation = latent / 'results/nominal_v2_dq_evaluation_20261015_171535'
    maps = latent / 'results/nominal_v2_dq_20261015_115613'
    baseline = transformer / 'results/task0_selected_v2_20261013_182000'
    dq_baseline = transformer / 'results/task0_selected_v2_dq_20261015_113235'
    for path in [evaluation / 'efficiency/completion.json', evaluation / 'closure/completion.json']:
        if json.loads(path.read_text()).get('completed') is not True:
            raise ValueError(f'Incomplete evaluation: {path}')
    exports = repo / 'exports'
    exports.mkdir(exist_ok=True)
    out = exports / ('Transformer_OT_v2_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    missing = []
    print('Collecting reports into:', out, flush=True)
    classifier = out / '01_Transformer_selected_spectra'
    copy_reports(transformer / 'derived/pid_spectra_v2_no_veto_selected', classifier / 'training_data_reports', missing, skip_figures=True)
    copy_reports(transformer / 'checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104', classifier / 'checkpoint_reports', missing, skip_figures=True)
    checkpoint_path = transformer / 'checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104/best.pt'
    if checkpoint_path.is_file():
        import torch
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        checkpoint_report = {
            'source': str(checkpoint_path), 'sha256': sha256(checkpoint_path),
            'classes': checkpoint_data['classes'], 'training_arguments': checkpoint_data['args'],
            'best_validation_macro_f1': checkpoint_data['val_macro_f1'],
            'normalisation_mean': checkpoint_data['mean'], 'normalisation_std': checkpoint_data['std'],
        }
        (classifier / 'classifier_configuration.json').write_text(json.dumps(checkpoint_report, indent=2), encoding='utf-8')
        del checkpoint_data
    metric_path = classifier / 'training_data_reports/test_metrics.json'
    if metric_path.is_file():
        metric = json.loads(metric_path.read_text())
        if metric.get('classes') != CLASSES:
            raise ValueError('Unexpected classifier test class order.')
        plot_matrix(metric['confusion'], 'Transformer: selected spectra test sample', classifier / 'confusion_spectra_test')
        pd.DataFrame([{
            'species': NAMES[c], 'n_test': int(np.sum(metric['confusion'][i])),
            'correct': metric['confusion'][i][i],
            'efficiency_percent': 100*metric['confusion'][i][i]/np.sum(metric['confusion'][i]),
        } for i, c in enumerate(CLASSES)]).to_csv(classifier / 'test_efficiency_by_class.csv', index=False)
    for source, name in [(baseline, '02_MC_TB_before_DQ'), (dq_baseline, '03_MC_TB_after_DQ_before_OT')]:
        destination = out / name
        copy_reports(source, destination, missing, skip_figures=True)
        if (destination / 'efficiency_by_energy.csv').is_file():
            baseline_figures(destination)
    copy_reports(maps, out / '04_Nominal_OT_training', missing)
    plot_training(maps, out / '04_Nominal_OT_training')
    copy_reports(evaluation, out / '05_Nominal_OT_evaluation', missing)
    summary = out / '00_Summary'
    summary.mkdir()
    summary_figures(evaluation, summary)
    scripts = out / '06_Code'
    scripts.mkdir()
    for path in sorted(transformer.glob('*.py')):
        target = scripts / 'transformer_no_veto_v2' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    wanted = {'conditional_ot_model.py', 'train_ot.py', 'train_domain_discriminator.py', 'collect_v2_results.py'}
    for path in sorted(latent.iterdir()):
        if path.is_file() and (path.name in wanted or (path.suffix in {'.py', '.sh'} and 'v2_dq' in path.stem)):
            target = scripts / 'latent_ot' / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    extraction = out / '07_Latent_verification'
    extraction.mkdir()
    for path in sorted((latent / 'derived_v2_dq').glob('*.verification.json')):
        shutil.copy2(path, extraction / path.name)
    for path in sorted((transformer / 'logs').glob('*selected*.log')):
        target = classifier / 'logs' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    for base in [transformer, latent]:
        for path in sorted((base / 'logs').glob('*v2_dq*.log')):
            target = out / '08_Logs' / base.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    readme = '''# Transformer and nominal OT: v2 results

## Folder guide
- 00_Summary: efficiency differences, AUC summary and readable energy tables.
- 01_Transformer_selected_spectra: training/test reports for the selected no-veto classifier.
- 02_MC_TB_before_DQ: historical v2 baseline with the earlier selections (theta < 80 degrees).
- 03_MC_TB_after_DQ_before_OT: current baseline, with DQ TB, theta < 20 degrees and BTF selections.
- 04_Nominal_OT_training: configuration, fitting selections, logs and verification for each class.
- 05_Nominal_OT_evaluation/efficiency: current before/after efficiencies, tables and confusion matrices.
- 05_Nominal_OT_evaluation/closure: ROC curves, AUCs and new discriminator splits.
- 06_Code: source snapshots for the classifier and DQ pipeline.
- 07_Latent_verification: extraction verification reports.
- 08_Logs: available execution logs.

## Interpretation
The classifier test set is drawn from selected spectra. The fixed-energy MC/TB
comparisons use separate samples and must not be labelled as the classifier test.
The before-DQ and after-DQ selections define different event populations.
The current OT comparison uses identical events before and after calibration.
PID efficiency is class recall. Differences are in percentage points (pp).
Carbon beam energies are in MeV/n. Other beam energies are in MeV.
Efficiency bars are Wilson z=1 counting intervals, without OT-map uncertainty.
Confusion matrices are normalised by true-class row; TB labels are beam labels.
Domain-discriminator AUC near 0.5 indicates reduced distinguishability for that
discriminator. It is not proof that the distributions are identical.
MC domain-test events were available to OT fitting, but not discriminator fitting.
TB validation was excluded from OT/discriminator fitting but included in the
fixed BTF selection medians. Derivation-only median comparisons are included.
Training objectives/displacements alone do not establish calibration quality.

## Scope and storage
This package covers the selected v2 classifier, its before/after-DQ baselines
and the completed nominal DQ OT analysis. It does not contain old-data OT,
LOEO or bootstrap results. New v2 DQ LOEO/bootstrap have not been run yet.
Raw parquet datasets, latent/event-prediction archives, classifier/OT checkpoints,
and fitted discriminator binaries remain on the analysis host. This is a report
and source bundle, not a complete backup of all data and trained models.
Source results were not modified. Missing optional folders are listed in
collection_report.json. Manifest hashes identify every included artifact.
'''
    (out / 'README.md').write_text(readme, encoding='utf-8')
    write_report = {'created': datetime.now().isoformat(), 'repo': str(repo),
        'classifier_checkpoint': str(transformer / 'checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104/best.pt'),
        'baseline_before_DQ': str(baseline), 'baseline_DQ': str(dq_baseline),
        'maps': str(maps), 'evaluation': str(evaluation), 'missing_optional_folders': missing}
    (out / 'collection_report.json').write_text(json.dumps(write_report, indent=2), encoding='utf-8')
    records = []
    for path in sorted(out.rglob('*')):
        if path.is_file():
            records.append({'file': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    pd.DataFrame(records).to_csv(out / 'manifest.csv', index=False)
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                package.write(path, path.relative_to(exports))
    print('Files collected:', len(records) + 1, flush=True)
    print('Missing optional folders:', missing, flush=True)
    print('Folder:', out, flush=True)
    print('ZIP:', archive, flush=True)
    print(f'ZIP size: {archive.stat().st_size / 1024**2:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
