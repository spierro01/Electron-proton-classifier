"""Export selected fixed-energy latents with a frozen no-veto classifier.

Reuse the completed DQ evaluation, its medians and its input fingerprints.
Export both TB subsets; compare only TB validation with reference predictions.
parquet_row indexes the current input file, before selection. The exported
is_derivation array indexes the selected TB rows, not the full parquet file.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import pid_data as PD
from predict import load_model
import evaluate_selected_v2_dq as selection


class ReferenceChecker:
    """Consume a reference CSV in exactly the same selected row order."""
    def __init__(self, path, classes):
        self.classes = classes
        self.reader = pd.read_csv(path, iterator=True)
        self.n = 0
        self.max_probability_difference = 0.0

    def check(self, rows, labels, energies, probabilities):
        if len(rows) == 0:
            return
        frame = self.reader.get_chunk(len(rows))
        if len(frame) != len(rows):
            raise ValueError("Reference CSV ended before the selected sample.")
        np.testing.assert_array_equal(frame['parquet_row'].to_numpy(), rows)
        np.testing.assert_array_equal(frame['truth'].to_numpy(), labels)
        np.testing.assert_allclose(frame['energy_mev'].to_numpy(), energies, rtol=0, atol=1e-8)
        predictions = np.asarray(self.classes)[probabilities.argmax(axis=1)]
        np.testing.assert_array_equal(frame['pred'].to_numpy(), predictions)
        expected = frame[[f'p_{c}' for c in self.classes]].to_numpy()
        np.testing.assert_allclose(probabilities, expected, rtol=2e-5, atol=2e-6)
        difference = float(np.max(np.abs(probabilities - expected)))
        self.max_probability_difference = max(self.max_probability_difference, difference)
        self.n += len(rows)

    def finish(self):
        try:
            remaining = self.reader.get_chunk(1)
        except StopIteration:
            remaining = pd.DataFrame()
        if len(remaining):
            raise ValueError("Reference CSV contains additional selected rows.")
        self.reader.close()


def selected_batches(path, domain, classes, medians, columns):
    offset = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        frame = batch.to_pandas()
        labels = frame['particle_type'].map(PD.LABEL_MAP)
        if labels.isna().any():
            raise ValueError('Unknown particle labels.')
        labels = labels.to_numpy()
        rows = np.arange(offset, offset + len(frame), dtype=np.int64)
        keep = np.ones(len(frame), dtype=bool)
        stages = selection.base_selection_stages(frame, labels, domain, classes)
        stages += selection.btf_selection_stages(frame, labels, medians)
        for _, cut in stages:
            keep &= cut
        offset += len(frame)
        if keep.any():
            yield frame.loc[keep], labels[keep], rows[keep]
    if offset != parquet.metadata.num_rows:
        raise ValueError('Incomplete parquet scan.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['MC', 'TB'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()
    if args.out.suffix != '.npz' or args.batch_size < 1:
        raise ValueError('Use a .npz output path and a positive batch size.')
    report_path = args.out.with_suffix('.verification.json')
    partial = args.out.with_name(args.out.stem + '.partial.npz')
    work = args.out.with_name(args.out.stem + '_work')
    for path in [args.out, report_path, partial, work]:
        if path.exists():
            raise FileExistsError(f'{path}: use a new output name.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')

    provenance = json.loads((args.evaluation / 'provenance.json').read_text())
    completion = json.loads((args.evaluation / 'completion.json').read_text())
    medians = json.loads((args.evaluation / 'btf_medians.json').read_text())
    if not completion.get('completed') or not completion.get('TB_validation_only'):
        raise ValueError('A completed validation-only reference evaluation is required.')
    if provenance['selection']['theta'] != 'theta < 20 degrees':
        raise ValueError('Reference evaluation uses a different angle selection.')
    if medians != provenance['BTF_selection']['medians']:
        raise ValueError('Median file differs from evaluation provenance.')
    print('Checking checkpoint, dataset and split fingerprints...', flush=True)
    checkpoint_hash = selection.sha256(args.ckpt)
    input_hash = selection.sha256(args.input)
    mask_hash = selection.sha256(args.mask)
    if checkpoint_hash != provenance['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the reference evaluation.')
    if input_hash != provenance['source_files'][args.domain]['sha256']:
        raise ValueError('Input dataset differs from the reference evaluation.')
    if mask_hash != provenance['mask_sha256']:
        raise ValueError('TB split differs from the reference evaluation.')
    mask = np.load(args.mask, allow_pickle=False)
    expected_tb_rows = provenance['source_files']['TB']['rows']
    if mask.dtype != np.bool_ or mask.shape != (expected_tb_rows,):
        raise ValueError('Invalid full-file TB split.')

    expected_layers = ['TR1', 'TR2'] + [f'RAN{i}' for i in range(12)] + ['EN1', 'EN2']
    if [name for name, _ in PD.LAYERS] != expected_layers:
        raise ValueError('Unexpected detector token layout.')
    if any('VETO' in c or c.startswith(('BOT_', 'LAT_')) for c in PD.required_columns()):
        raise ValueError('Veto columns must not enter the classifier.')
    model, classes, mean, std, _ = load_model(args.ckpt, args.device)
    if classes != ['e', 'p', 'C'] or mean.shape != (16, 8) or std.shape != (16, 8):
        raise ValueError('Unexpected classes or normalisation shape.')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid checkpoint normalisation.')
    model.eval()
    model.requires_grad_(False)
    boot, run = ('file_boot_nr', 'file_run_nr') if args.domain == 'MC' else ('boot_nr', 'run_nr')
    columns = list(dict.fromkeys(PD.required_columns() + selection.TR1 + selection.TR2
        + selection.VETO + selection.EDEP + ['particle_type', 'energy_mev', 'theta', boot, run]))
    missing = set(columns) - set(pq.ParquetFile(args.input).schema_arrow.names)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    # First pass reads only selection columns to size disk-backed arrays.
    cut_columns = list(dict.fromkeys(selection.TR1 + selection.TR2 + selection.VETO
        + selection.EDEP + ['particle_type', 'energy_mev', 'theta', boot, run]))
    kwargs = dict(path=args.input, domain=args.domain, classes=classes,
                  medians=medians[args.domain]['full'])
    n = sum(len(rows) for _, _, rows in selected_batches(**kwargs, columns=cut_columns))
    if n == 0:
        raise ValueError('No selected events.')
    print(f'{args.domain}: {n:,} selected events. Classifier frozen.', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    arrays = {}
    specifications = {'z_enc': (np.float32, (n, 64)), 'logits': (np.float32, (n, 3)),
                      'label': (np.int64, (n,)), 'energy_mev': (np.float64, (n,)),
                      'parquet_row': (np.int64, (n,))}
    if args.domain == 'TB':
        specifications['is_derivation'] = (np.bool_, (n,))
    for key, (dtype, shape) in specifications.items():
        arrays[key] = np.lib.format.open_memmap(work / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
    reference = ReferenceChecker(args.evaluation / f'predictions_{args.domain}_selected.csv', classes)
    position = 0
    for frame, labels, rows in selected_batches(**kwargs, columns=columns):
        x = PD.build_matrix(frame)
        x = ((x - mean) / std).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite normalised classifier input.')
        size = len(rows)
        probabilities = np.empty((size, 3), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, size, args.batch_size):
                stop = min(start + args.batch_size, size)
                tensor = torch.from_numpy(x[start:stop]).to(args.device)
                z = model.forward_features(tensor)
                logits = model.head_from_latent(z)
                if z.shape != (stop - start, 64):
                    raise ValueError('Unexpected encoder latent shape.')
                if not torch.isfinite(z).all() or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite latent or logits.')
                arrays['z_enc'][position + start:position + stop] = z.cpu().numpy()
                arrays['logits'][position + start:position + stop] = logits.cpu().numpy()
                probabilities[start:stop] = logits.softmax(dim=1).cpu().numpy()
        energies = frame['energy_mev'].to_numpy(dtype=float)
        sl = slice(position, position + size)
        arrays['label'][sl] = np.array([classes.index(c) for c in labels], dtype=np.int64)
        arrays['energy_mev'][sl] = energies
        arrays['parquet_row'][sl] = rows
        compare = np.ones(size, dtype=bool)
        if args.domain == 'TB':
            arrays['is_derivation'][sl] = mask[rows]
            compare = ~mask[rows]
        reference.check(rows[compare], labels[compare], energies[compare], probabilities[compare])
        position += size
        print(f'Processed {position:,}/{n:,}', flush=True)
    if position != n:
        raise ValueError('Selected event count changed between passes.')
    reference.finish()
    if selection.sha256(args.mask) != mask_hash:
        raise RuntimeError('TB split changed during extraction.')
    metadata = {
        'domain': args.domain, 'source': str(args.input.resolve()), 'source_sha256': input_hash,
        'checkpoint': str(args.ckpt.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'mask': str(args.mask.resolve()), 'mask_sha256': mask_hash,
        'evaluation': str(args.evaluation.resolve()), 'evaluation_provenance': provenance,
        'extractor_sha256': selection.sha256(__file__),
        'selection_script_sha256': selection.sha256(selection.__file__),
        'n_selected': n, 'latent_dimension': 64, 'classes': classes,
        'row_index_definition': 'original row in the current domain parquet before selection',
        'split_definition': 'is_derivation indexes selected TB rows; full-file split indexed by parquet_row',
    }
    for array in arrays.values():
        array.flush()
    print('Saving selected latent archive...', flush=True)
    np.savez_compressed(partial, **arrays, classes=np.asarray(classes),
                        checkpoint_sha256=np.asarray(checkpoint_hash),
                        metadata_json=np.asarray(json.dumps(metadata)))
    print('Verifying saved arrays...', flush=True)
    with np.load(partial, allow_pickle=False) as saved:
        for key, array in arrays.items():
            np.testing.assert_array_equal(saved[key], array)
        np.testing.assert_array_equal(saved['classes'], np.asarray(classes))
        if json.loads(saved['metadata_json'].item()) != metadata:
            raise ValueError('Saved metadata mismatch.')
    report = {'completed': True, 'domain': args.domain, 'n_selected': n,
              'saved_arrays_verified': True, 'reference_predictions_match': True,
              'reference_rows_checked': reference.n,
              'max_probability_difference': reference.max_probability_difference,
              'mask_unchanged': True, 'checkpoint_sha256': checkpoint_hash}
    if args.domain == 'TB':
        report['derivation_events'] = int(arrays['is_derivation'].sum())
        report['validation_events'] = n - report['derivation_events']
    partial.rename(args.out)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Keep disk-backed work arrays for recovery; never remove unrelated files.
    print(json.dumps(report, indent=2), flush=True)
    print('Saved:', args.out.resolve(), flush=True)
    print('Work arrays retained in:', work.resolve(), flush=True)


if __name__ == '__main__':
    main()
