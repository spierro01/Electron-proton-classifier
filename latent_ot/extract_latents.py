"""Task 1: extract classifier latents from fixed-energy MC and test beam.

Requirements:
- Stream parquet data with iter_batches().
- Keep the classifier frozen.
- Use the checkpoint's mean and std.
- Preserve original event order.
- Reuse the existing TB derivation/validation split.
- Save one NPZ per domain.
- Verify saved predictions against full predict.py output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


# This script lives in Electron-proton-classifier/latent_ot.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRANSFORMER_DIR = ROOT / "transformer"

sys.path.insert(0, str(TRANSFORMER_DIR))

import pid_data as PD
from pid_model import ParticleTransformer
from tb_split import get_tb_split


def sha256_file(path: Path) -> str:
    """Compute a file fingerprint without loading the whole file."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_model(checkpoint: Path, device: str):
    """Rebuild the classifier and load the original MC normalisation."""
    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    config = ckpt["args"]
    classes = list(ckpt["classes"])

    if len(classes) != len(set(classes)):
        raise ValueError("Duplicate class names in checkpoint.")

    model = ParticleTransformer(
        num_classes=len(classes),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ffn_dim=config["ffn_dim"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    # No training and no parameter updates.
    model.eval()
    model.requires_grad_(False)

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    expected_shape = (PD.N_STEPS, PD.N_FEATURES)

    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            f"Expected normalisation shape {expected_shape}, "
            f"got mean={mean.shape}, std={std.shape}."
        )

    if (
        not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("Invalid checkpoint normalisation statistics.")

    return model, classes, mean, std, config


def calculate_metrics(logits, labels, classes):
    """Accuracy and confusion matrix from stored logits."""
    predicted = logits.argmax(axis=1)

    matrix = np.zeros(
        (len(classes), len(classes)),
        dtype=np.int64,
    )
    np.add.at(matrix, (labels, predicted), 1)

    per_class = {}

    for i, name in enumerate(classes):
        n = int(matrix[i].sum())

        if n:
            per_class[name] = {
                "n": n,
                "correct": int(matrix[i, i]),
                "recall": float(matrix[i, i] / n),
            }

    return {
        "n": int(len(labels)),
        "accuracy": float((predicted == labels).mean()),
        "per_class": per_class,
        "confusion": matrix.tolist(),
    }


def verify_reference_csv(csv_path, predicted, labels, classes):
    """Check predictions against full predict.py output.

    The reference CSV must have been generated on the same unchanged
    parquet, without --limit, preserving its original row order.
    """
    offset = 0
    mismatches = 0
    class_names = np.asarray(classes)

    for chunk in pd.read_csv(
        csv_path,
        usecols=["pred", "truth"],
        chunksize=100_000,
    ):
        end = offset + len(chunk)

        if end > len(labels):
            raise ValueError("Reference CSV contains too many rows.")

        reference_labels = chunk["truth"].map(PD.LABEL_MAP)

        if reference_labels.isna().any():
            raise ValueError("Unknown labels in reference CSV.")

        expected_labels = class_names[labels[offset:end]]

        if not np.array_equal(
            reference_labels.to_numpy(),
            expected_labels,
        ):
            raise ValueError(
                "Reference labels do not match parquet order."
            )

        if not chunk["pred"].isin(classes).all():
            raise ValueError("Unknown predictions in reference CSV.")

        expected_predictions = class_names[predicted[offset:end]]

        mismatches += int(
            (
                chunk["pred"].to_numpy()
                != expected_predictions
            ).sum()
        )

        offset = end

    if offset != len(labels):
        raise ValueError("Reference CSV contains too few rows.")

    if mismatches:
        raise ValueError(
            f"{mismatches} predictions differ from predict.py. "
            "Check checkpoint, input order and numerical behaviour "
            "before continuing."
        )

    return {
        "reference_csv": str(csv_path.resolve()),
        "rows_checked": offset,
        "prediction_mismatches": mismatches,
        "passed": True,
    }


@torch.no_grad()
def extract(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for path in [args.input, args.ckpt, args.reference]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Prevent accidental use of the classifier-training spectrum.
    if "spectra" in args.input.name.lower():
        raise ValueError(
            "Use dumpMC.parquet at fixed energies, "
            "not dumpMC_spectra.parquet."
        )

    output = args.out
    partial = output.with_name(output.stem + ".partial.npz")
    report_path = output.with_suffix(".verification.json")

    for path in [output, partial, report_path]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}. "
                "Inspect it or choose another --out filename."
            )

    checkpoint_hash = sha256_file(args.ckpt)
    input_stat = args.input.stat()

    model, classes, mean, std, config = load_model(
        args.ckpt, device
    )

    parquet = pq.ParquetFile(args.input)
    n = parquet.metadata.num_rows

    if n == 0:
        raise ValueError("The input parquet is empty.")

    # Labels and energy are not included in required_columns().
    columns = list(dict.fromkeys(
        PD.required_columns() + ["particle_type", "energy_mev"]
    ))

    available = set(parquet.schema_arrow.names)
    missing = [name for name in columns if name not in available]

    if missing:
        raise ValueError(
            f"Missing {len(missing)} columns: {missing[:10]}"
        )

    split = None
    split_hash = None

    if args.domain == "TB":
        # Require the existing split: do not silently create another.
        if not args.mask.is_file():
            raise FileNotFoundError(
                f"Existing TB split required: {args.mask}"
            )

        split_hash = sha256_file(args.mask)
        split = get_tb_split(n, mask_file=args.mask)

        if split.dtype != np.bool_ or split.shape != (n,):
            raise ValueError(
                "TB split must be a boolean array matching parquet rows."
            )

        if not split.any() or split.all():
            raise ValueError("TB split contains an empty subset.")

    latent_dim = config["d_model"]
    label_lookup = {
        name: index for index, name in enumerate(classes)
    }

    # Only compact outputs are allocated for the entire dataset.
    # Raw detector inputs remain limited to a single read batch.
    z_enc = np.empty((n, latent_dim), dtype=np.float32)
    logits = np.empty((n, len(classes)), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    energies = np.empty(n, dtype=np.float64)

    print(f"Domain: {args.domain}", flush=True)
    print(f"Input: {args.input}", flush=True)
    print(f"Checkpoint: {args.ckpt}", flush=True)
    print(f"Classes: {classes}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Events: {n:,}", flush=True)
    print(f"Latent dimension: {latent_dim}", flush=True)
    print(
        "Classifier frozen; checkpoint normalisation reused.",
        flush=True,
    )

    offset = 0
    checked_forward = False

    for batch in parquet.iter_batches(
        batch_size=args.read_batch,
        columns=columns,
    ):
        df = batch.to_pandas()
        end = offset + len(df)

        canonical = df["particle_type"].map(PD.LABEL_MAP)

        if canonical.isna().any():
            raise ValueError("Unknown particle_type values.")

        if not canonical.isin(classes).all():
            raise ValueError(
                "Input contains species absent from the checkpoint."
            )

        labels[offset:end] = canonical.map(
            label_lookup
        ).to_numpy(dtype=np.int64)

        energies[offset:end] = pd.to_numeric(
            df["energy_mev"],
            errors="raise",
        ).to_numpy(dtype=np.float64)

        if not np.isfinite(energies[offset:end]).all():
            raise ValueError("Non-finite beam energies.")

        X = PD.build_matrix(df)

        # Exactly the mean/std stored during MC training.
        X = ((X - mean) / std).astype(np.float32, copy=False)

        if not np.isfinite(X).all():
            raise ValueError("Non-finite normalised detector inputs.")

        for start in range(0, len(X), args.batch_size):
            stop = min(start + args.batch_size, len(X))

            x = torch.from_numpy(X[start:stop]).to(device)
            representations = model.latents(x)

            z = representations["z_enc"]
            scores = representations["logits"]

            if not checked_forward:
                # Verify that extracting logits is consistent with
                # the standard inference path in evaluation mode.
                torch.testing.assert_close(
                    scores,
                    model(x),
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    scores,
                    model.head_from_latent(z),
                    rtol=1e-5,
                    atol=1e-6,
                )
                checked_forward = True

            z_array = z.cpu().numpy()
            score_array = scores.cpu().numpy()

            if z_array.shape != (stop - start, latent_dim):
                raise ValueError("Unexpected latent shape.")

            if score_array.shape != (stop - start, len(classes)):
                raise ValueError("Unexpected logits shape.")

            if (
                not np.isfinite(z_array).all()
                or not np.isfinite(score_array).all()
            ):
                raise ValueError("Non-finite model outputs.")

            z_enc[offset + start:offset + stop] = z_array
            logits[offset + start:offset + stop] = score_array

        offset = end

        print(
            f"Processed {offset:,} / {n:,}",
            flush=True,
        )

    if offset != n:
        raise RuntimeError("Incomplete extraction.")

    if sha256_file(args.ckpt) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during extraction.")

    if split is not None:
        if sha256_file(args.mask) != split_hash:
            raise RuntimeError("TB split changed during extraction.")

    final_stat = args.input.stat()
    if (
        final_stat.st_size != input_stat.st_size
        or final_stat.st_mtime_ns != input_stat.st_mtime_ns
    ):
        raise RuntimeError("Input parquet changed during extraction.")

    metadata = {
        "domain": args.domain,
        "input_path": str(args.input.resolve()),
        "input_size_bytes": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "checkpoint_path": str(args.ckpt.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "split_path": (
            str(args.mask.resolve()) if split is not None else None
        ),
        "split_sha256": split_hash,
        "classes": classes,
        "latent_dimension": latent_dim,
        "device": device,
        "torch_version": str(torch.__version__),
        "batch_size": args.batch_size,
        "read_batch": args.read_batch,
        "row_order": "Original parquet order, no filtering or shuffling.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "z_enc": z_enc,
        "logits": logits,
        "label": labels,
        "energy_mev": energies,
        "classes": np.asarray(classes, dtype=str),
        "parquet_row": np.arange(n, dtype=np.int64),
        "checkpoint_sha256": np.asarray(checkpoint_hash),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }

    if split is not None:
        arrays["is_derivation"] = split

    print("Saving NPZ...", flush=True)
    np.savez(partial, **arrays)
    del arrays

    print("Verifying saved arrays...", flush=True)

    with np.load(partial, allow_pickle=False) as saved:
        stored_z = saved["z_enc"]

        if not np.array_equal(stored_z, z_enc):
            raise ValueError("Latents changed during saving.")

        del stored_z

        stored_logits = saved["logits"]
        stored_labels = saved["label"]

        if not np.array_equal(stored_logits, logits):
            raise ValueError("Logits changed during saving.")

        if not np.array_equal(stored_labels, labels):
            raise ValueError("Labels changed during saving.")

        if not np.array_equal(saved["energy_mev"], energies):
            raise ValueError("Energies changed during saving.")

        if saved["classes"].tolist() != classes:
            raise ValueError("Class order changed during saving.")

        if not np.array_equal(saved["parquet_row"], np.arange(n)):
            raise ValueError("Event order changed during saving.")

        if split is not None:
            if not np.array_equal(saved["is_derivation"], split):
                raise ValueError("Saved TB split differs from original.")

        reference_check = verify_reference_csv(
            args.reference,
            stored_logits.argmax(axis=1),
            stored_labels,
            classes,
        )

        full_metrics = calculate_metrics(
            stored_logits,
            stored_labels,
            classes,
        )

        report = {
            "metadata": metadata,
            "z_enc_shape": [n, latent_dim],
            "logits_shape": [n, len(classes)],
            "predict_py_check": reference_check,
            "whole_file_consistency_check": full_metrics,
        }

        if split is not None:
            report["n_derivation"] = int(split.sum())
            report["n_validation"] = int((~split).sum())
            report["TB_validation_all_energies"] = calculate_metrics(
                stored_logits[~split],
                stored_labels[~split],
                classes,
            )
            report["note"] = (
                "Whole-file metrics only verify extraction against "
                "predict.py. Scientific evaluation must use TB validation "
                "and select energies shared with MC."
            )

    # Final filename is used only after checks pass.
    partial.rename(output)

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        "\nPASS: all saved predictions match predict.py.",
        flush=True,
    )
    print(
        "\nWhole-file consistency check:\n"
        + json.dumps(full_metrics, indent=2),
        flush=True,
    )

    if split is not None:
        print(f"\nTB derivation: {int(split.sum()):,}")
        print(f"TB validation: {int((~split).sum()):,}")

    print(f"\nz_enc shape: {z_enc.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"Saved: {output.resolve()}")
    print(f"Verification report: {report_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Export frozen classifier latents for Task 1."
    )

    parser.add_argument(
        "--domain",
        choices=["MC", "TB"],
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument(
        "--ckpt",
        type=Path,
        default=TRANSFORMER_DIR / "checkpoints/pid_transformer/best.pt",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=HERE / "derived/tb_is_derivation.npy",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--read_batch", type=int, default=50_000)

    args = parser.parse_args()

    if args.batch_size <= 0 or args.read_batch <= 0:
        parser.error("Batch sizes must be positive.")

    if args.out.suffix.lower() != ".npz":
        parser.error("--out must have the .npz extension.")

    extract(args)


if __name__ == "__main__":
    main()