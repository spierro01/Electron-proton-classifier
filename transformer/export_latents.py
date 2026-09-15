"""Extract CLS latents z_enc from a trained ParticleTransformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import pid_data as PD
from pid_model import ParticleTransformer


PARQUET_CHUNK = 50_000


def load_model(ckpt_path: Path, device: str):
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True,
    )

    args = checkpoint["args"]
    classes = checkpoint["classes"]

    model = ParticleTransformer(
        num_classes=len(classes),
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        ffn_dim=args["ffn_dim"],
        dropout=args["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)

    return model, classes, mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Export z_enc latent vectors from a parquet file."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input MC or test-beam parquet.",
    )

    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/pid_transformer/best.pt"),
        help="Trained Transformer checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which to save z_enc.npy and related files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, mean, std = load_model(args.ckpt, device)

    parquet = pq.ParquetFile(args.input)

    available_columns = set(parquet.schema.names)
    required_columns = PD.required_columns()

    missing = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing:
        raise SystemExit(
            f"Missing {len(missing)} required columns, for example: "
            f"{missing[:4]}"
        )

    has_truth = "particle_type" in available_columns
    columns = required_columns + (
        ["particle_type"] if has_truth else []
    )

    n_events = parquet.metadata.num_rows
    d_model = model.cls_token.shape[-1]

    args.output.mkdir(parents=True, exist_ok=True)

    z_file = args.output / "z_enc.npy"
    probabilities_file = args.output / "probabilities.npy"
    predictions_file = args.output / "predictions.npy"

    z_enc = np.lib.format.open_memmap(
        z_file,
        mode="w+",
        dtype=np.float32,
        shape=(n_events, d_model),
    )

    probabilities = np.lib.format.open_memmap(
        probabilities_file,
        mode="w+",
        dtype=np.float32,
        shape=(n_events, len(classes)),
    )

    predictions = np.lib.format.open_memmap(
        predictions_file,
        mode="w+",
        dtype=np.int64,
        shape=(n_events,),
    )

    truth = None

    if has_truth:
        truth = np.lib.format.open_memmap(
            args.output / "truth.npy",
            mode="w+",
            dtype=np.int64,
            shape=(n_events,),
        )

    class_to_index = {
        particle_class: index
        for index, particle_class in enumerate(classes)
    }

    print(f"Input: {args.input}")
    print(f"Events: {n_events:,}")
    print(f"Classes: {classes}")
    print(f"Latent dimension: {d_model}")
    print(f"Device: {device}")

    offset = 0

    for record_batch in parquet.iter_batches(
        batch_size=PARQUET_CHUNK,
        columns=columns,
    ):
        dataframe = record_batch.to_pandas()
        X = PD.build_matrix(dataframe)

        for start in range(0, len(X), args.batch_size):
            end = min(start + args.batch_size, len(X))

            x_batch = (X[start:end] - mean) / std

            tensor = torch.from_numpy(
                x_batch.astype(np.float32)
            ).to(device)

            with torch.no_grad():
                latent_dict = model.latents(tensor)

                batch_z_enc = (
                    latent_dict["z_enc"]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                batch_probabilities = (
                    torch.softmax(
                        latent_dict["logits"],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            destination = slice(offset + start, offset + end)

            z_enc[destination] = batch_z_enc
            probabilities[destination] = batch_probabilities
            predictions[destination] = batch_probabilities.argmax(axis=1)

        if has_truth:
            canonical_truth = dataframe["particle_type"].map(
                PD.LABEL_MAP
            )

            truth_values = np.asarray(
                [
                    class_to_index.get(value, -1)
                    for value in canonical_truth
                ],
                dtype=np.int64,
            )

            truth[offset:offset + len(dataframe)] = truth_values

        offset += len(dataframe)

        print(f"Processed {offset:,}/{n_events:,} events")

    z_enc.flush()
    probabilities.flush()
    predictions.flush()

    if truth is not None:
        truth.flush()

    metadata = {
        "checkpoint": str(args.ckpt),
        "input": str(args.input),
        "classes": classes,
        "n_events": int(n_events),
        "latent_dimension": int(d_model),
        "has_truth": bool(has_truth),
    }

    with open(args.output / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved:")
    print(f"  {z_file}")
    print(f"  {probabilities_file}")
    print(f"  {predictions_file}")

    if has_truth:
        print(f"  {args.output / 'truth.npy'}")

    print(f"  {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()