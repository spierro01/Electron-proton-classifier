"""Create balanced class-conditional OT datasets from Transformer latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ["e", "p", "C"]


def split_indices(
    indices: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/test split of already selected unpaired events."""

    shuffled = rng.permutation(indices)
    n_test = int(len(shuffled) * test_fraction)

    test_indices = shuffled[:n_test]
    train_indices = shuffled[n_test:]

    return train_indices, test_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare class-conditional real-to-MC OT datasets."
    )

    parser.add_argument(
        "--mc-dir",
        type=Path,
        default=Path("derived/latents/mc"),
        help="Directory containing MC latent files.",
    )

    parser.add_argument(
        "--tb-dir",
        type=Path,
        default=Path("derived/latents/tb"),
        help="Directory containing test-beam latent files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("derived/ot_data"),
        help="Output directory for OT arrays.",
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50_000,
        help="Maximum balanced number of events for each particle species.",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction reserved for OT closure evaluation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.max_per_class < 1:
        parser.error("--max-per-class must be at least 1")

    if not 0.0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must be between 0 and 0.5")

    rng = np.random.default_rng(args.seed)

    print("Loading MC latents...")
    z_mc = np.load(
        args.mc_dir / "z_enc.npy",
        mmap_mode="r",
    )

    y_mc = np.load(
        args.mc_dir / "truth.npy",
        mmap_mode="r",
    )

    print("Loading test-beam latents...")
    z_tb = np.load(
        args.tb_dir / "z_enc.npy",
        mmap_mode="r",
    )

    y_tb = np.load(
        args.tb_dir / "predictions.npy",
        mmap_mode="r",
    )

    probabilities_tb = np.load(
        args.tb_dir / "probabilities.npy",
        mmap_mode="r",
    )

    confidence_tb = probabilities_tb.max(axis=1)

    print(f"MC latent shape: {z_mc.shape}")
    print(f"TB latent shape: {z_tb.shape}")

    args.output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "direction": "test_beam_to_mc",
        "classes": CLASS_NAMES,
        "max_per_class": args.max_per_class,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "class_counts": {},
    }

    for class_index, class_name in enumerate(CLASS_NAMES):
        # MC has truth labels. Helium was saved as -1 and is excluded.
        mc_indices = np.flatnonzero(y_mc == class_index)

        # Test beam has pseudo-labels from the frozen Transformer.
        tb_candidates = np.flatnonzero(y_tb == class_index)

        if len(mc_indices) == 0:
            raise RuntimeError(
                f"No MC events for class {class_name}."
            )

        if len(tb_candidates) == 0:
            raise RuntimeError(
                f"No test-beam events predicted as {class_name}."
            )

        # Select the most confident pseudo-labelled test-beam events.
        confidence_order = np.argsort(
            confidence_tb[tb_candidates]
        )[::-1]

        tb_ranked = tb_candidates[confidence_order]

        n_samples = min(
            len(mc_indices),
            len(tb_ranked),
            args.max_per_class,
        )

        # MC events are randomly selected: they already have true labels.
        mc_selected = rng.choice(
            mc_indices,
            size=n_samples,
            replace=False,
        )

        # Test-beam events are selected by descending confidence.
        tb_selected = tb_ranked[:n_samples]

        real_train_indices, real_test_indices = split_indices(
            tb_selected,
            rng,
            args.test_fraction,
        )

        mc_train_indices, mc_test_indices = split_indices(
            mc_selected,
            rng,
            args.test_fraction,
        )

        class_directory = args.output / class_name
        class_directory.mkdir(parents=True, exist_ok=True)

        # OT direction: source = real test beam; target = MC.
        np.save(
            class_directory / "real_train.npy",
            z_tb[real_train_indices],
        )

        np.save(
            class_directory / "real_test.npy",
            z_tb[real_test_indices],
        )

        np.save(
            class_directory / "mc_train.npy",
            z_mc[mc_train_indices],
        )

        np.save(
            class_directory / "mc_test.npy",
            z_mc[mc_test_indices],
        )

        selected_confidence = confidence_tb[tb_selected]

        metadata["class_counts"][class_name] = {
            "mc_available": int(len(mc_indices)),
            "tb_predicted_available": int(len(tb_candidates)),
            "balanced_total": int(n_samples),
            "train_per_domain": int(len(real_train_indices)),
            "test_per_domain": int(len(real_test_indices)),
            "tb_selected_min_confidence": float(
                selected_confidence.min()
            ),
            "tb_selected_mean_confidence": float(
                selected_confidence.mean()
            ),
        }

        print(
            f"{class_name}: "
            f"MC available={len(mc_indices):,} | "
            f"TB predicted={len(tb_candidates):,} | "
            f"used={n_samples:,} | "
            f"TB mean confidence={selected_confidence.mean():.4f}"
        )

    metadata_path = args.output / "metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    print("\nCreated OT datasets:")
    print(args.output)
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()