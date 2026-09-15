"""Task 3: train one class-conditional OT map from saved latents."""

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import time

import numpy as np
import torch

from ot_model import NeuralOT, OTConfig
from tb_split import get_tb_split


HERE = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--class-name", choices=["e", "p", "C"], required=True)
    parser.add_argument("--mc", type=Path, default=HERE / "derived/latents_MC.npz")
    parser.add_argument("--tb", type=Path, default=HERE / "derived/latents_TB.npz")
    parser.add_argument("--mask", type=Path,
                        default=HERE / "derived/tb_is_derivation.npy")
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--identity-steps", type=int, default=1000)
    parser.add_argument("--g-updates", type=int, default=10)
    parser.add_argument("--f-updates", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)

    args = parser.parse_args()

    for path in [args.mc, args.tb, args.mask]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.out.exists():
        raise FileExistsError(
            f"{args.out} already exists. Use a new output directory."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if min(
        args.hidden, args.layers, args.batch_size, args.steps,
        args.g_updates, args.f_updates, args.log_every
    ) <= 0 or args.identity_steps < 0:
        parser.error("Invalid training parameters.")

    set_seed(args.seed)

    # Select MC events of the requested class.
    with np.load(args.mc, allow_pickle=False) as data:
        classes = data["classes"].tolist()
        checkpoint_hash = str(data["checkpoint_sha256"].item())

        if args.class_name not in classes:
            raise ValueError("Requested class is absent from checkpoint.")

        class_index = classes.index(args.class_name)
        labels = data["label"]
        energy = data["energy_mev"]
        mc_indices = np.flatnonzero(labels == class_index)
        mc_energy = energy[mc_indices]

        z_all = data["z_enc"]
        source = z_all[mc_indices].copy()
        del z_all

    # Select ONLY TB derivation events of the same class.
    mask_hash = sha256_file(args.mask)

    with np.load(args.tb, allow_pickle=False) as data:
        if data["classes"].tolist() != classes:
            raise ValueError("Different class order in MC and TB.")

        if str(data["checkpoint_sha256"].item()) != checkpoint_hash:
            raise ValueError("MC and TB use different PID checkpoints.")

        labels = data["label"]
        energy = data["energy_mev"]

        split = get_tb_split(len(labels), mask_file=args.mask)

        if split.dtype != np.bool_ or split.shape != labels.shape:
            raise ValueError("Invalid TB split.")

        if not np.array_equal(split, data["is_derivation"]):
            raise ValueError("TB split differs from the extraction split.")

        tb_indices = np.flatnonzero(
            (labels == class_index) & split
        )
        tb_energy = energy[tb_indices]

        z_all = data["z_enc"]
        target = z_all[tb_indices].copy()
        del z_all

    # Use only energies represented on both sides.
    common = np.intersect1d(
        np.unique(mc_energy),
        np.unique(tb_energy),
    )

    if len(common) == 0:
        raise ValueError("No common energies.")

    excluded_mc = np.setdiff1d(np.unique(mc_energy), common)
    excluded_tb = np.setdiff1d(np.unique(tb_energy), common)

    keep_mc = np.isin(mc_energy, common)
    keep_tb = np.isin(tb_energy, common)

    source = source[keep_mc]
    target = target[keep_tb]
    mc_indices = mc_indices[keep_mc]
    tb_indices = tb_indices[keep_tb]
    mc_energy = mc_energy[keep_mc]
    tb_energy = tb_energy[keep_tb]

    if source.shape[1:] != (64,) or target.shape[1:] != (64,):
        raise ValueError("Expected 64-dimensional latents.")

    config = OTConfig(
        in_dim=64,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        batch_size=args.batch_size,
        outer_steps=args.steps,
        identity_steps=args.identity_steps,
        g_updates=args.g_updates,
        f_updates=args.f_updates,
        lr=args.lr,
        min_lr=args.min_lr,
        log_every=args.log_every,
        seed=args.seed,
    )

    metadata = {
        "class_name": args.class_name,
        "classes": classes,
        "class_index": class_index,
        "checkpoint_sha256": checkpoint_hash,
        "tb_split_sha256": mask_hash,
        "mc_file": str(args.mc.resolve()),
        "tb_file": str(args.tb.resolve()),
        "source_events": len(source),
        "target_derivation_events": len(target),
        "common_energies": common.tolist(),
        "excluded_mc_energies": excluded_mc.tolist(),
        "excluded_tb_energies": excluded_tb.tolist(),
        "energy_sampling": (
            "Original energy mixtures pooled within each class; "
            "no energy conditioning in Task 3."
        ),
        "normalisation": (
            "Separate source and target statistics, "
            "TB derivation only."
        ),
        "config": asdict(config),
    }

    args.out.mkdir(parents=True)

    (args.out / "config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    np.savez(
        args.out / "selection.npz",
        mc_rows=mc_indices,
        tb_derivation_rows=tb_indices,
        mc_energy=mc_energy,
        tb_energy=tb_energy,
        common_energies=common,
    )

    print(f"Class: {args.class_name}", flush=True)
    print(f"Source MC: {len(source):,}", flush=True)
    print(f"Target TB derivation: {len(target):,}", flush=True)
    print(f"Common energies: {common.tolist()}", flush=True)
    print(f"Excluded TB energies: {excluded_tb.tolist()}", flush=True)

    transport = NeuralOT(config, device=args.device)
    print(f"Device: {transport.device}", flush=True)

    log_fields = [
        "outer_step", "loss_g", "loss_f",
        "mean_l2_standardised", "rms_coordinate_standardised",
        "mean_l2_original_latent", "lr_used",
    ]

    start = time.perf_counter()

    with (args.out / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=log_fields)
        writer.writeheader()

        def log_callback(record):
            writer.writerow(record)
            stream.flush()

        transport.fit(source, target, log_callback=log_callback)

    if sha256_file(args.mask) != mask_hash:
        raise RuntimeError("TB split changed during training.")

    elapsed = time.perf_counter() - start

    # Check reload invariance on a fixed source-only sample.
    rng = np.random.default_rng(args.seed + 2)
    check_indices = rng.choice(
        len(source),
        size=min(512, len(source)),
        replace=False,
    )
    check_source = source[check_indices]
    before = transport.transform(check_source, batch_size=128)

    checkpoint_path = args.out / "map.pt"
    transport.save(checkpoint_path, metadata=metadata)

    loaded = NeuralOT.load(checkpoint_path, device=args.device)
    after = loaded.transform(check_source, batch_size=128)

    np.testing.assert_allclose(
        before, after, rtol=1e-5, atol=1e-6
    )

    verification = {
        "reload_invariance_passed": True,
        "n_checked": len(check_source),
        "max_absolute_difference": float(np.max(np.abs(before - after))),
        "rtol": 1e-5,
        "atol": 1e-6,
        "elapsed_seconds": elapsed,
        "tb_validation_used_for_training": False,
    }

    (args.out / "verification.json").write_text(
        json.dumps(verification, indent=2),
        encoding="utf-8",
    )

    print("\nPASS: reload invariance.", flush=True)
    print(f"Training time: {elapsed:.1f} seconds", flush=True)
    print(f"Saved map: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()