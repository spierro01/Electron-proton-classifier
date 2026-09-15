"""Training entry point for the neural OT toy exercise."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from model import NeuralOT, OTConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/toy_ot_data.npz"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/neural_ot.pt"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--f-updates-per-g",
        type=int,
        default=1,
        help="Numero di update di f per ogni blocco.",
    )

    parser.add_argument(
        "--g-updates-per-f",
        type=int,
        default=1,
        help="Numero di update di g dopo ogni blocco di f.",
    )

    args = parser.parse_args()

    if args.f_updates_per_g < 1:
        parser.error("--f-updates-per-g deve essere almeno 1")

    if args.g_updates_per_f < 1:
        parser.error("--g-updates-per-f deve essere almeno 1")

    set_seed(args.seed)

    data = np.load(args.data)

    source_train = data["source_train"]
    target_train = data["target_train"]

    config = OTConfig(
        epochs=args.epochs,
        seed=args.seed,
        f_updates_per_g=args.f_updates_per_g,
        g_updates_per_f=args.g_updates_per_f,
    )

    transport = NeuralOT(config)

    print(f"Device: {transport.device}")

    print(
        f"Update ratio: "
        f"{config.f_updates_per_g} f update(s), then "
        f"{config.g_updates_per_f} g update(s)"
    )

    transport.fit(source_train, target_train)
    transport.save(args.checkpoint)

    print(f"Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()