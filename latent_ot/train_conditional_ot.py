"""Task 6: energy-conditioned neural OT.

One map per particle class.
Only TB derivation is used for fitting.
Source and target minibatches share the same energy multiset.
"""

from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from conditional_ot_model import ConditionalICNN, conditional_gradient
from train_ot import set_seed, sha256_file


HERE = Path(__file__).resolve().parent


class ConditionalTransport:
    """Conditional transport with source/target normalisation."""

    def __init__(self, architecture, device="cpu"):
        self.architecture = architecture
        self.device = torch.device(device)

        self.f = ConditionalICNN(**architecture).to(self.device)
        self.g = ConditionalICNN(**architecture).to(self.device)

        self.stats = None

    def normalise_energy(self, energy):
        return (
            (np.asarray(energy, dtype=np.float32) - self.stats["energy_mean"])
            / self.stats["energy_std"]
        ).reshape(-1, 1)

    def transform(self, source, energy, batch_size=256):
        if self.stats is None:
            raise RuntimeError("Missing normalisation statistics.")

        source = np.asarray(source, dtype=np.float32)
        energy = np.asarray(energy)

        if source.ndim != 2 or source.shape[1] != 64:
            raise ValueError("Expected source shape (N, 64).")
        if energy.shape != (len(source),):
            raise ValueError("Energy/event alignment error.")
        if not np.isfinite(source).all() or not np.isfinite(energy).all():
            raise ValueError("Non-finite inputs.")
        if batch_size < 1:
            raise ValueError("Invalid batch size.")

        result = np.empty_like(source)
        self.g.eval()

        for start in range(0, len(source), batch_size):
            stop = min(start + batch_size, len(source))

            z = (
                (source[start:stop] - self.stats["source_mean"])
                / self.stats["source_std"]
            ).astype(np.float32)
            e = self.normalise_energy(energy[start:stop])

            mapped = conditional_gradient(
                self.g,
                torch.from_numpy(z).to(self.device),
                torch.from_numpy(e).to(self.device),
                create_graph=False,
            ).detach().cpu().numpy()

            result[start:stop] = (
                mapped * self.stats["target_std"]
                + self.stats["target_mean"]
            )

        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite transported latents.")

        return result

    def save(self, path, metadata, history):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)

        torch.save({
            "architecture": self.architecture,
            "f_state_dict": self.f.state_dict(),
            "g_state_dict": self.g.state_dict(),
            "stats": {
                key: np.asarray(value).tolist()
                for key, value in self.stats.items()
            },
            "metadata": metadata,
            "history": history,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
        model = cls(checkpoint["architecture"], device=device)
        model.f.load_state_dict(checkpoint["f_state_dict"])
        model.g.load_state_dict(checkpoint["g_state_dict"])
        model.stats = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in checkpoint["stats"].items()
        }
        model.f.eval()
        model.g.eval()
        return model


def read_class(path, cls, mask=None):
    with np.load(path, allow_pickle=False) as data:
        classes = data["classes"].tolist()
        index = classes.index(cls)
        labels = data["label"]
        energy = data["energy_mev"]

        np.testing.assert_array_equal(
            data["parquet_row"], np.arange(len(labels))
        )

        selected = labels == index
        if mask is not None:
            if mask.dtype != np.bool_ or mask.shape != labels.shape:
                raise ValueError("Invalid TB mask.")
            np.testing.assert_array_equal(mask, data["is_derivation"])
            selected &= mask

        rows = np.flatnonzero(selected)
        z = data["z_enc"][rows]
        e = energy[rows]
        checkpoint = str(data["checkpoint_sha256"].item())

    if z.shape != (len(rows), 64):
        raise ValueError("Unexpected latent dimensions.")
    if not np.isfinite(z).all() or not np.isfinite(e).all():
        raise ValueError("Non-finite data.")

    return z, e, rows, classes, checkpoint


def statistics(x):
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def optimizer_step(loss, optimizer, parameters):
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite training loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=10.0, error_if_nonfinite=True
    )
    optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", choices=["e", "p", "C"], required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--identity-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(
            f"{args.out} already exists. Use a new directory."
        )
    if min(args.hidden, args.batch_size, args.steps, args.log_every) < 1:
        raise ValueError("Invalid training parameters.")
    if args.identity_steps < 0:
        raise ValueError("Invalid identity-step count.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    set_seed(args.seed)

    mc_path = HERE / "derived/latents_MC.npz"
    tb_path = HERE / "derived/latents_TB.npz"
    mask_path = HERE / "derived/tb_is_derivation.npy"

    for path in [mc_path, tb_path, mask_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    mask_hash = sha256_file(mask_path)
    mask = np.load(mask_path, allow_pickle=False)

    source, source_e, mc_rows, classes, checkpoint = read_class(
        mc_path, args.class_name
    )
    target, target_e, tb_rows, tb_classes, tb_checkpoint = read_class(
        tb_path, args.class_name, mask=mask
    )

    if classes != tb_classes or checkpoint != tb_checkpoint:
        raise ValueError("MC/TB class or PID-checkpoint mismatch.")

    common = np.intersect1d(np.unique(source_e), np.unique(target_e))
    if len(common) == 0:
        raise ValueError("No common energies.")

    excluded_tb = np.setdiff1d(np.unique(target_e), common)
    keep_mc = np.isin(source_e, common)
    keep_tb = np.isin(target_e, common)

    source, source_e, mc_rows = (
        source[keep_mc], source_e[keep_mc], mc_rows[keep_mc]
    )
    target, target_e, tb_rows = (
        target[keep_tb], target_e[keep_tb], tb_rows[keep_tb]
    )

    architecture = {
        "in_dim": 64,
        "hidden_dim": args.hidden,
        "num_layers": 3,
        "condition_dim": 32,
    }
    transport = ConditionalTransport(architecture, device=args.device)

    source_mean, source_std = statistics(source)
    target_mean, target_std = statistics(target)

    # One shared energy normalisation based on common beam settings.
    # It does not depend on MC/TB event fractions.
    energy_mean = np.float32(common.mean())
    energy_std = np.float32(common.std())
    if energy_std < 1e-6:
        energy_std = np.float32(1.0)

    transport.stats = {
        "source_mean": source_mean,
        "source_std": source_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }

    source_n = ((source - source_mean) / source_std).astype(np.float32)
    target_n = ((target - target_mean) / target_std).astype(np.float32)

    source_pools = [
        np.flatnonzero(source_e == energy) for energy in common
    ]
    target_pools = [
        np.flatnonzero(target_e == energy) for energy in common
    ]
    rng = np.random.default_rng(args.seed)

    def matched_batch():
        # Draw one multiset of energies, shared by MC and TB.
        # Energies are sampled with equal probability.
        bins = rng.integers(0, len(common), size=args.batch_size)
        source_indices = np.empty(args.batch_size, dtype=np.int64)
        target_indices = np.empty(args.batch_size, dtype=np.int64)

        for k in np.unique(bins):
            positions = np.flatnonzero(bins == k)
            source_indices[positions] = rng.choice(
                source_pools[k], size=len(positions), replace=True
            )
            target_indices[positions] = rng.choice(
                target_pools[k], size=len(positions), replace=True
            )

        # Same energy at each batch position, but independent events.
        np.testing.assert_array_equal(
            source_e[source_indices], target_e[target_indices]
        )

        z = torch.from_numpy(source_n[source_indices]).to(args.device)
        y = torch.from_numpy(target_n[target_indices]).to(args.device)
        e = torch.from_numpy(
            transport.normalise_energy(common[bins])
        ).to(args.device)

        return z, y, e

    metadata = {
        "class_name": args.class_name,
        "classes": classes,
        "checkpoint_sha256": checkpoint,
        "tb_split_sha256": mask_hash,
        "source_events": len(source),
        "target_derivation_events": len(target),
        "common_energies": common.tolist(),
        "excluded_tb_energies": excluded_tb.tolist(),
        "architecture": architecture,
        "batch_size": args.batch_size,
        "outer_steps": args.steps,
        "identity_steps": args.identity_steps,
        "g_updates": 10,
        "f_updates": 4,
        "lr": 5e-4,
        "min_lr": 5e-6,
        "betas": [0.0, 0.9],
        "weight_decay": 0.0,
        "grad_clip": 10.0,
        "seed": args.seed,
        "device": args.device,
        "energy_sampling": (
            "Uniform over common energies; identical energy multiset "
            "for source and target; independent events."
        ),
        "latent_statistics": (
            "Separate domain statistics on all selected fitting events."
        ),
        "energy_statistics": (
            "Shared mean/std of the unique common beam energies."
        ),
        "TB_validation_used": False,
    }

    args.out.mkdir(parents=True)
    (args.out / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    np.savez(
        args.out / "selection.npz",
        mc_rows=mc_rows,
        tb_derivation_rows=tb_rows,
        mc_energy=source_e,
        tb_energy=target_e,
        common_energies=common,
    )

    print(f"Class: {args.class_name}", flush=True)
    print(f"Source MC: {len(source):,}", flush=True)
    print(f"Target TB derivation: {len(target):,}", flush=True)
    print(f"Common energies: {common.tolist()}", flush=True)
    print(f"Excluded TB energies: {excluded_tb.tolist()}", flush=True)
    print(f"Device: {args.device}", flush=True)
    print("Energy-matched batches; gradients only with respect to z.",
          flush=True)

    start_time = time.perf_counter()
    transport.f.train()
    transport.g.train()

    def new_optimizer(parameters):
        return torch.optim.Adam(
            parameters, lr=5e-4, betas=(0.0, 0.9), weight_decay=0.0
        )

    pre_optimizer = new_optimizer(transport.g.parameters())
    identity_history = []

    for step in range(1, args.identity_steps + 1):
        z, _, e = matched_batch()
        mapped = conditional_gradient(
            transport.g, z, e, create_graph=True
        )
        loss = F.mse_loss(mapped, z)
        optimizer_step(loss, pre_optimizer, transport.g.parameters())

        if step == 1 or step % args.log_every == 0 or step == args.identity_steps:
            identity_history.append({"step": step, "mse": loss.item()})
            print(
                f"Identity {step}/{args.identity_steps} "
                f"| MSE={loss.item():.6g}", flush=True
            )

    (args.out / "identity_history.json").write_text(
        json.dumps(identity_history, indent=2), encoding="utf-8"
    )

    optimizer_g = new_optimizer(transport.g.parameters())
    optimizer_f = new_optimizer(transport.f.parameters())
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=args.steps, eta_min=5e-6
    )
    scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_f, T_max=args.steps, eta_min=5e-6
    )

    monitor_rng = np.random.default_rng(args.seed + 1)
    monitor_indices = monitor_rng.choice(
        len(source), size=min(1024, len(source)), replace=False
    )
    monitor_z = torch.from_numpy(source_n[monitor_indices]).to(args.device)
    monitor_e = torch.from_numpy(
        transport.normalise_energy(source_e[monitor_indices])
    ).to(args.device)

    history = []
    fields = [
        "outer_step", "loss_g", "loss_f",
        "mean_l2_standardised", "lr_used",
    ]

    with (args.out / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for step in range(1, args.steps + 1):
            lr_used = optimizer_g.param_groups[0]["lr"]
            losses_g, losses_f = [], []

            transport.f.requires_grad_(False)
            try:
                for _ in range(10):
                    z, _, e = matched_batch()
                    mapped = conditional_gradient(
                        transport.g, z, e, create_graph=True
                    )
                    loss_g = (
                        transport.f(mapped, e).squeeze(1)
                        - (z * mapped).sum(dim=1)
                    ).mean()
                    optimizer_step(
                        loss_g, optimizer_g, transport.g.parameters()
                    )
                    losses_g.append(loss_g.item())
            finally:
                transport.f.requires_grad_(True)

            for _ in range(4):
                z, y, e = matched_batch()
                mapped = conditional_gradient(
                    transport.g, z, e, create_graph=False
                ).detach()

                loss_f = (
                    transport.f(y, e).mean()
                    - transport.f(mapped, e).mean()
                    + (z * mapped).sum(dim=1).mean()
                )
                optimizer_step(
                    loss_f, optimizer_f, transport.f.parameters()
                )
                losses_f.append(loss_f.item())

            scheduler_g.step()
            scheduler_f.step()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                mapped = conditional_gradient(
                    transport.g, monitor_z, monitor_e,
                    create_graph=False,
                ).detach()

                if not torch.isfinite(mapped).all():
                    raise FloatingPointError("Non-finite monitor output.")

                displacement = (
                    mapped - monitor_z
                ).norm(dim=1).mean().item()

                record = {
                    "outer_step": step,
                    "loss_g": float(np.mean(losses_g)),
                    "loss_f": float(np.mean(losses_f)),
                    "mean_l2_standardised": displacement,
                    "lr_used": lr_used,
                }
                history.append(record)
                writer.writerow(record)
                stream.flush()

                print(
                    f"OT {step}/{args.steps} "
                    f"| g={record['loss_g']:.6g} "
                    f"| f={record['loss_f']:.6g} "
                    f"| displacement={displacement:.6g} "
                    f"| lr={lr_used:.3g}",
                    flush=True,
                )

    if sha256_file(mask_path) != mask_hash:
        raise RuntimeError("Original TB mask changed.")

    check_rows = monitor_indices[:256]
    before = transport.transform(
        source[check_rows], source_e[check_rows], batch_size=128
    )

    path = args.out / "map.pt"
    transport.save(path, metadata, history)

    loaded = ConditionalTransport.load(path, device=args.device)
    after = loaded.transform(
        source[check_rows], source_e[check_rows], batch_size=128
    )
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    elapsed = time.perf_counter() - start_time
    verification = {
        "reload_invariance_passed": True,
        "max_absolute_difference": float(np.max(np.abs(before - after))),
        "elapsed_seconds": elapsed,
        "TB_validation_used": False,
    }
    (args.out / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    print("\nPASS: reload invariance.", flush=True)
    print(f"Training time: {elapsed:.1f} seconds", flush=True)
    print(f"Saved map: {path}", flush=True)


if __name__ == "__main__":
    main()