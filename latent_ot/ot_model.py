"""Neural quadratic-cost OT for latent-space calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """Positive weights, with fan-in-scaled initialization."""

    def __init__(self, in_features, out_features):
        super().__init__()

        # Initial positive weights approximately 1 / in_features.
        initial_weight = 1.0 / in_features
        initial_raw = math.log(math.expm1(initial_weight))

        self.raw_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.normal_(
            self.raw_weight,
            mean=initial_raw,
            std=0.05,
        )

    def forward(self, x):
        return F.linear(x, F.softplus(self.raw_weight))


class ICNN(nn.Module):
    """Convex scalar potential with a trainable positive quadratic term."""

    def __init__(self, in_dim=64, hidden_dim=2048, num_layers=3):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be positive.")

        self.input_layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.hidden_layers = nn.ModuleList([
            PositiveLinear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        self.output_hidden = PositiveLinear(hidden_dim, 1)
        self.output_input = nn.Linear(in_dim, 1)

        # Positive, learnable diagonal quadratic coefficients.
        # Initially equal to 1: approximately identity transport.
        self.raw_quadratic = nn.Parameter(
            torch.full((in_dim,), math.log(math.expm1(1.0)))
        )

        # Small input contributions at initialization.
        for layer in self.input_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

        nn.init.zeros_(self.output_input.weight)
        nn.init.zeros_(self.output_input.bias)

    @staticmethod
    def activation(x):
        # Convex, non-decreasing; subtracting a constant preserves convexity.
        return F.softplus(x) - math.log(2.0)

    def forward(self, x):
        h = self.activation(self.input_layers[0](x))

        for hidden, direct in zip(
            self.hidden_layers,
            self.input_layers[1:],
        ):
            h = self.activation(hidden(h) + direct(x))

        quadratic = 0.5 * (
            F.softplus(self.raw_quadratic) * x.square()
        ).sum(dim=1, keepdim=True)

        return (
            self.output_hidden(h)
            + self.output_input(x)
            + quadratic
        )


def gradient_of_potential(potential, x, *, create_graph):
    """Differentiate the scalar potential with respect to its input."""
    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        values = potential(x_grad).sum()

        return torch.autograd.grad(
            values,
            x_grad,
            create_graph=create_graph,
        )[0]


@dataclass(frozen=True)
class OTConfig:
    in_dim: int = 64
    hidden_dim: int = 2048
    num_layers: int = 3

    batch_size: int = 1024
    outer_steps: int = 10_000
    g_updates: int = 10
    f_updates: int = 4

    identity_steps: int = 1000

    lr: float = 5e-4
    min_lr: float = 5e-6
    beta1: float = 0.0
    beta2: float = 0.9

    grad_clip: float = 10.0
    log_every: int = 100
    seed: int = 42


class NeuralOT:
    def __init__(self, config=OTConfig(), device=None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.f = ICNN(
            config.in_dim,
            config.hidden_dim,
            config.num_layers,
        ).to(self.device)

        self.g = ICNN(
            config.in_dim,
            config.hidden_dim,
            config.num_layers,
        ).to(self.device)

        self.source_mean = None
        self.source_std = None
        self.target_mean = None
        self.target_std = None
        self.history = []

    def _check_fitted(self):
        if self.source_mean is None or self.target_mean is None:
            raise RuntimeError("The map has no fitted normalisation.")

    @staticmethod
    def _validate_array(x, dimension, name):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2 or x.shape[1] != dimension or len(x) == 0:
            raise ValueError(
                f"{name} must have shape (N, {dimension}), with N > 0."
            )

        if not np.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values.")

        return x

    @staticmethod
    def _statistics(x):
        mean = x.mean(axis=0, dtype=np.float64)
        std = x.std(axis=0, dtype=np.float64)

        # Center constant/nearly constant dimensions without division by zero.
        std = np.where(std < 1e-6, 1.0, std)

        return mean.astype(np.float32), std.astype(np.float32)

    def _optimizer_step(self, loss, optimizer, parameters):
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite loss.")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters,
            self.config.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()

    def fit(self, source_train, target_train, log_callback=None):
        """Fit on unpaired source and target arrays of possibly unequal size."""
        c = self.config

        if min(
            c.batch_size, c.outer_steps, c.g_updates,
            c.f_updates, c.log_every
        ) <= 0 or c.identity_steps < 0:
            raise ValueError("Invalid training configuration.")

        if not 0 < c.min_lr <= c.lr:
            raise ValueError("Require 0 < min_lr <= lr.")

        source_train = self._validate_array(
            source_train, c.in_dim, "source"
        )
        target_train = self._validate_array(
            target_train, c.in_dim, "target"
        )

        self.source_mean, self.source_std = self._statistics(source_train)
        self.target_mean, self.target_std = self._statistics(target_train)

        # Keep the datasets on CPU; transfer only minibatches.
        source = (
            (source_train - self.source_mean) / self.source_std
        ).astype(np.float32)

        target = (
            (target_train - self.target_mean) / self.target_std
        ).astype(np.float32)

        rng = np.random.default_rng(c.seed)
        self.history = []

        def sample(array):
            indices = rng.integers(0, len(array), size=c.batch_size)
            return torch.from_numpy(array[indices]).to(self.device)

        self.f.train()
        self.g.train()

        # ------------------------------------------------------------
        # Identity pre-training: grad g(z) approximately z.
        # This is not source-target event matching.
        # ------------------------------------------------------------
        pre_optimizer = torch.optim.Adam(
            self.g.parameters(),
            lr=c.lr,
            betas=(c.beta1, c.beta2),
            weight_decay=0.0,
        )

        for step in range(1, c.identity_steps + 1):
            z = sample(source)
            pre_optimizer.zero_grad(set_to_none=True)

            transported = gradient_of_potential(
                self.g, z, create_graph=True
            )
            loss = F.mse_loss(transported, z)

            self._optimizer_step(
                loss, pre_optimizer, self.g.parameters()
            )

            if (
                step == 1
                or step % c.log_every == 0
                or step == c.identity_steps
            ):
                print(
                    f"Identity {step}/{c.identity_steps} "
                    f"| MSE={loss.item():.6g}",
                    flush=True,
                )

        # Start OT with fresh optimizer states.
        optimizer_g = torch.optim.Adam(
            self.g.parameters(),
            lr=c.lr,
            betas=(c.beta1, c.beta2),
            weight_decay=0.0,
        )
        optimizer_f = torch.optim.Adam(
            self.f.parameters(),
            lr=c.lr,
            betas=(c.beta1, c.beta2),
            weight_decay=0.0,
        )

        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_g, T_max=c.outer_steps, eta_min=c.min_lr
        )
        scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_f, T_max=c.outer_steps, eta_min=c.min_lr
        )

        # Fixed source-only sample for displacement diagnostics.
        monitor_rng = np.random.default_rng(c.seed + 1)
        monitor_indices = monitor_rng.choice(
            len(source),
            size=min(1024, len(source)),
            replace=False,
        )
        monitor_z = torch.from_numpy(
            source[monitor_indices]
        ).to(self.device)

        target_std = torch.from_numpy(self.target_std).to(self.device)
        target_mean = torch.from_numpy(self.target_mean).to(self.device)
        original_monitor = torch.from_numpy(
            source_train[monitor_indices]
        ).to(self.device)

        # ------------------------------------------------------------
        # Each outer step: 10 g-updates, then 4 f-updates by default.
        # ------------------------------------------------------------
        for step in range(1, c.outer_steps + 1):
            lr_used = optimizer_g.param_groups[0]["lr"]
            g_losses = []
            f_losses = []

            self.f.requires_grad_(False)

            try:
                for _ in range(c.g_updates):
                    z = sample(source)
                    optimizer_g.zero_grad(set_to_none=True)

                    transported = gradient_of_potential(
                        self.g, z, create_graph=True
                    )

                    # min_g E_source[f(T(z)) - <z,T(z)>]
                    loss_g = (
                        self.f(transported).squeeze(1)
                        - (z * transported).sum(dim=1)
                    ).mean()

                    self._optimizer_step(
                        loss_g, optimizer_g, self.g.parameters()
                    )
                    g_losses.append(float(loss_g.detach()))
            finally:
                self.f.requires_grad_(True)

            for _ in range(c.f_updates):
                z = sample(source)
                y = sample(target)

                transported = gradient_of_potential(
                    self.g, z, create_graph=False
                ).detach()

                optimizer_f.zero_grad(set_to_none=True)

                # Negative dual objective: minimizing this maximizes over f.
                loss_f = (
                    self.f(y).mean()
                    - self.f(transported).mean()
                    + (z * transported).sum(dim=1).mean()
                )

                self._optimizer_step(
                    loss_f, optimizer_f, self.f.parameters()
                )
                f_losses.append(float(loss_f.detach()))

            scheduler_g.step()
            scheduler_f.step()

            if (
                step == 1
                or step % c.log_every == 0
                or step == c.outer_steps
            ):
                mapped = gradient_of_potential(
                    self.g, monitor_z, create_graph=False
                ).detach()

                if not torch.isfinite(mapped).all().item():
                    raise FloatingPointError("Non-finite transport output.")

                with torch.no_grad():
                    difference = mapped - monitor_z
                    mean_l2 = difference.norm(dim=1).mean().item()
                    rms_coordinate = difference.square().mean().sqrt().item()

                    # Actual displacement in the original latent coordinates.
                    calibrated = mapped * target_std + target_mean
                    original_l2 = (
                        calibrated - original_monitor
                    ).norm(dim=1).mean().item()

                record = {
                    "outer_step": step,
                    "loss_g": float(np.mean(g_losses)),
                    "loss_f": float(np.mean(f_losses)),
                    "mean_l2_standardised": mean_l2,
                    "rms_coordinate_standardised": rms_coordinate,
                    "mean_l2_original_latent": original_l2,
                    "lr_used": lr_used,
                }
                self.history.append(record)

                if log_callback is not None:
                    log_callback(record)

                print(
                    f"OT {step}/{c.outer_steps} "
                    f"| g={record['loss_g']:.6g} "
                    f"| f={record['loss_f']:.6g} "
                    f"| displacement={mean_l2:.6g} "
                    f"| lr={lr_used:.3g}",
                    flush=True,
                )

        return self

    def transform(self, source, batch_size=1024):
        """Return calibrated vectors in original target latent coordinates."""
        self._check_fitted()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        source = self._validate_array(
            source, self.config.in_dim, "source"
        )
        result = np.empty_like(source)

        self.g.eval()

        for start in range(0, len(source), batch_size):
            stop = min(start + batch_size, len(source))

            normalised = (
                source[start:stop] - self.source_mean
            ) / self.source_std

            tensor = torch.from_numpy(
                normalised.astype(np.float32)
            ).to(self.device)

            mapped = gradient_of_potential(
                self.g, tensor, create_graph=False
            ).detach().cpu().numpy()

            result[start:stop] = (
                mapped * self.target_std + self.target_mean
            )

        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite calibrated latents.")

        return result

    apply = transform

    def save(self, path, metadata=None):
        self._check_fitted()

        path = Path(path)
        if path.exists():
            raise FileExistsError(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "config": asdict(self.config),
            "f_state_dict": self.f.state_dict(),
            "g_state_dict": self.g.state_dict(),
            "standardisation": {
                "source_mean": self.source_mean.tolist(),
                "source_std": self.source_std.tolist(),
                "target_mean": self.target_mean.tolist(),
                "target_std": self.target_std.tolist(),
            },
            "metadata": metadata or {},
            "history": self.history,
        }, path)

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )

        transport = cls(
            OTConfig(**checkpoint["config"]),
            device=device,
        )

        transport.f.load_state_dict(checkpoint["f_state_dict"])
        transport.g.load_state_dict(checkpoint["g_state_dict"])

        for name, values in checkpoint["standardisation"].items():
            setattr(
                transport,
                name,
                np.asarray(values, dtype=np.float32),
            )

        transport.history = checkpoint.get("history", [])
        transport.f.eval()
        transport.g.eval()

        return transport