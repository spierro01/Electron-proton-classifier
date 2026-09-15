"""Neural quadratic-cost optimal transport with ICNNs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """Linear layer with strictly positive weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()

        self.raw_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.bias = (
            nn.Parameter(torch.zeros(out_features))
            if bias
            else None
        )

        nn.init.normal_(
            self.raw_weight,
            mean=-3.0,
            std=0.15,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            F.softplus(self.raw_weight),
            self.bias,
        )


class ICNN(nn.Module):
    """Scalar input-convex neural network."""

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        strong_convexity: float = 0.0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least one")

        self.input_layers = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.hidden_layers = nn.ModuleList(
            [
                PositiveLinear(hidden_dim, hidden_dim)
                for _ in range(num_layers - 1)
            ]
        )

        self.output_hidden = PositiveLinear(hidden_dim, 1)
        self.output_input = nn.Linear(in_dim, 1)

        self.activation = nn.Softplus()
        self.strong_convexity = float(strong_convexity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.input_layers[0](x))

        for hidden_layer, input_layer in zip(
            self.hidden_layers,
            self.input_layers[1:],
        ):
            z = self.activation(
                hidden_layer(z) + input_layer(x)
            )

        output = self.output_hidden(z) + self.output_input(x)

        if self.strong_convexity:
            output = output + (
                0.5
                * self.strong_convexity
                * x.square().sum(dim=1, keepdim=True)
            )

        return output


def gradient_of_potential(
    potential: nn.Module,
    x: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return grad_x potential(x)."""

    x_for_grad = x.detach().requires_grad_(True)

    value = potential(x_for_grad).sum()

    return torch.autograd.grad(
        value,
        x_for_grad,
        create_graph=create_graph,
    )[0]


@dataclass(frozen=True)
class OTConfig:
    in_dim: int = 2
    hidden_dim: int = 64
    num_layers: int = 3

    strong_convexity_g: float = 0.1

    lr_f: float = 2e-4
    lr_g: float = 2e-4

    batch_size: int = 512
    epochs: int = 200

    # Numero di update di f per blocco.
    f_updates_per_g: int = 1

    # Numero di update di g dopo il blocco di update di f.
    g_updates_per_f: int = 1

    grad_clip_norm: float = 5.0
    seed: int = 42


class NeuralOT:
    """Fitted OT map T(x) = grad g(x)."""

    def __init__(
        self,
        config: OTConfig = OTConfig(),
        device: Optional[torch.device] = None,
    ):
        self.config = config

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
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
            strong_convexity=config.strong_convexity_g,
        ).to(self.device)

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Call fit() before transform() or apply()."
            )

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()

        return (
            np.asarray(x, dtype=np.float32) - self.mean
        ) / self.std

    def fit(
        self,
        source_train: np.ndarray,
        target_train: np.ndarray,
    ) -> "NeuralOT":
        source_train = np.asarray(
            source_train,
            dtype=np.float32,
        )
        target_train = np.asarray(
            target_train,
            dtype=np.float32,
        )

        if (
            source_train.ndim != 2
            or target_train.shape != source_train.shape
        ):
            raise ValueError(
                "Source and target arrays must have the same 2D shape."
            )

        if source_train.shape[1] != self.config.in_dim:
            raise ValueError(
                "Input data dimension does not match config.in_dim."
            )

        # Common train-only coordinate system.
        pooled = np.vstack([source_train, target_train])

        self.mean = pooled.mean(axis=0).astype(np.float32)
        self.std = pooled.std(axis=0).clip(1e-6).astype(np.float32)

        source = torch.as_tensor(
            self._standardise(source_train),
            device=self.device,
        )

        target = torch.as_tensor(
            self._standardise(target_train),
            device=self.device,
        )

        optimizer_f = torch.optim.Adam(
            self.f.parameters(),
            lr=self.config.lr_f,
            betas=(0.5, 0.9),
        )

        optimizer_g = torch.optim.Adam(
            self.g.parameters(),
            lr=self.config.lr_g,
            betas=(0.5, 0.9),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)

        n_samples = len(source)

        self.f.train()
        self.g.train()

        for epoch in range(1, self.config.epochs + 1):
            source_permutation = torch.randperm(
                n_samples,
                generator=generator,
            )

            target_permutation = torch.randperm(
                n_samples,
                generator=generator,
            )

            f_losses = []
            g_losses = []

            for start in range(
                0,
                n_samples - self.config.batch_size + 1,
                self.config.batch_size,
            ):
                source_indices = source_permutation[
                    start:start + self.config.batch_size
                ].to(self.device)

                target_indices = target_permutation[
                    start:start + self.config.batch_size
                ].to(self.device)

                source_batch = source[source_indices]
                target_batch = target[target_indices]

                # ------------------------------------------------
                # max_f: T_g is treated as fixed.
                # ------------------------------------------------
                transported_fixed = gradient_of_potential(
                    self.g,
                    source_batch,
                    create_graph=False,
                ).detach()

                dot_fixed = (
                    source_batch * transported_fixed
                ).sum(dim=1, keepdim=True)

                for _ in range(self.config.f_updates_per_g):
                    optimizer_f.zero_grad(set_to_none=True)

                    # Negative of:
                    # E_source[f(T_g(x)) - <x, T_g(x)>] - E_target[f(y)]
                    loss_f = self.f(target_batch).mean() - (
                        self.f(transported_fixed) - dot_fixed
                    ).mean()

                    loss_f.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.f.parameters(),
                        self.config.grad_clip_norm,
                    )

                    optimizer_f.step()

                    f_losses.append(float(loss_f.detach()))

                # ------------------------------------------------
                # min_g: f parameters are frozen, but df/dT is kept.
                # ------------------------------------------------
                for parameter in self.f.parameters():
                    parameter.requires_grad_(False)

                for _ in range(self.config.g_updates_per_f):
                    optimizer_g.zero_grad(set_to_none=True)

                    transported = gradient_of_potential(
                        self.g,
                        source_batch,
                        create_graph=True,
                    )

                    dot_term = (
                        source_batch * transported
                    ).sum(dim=1, keepdim=True)

                    loss_g = (
                        self.f(transported) - dot_term
                    ).mean()

                    loss_g.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.g.parameters(),
                        self.config.grad_clip_norm,
                    )

                    optimizer_g.step()

                    g_losses.append(float(loss_g.detach()))

                for parameter in self.f.parameters():
                    parameter.requires_grad_(True)

            if (
                epoch == 1
                or epoch % 10 == 0
                or epoch == self.config.epochs
            ):
                print(
                    f"epoch {epoch:03d}/{self.config.epochs} | "
                    f"loss_f={np.mean(f_losses):.5f} | "
                    f"loss_g={np.mean(g_losses):.5f}"
                )

        return self

    def transform(
        self,
        source: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Apply T(x) = grad g(x) in batches."""

        source_normalised = self._standardise(source)

        self.g.eval()

        batches = []

        with torch.enable_grad():
            for start in range(
                0,
                len(source_normalised),
                batch_size,
            ):
                batch = torch.as_tensor(
                    source_normalised[start:start + batch_size],
                    device=self.device,
                )

                mapped = gradient_of_potential(
                    self.g,
                    batch,
                    create_graph=False,
                )

                batches.append(mapped.detach().cpu().numpy())

        mapped_normalised = np.vstack(batches)

        return mapped_normalised * self.std + self.mean

    apply = transform

    def save(self, path: str | Path) -> None:
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "config": asdict(self.config),
                "f_state_dict": self.f.state_dict(),
                "g_state_dict": self.g.state_dict(),
                "standardisation": {
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                },
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "NeuralOT":
        try:
            checkpoint: Dict[str, Any] = torch.load(
                path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location=device,
            )

        model = cls(
            OTConfig(**checkpoint["config"]),
            device=device,
        )

        model.f.load_state_dict(checkpoint["f_state_dict"])
        model.g.load_state_dict(checkpoint["g_state_dict"])

        model.mean = np.asarray(
            checkpoint["standardisation"]["mean"],
            dtype=np.float32,
        )

        model.std = np.asarray(
            checkpoint["standardisation"]["std"],
            dtype=np.float32,
        )

        model.f.eval()
        model.g.eval()

        return model