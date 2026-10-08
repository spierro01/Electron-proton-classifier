"""Energy-conditioned convex potentials for latent-space OT.

For each fixed energy E, the potential is convex in z.
The energy is a condition, not a transported coordinate.

Inputs:
    z: (B, 64), standardised latent vectors
    energy: (B, 1), standardised beam energy

Output:
    scalar potential: (B, 1)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """Non-negative weights with fan-in-scaled initialisation."""

    def __init__(self, in_features, out_features):
        super().__init__()

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


class ConditionalICNN(nn.Module):
    """Partially input-convex potential: convex in z at fixed energy."""

    def __init__(
        self,
        in_dim=64,
        hidden_dim=128,
        num_layers=3,
        condition_dim=32,
        input_modulation=False,
    ):
        super().__init__()

        if min(in_dim, hidden_dim, num_layers, condition_dim) < 1:
            raise ValueError("All architecture dimensions must be positive.")

        self.in_dim = in_dim
        self.input_modulation = input_modulation

        # This branch depends ONLY on energy.
        # Its weights do not require convexity constraints.
        self.energy_encoder = nn.Sequential(
            nn.Linear(1, condition_dim),
            nn.Tanh(),
            nn.Linear(condition_dim, condition_dim),
            nn.Tanh(),
        )

        # Affine functions of z remain convex at each fixed energy.
        self.z_layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # At fixed energy, this rescaling is linear in z.
        if input_modulation:
            self.input_gates = nn.ModuleList([
                nn.Linear(condition_dim, in_dim)
                for _ in range(num_layers)
            ])

        self.energy_biases = nn.ModuleList([
            nn.Linear(condition_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Convex hidden-to-hidden path.
        self.hidden_layers = nn.ModuleList([
            PositiveLinear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        self.hidden_gates = nn.ModuleList([
            nn.Linear(condition_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        self.output_hidden = PositiveLinear(hidden_dim, 1)
        self.output_gate = nn.Linear(condition_dim, hidden_dim)

        # Energy-dependent affine term in z.
        self.affine_coefficients = nn.Linear(condition_dim, in_dim)
        self.energy_offset = nn.Linear(condition_dim, 1)

        # Positive, trainable diagonal quadratic coefficients.
        # Initial coefficient equals one.
        self.raw_quadratic = nn.Parameter(
            torch.full(
                (in_dim,),
                math.log(math.expm1(1.0)),
            )
        )

        self._initialise()

    def _initialise(self):
        for layer in self.z_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

        # Start with a weak energy-dependent perturbation.
        for layer in self.energy_biases:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

        for layer in list(self.hidden_gates) + [self.output_gate]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

        # Initialise input modulation close to one.
        if self.input_modulation:
            for layer in self.input_gates:
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.ones_(layer.bias)

        nn.init.zeros_(self.affine_coefficients.weight)
        nn.init.zeros_(self.affine_coefficients.bias)
        nn.init.zeros_(self.energy_offset.weight)
        nn.init.zeros_(self.energy_offset.bias)

    @staticmethod
    def activation(x):
        # Convex and non-decreasing.
        return F.softplus(x) - math.log(2.0)

    @staticmethod
    def bounded_gate(x):
        # Positive and bounded: 0 < gate < 1.
        return torch.sigmoid(x)

    def forward(self, z, energy):
        if z.ndim != 2 or z.shape[1] != self.in_dim:
            raise ValueError(f"Expected z with shape (B, {self.in_dim}).")

        if energy.ndim == 1:
            energy = energy[:, None]

        if energy.shape != (len(z), 1):
            raise ValueError("Expected energy with shape (B, 1).")

        condition = self.energy_encoder(energy)

        def z_input(index):
            if self.input_modulation:
                return z * self.input_gates[index](condition)
            return z

        h = self.activation(
            self.z_layers[0](z_input(0))
            + self.energy_biases[0](condition)
        )

        for index, positive_layer in enumerate(self.hidden_layers):
            gate = self.bounded_gate(
                self.hidden_gates[index](condition)
            )

            h = self.activation(
                positive_layer(h * gate)
                + self.z_layers[index + 1](z_input(index + 1))
                + self.energy_biases[index + 1](condition)
            )

        output_gate = self.bounded_gate(
            self.output_gate(condition)
        )

        convex_output = self.output_hidden(h * output_gate)

        affine_output = (
            self.affine_coefficients(condition) * z
        ).sum(dim=1, keepdim=True)

        quadratic = 0.5 * (
            F.softplus(self.raw_quadratic) * z.square()
        ).sum(dim=1, keepdim=True)

        return (
            convex_output
            + affine_output
            + quadratic
            + self.energy_offset(condition)
        )


def conditional_gradient(potential, z, energy, *, create_graph):
    """Return grad_z potential(z, E), keeping E fixed."""

    with torch.enable_grad():
        z_grad = z.detach().requires_grad_(True)

        # E is not a transported coordinate.
        fixed_energy = energy.detach()

        values = potential(z_grad, fixed_energy).sum()

        gradient = torch.autograd.grad(
            values,
            z_grad,
            create_graph=create_graph,
        )[0]

    return gradient


def smoke_test():
    """Basic shape, gradient and numerical-convexity checks."""

    torch.manual_seed(42)

    potential = ConditionalICNN(input_modulation=True)
    z = torch.randn(16, 64)
    energy = torch.linspace(-1, 1, 16).reshape(-1, 1)

    values = potential(z, energy)
    transported = conditional_gradient(
        potential, z, energy, create_graph=True
    )

    assert values.shape == (16, 1)
    assert transported.shape == (16, 64)
    assert torch.isfinite(transported).all()

    # Verify that transport-based training can backpropagate.
    loss = (transported - z).square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in potential.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in gradients)

    # Jensen check at FIXED energy.
    # This numerical check supplements the architectural argument.
    with torch.no_grad():
        a = torch.randn(16, 64)
        b = torch.randn(16, 64)
        weight = 0.37

        lhs = potential(weight * a + (1 - weight) * b, energy)
        rhs = (
            weight * potential(a, energy)
            + (1 - weight) * potential(b, energy)
        )

        assert torch.all(lhs <= rhs + 1e-4)

    print("PASS: potential shape (16, 1)")
    print("PASS: transport shape (16, 64)")
    print("PASS: finite gradients and backpropagation")
    print("PASS: numerical convexity check at fixed energy")
    print("Architecture check only: no OT map has been trained.")


if __name__ == "__main__":
    smoke_test()