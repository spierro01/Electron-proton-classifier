import torch
import torch.nn as nn

class ICNN(nn.Module):
    # Input-Convex Neural Network
    def __init__(self, in_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers

        # Connessioni dirette dall'input x a ciascun layer nascosto
        self.fc_x = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_out_x = nn.Linear(in_dim, 1)

        # Connessioni convesse tra stati z (pesi >= 0)
        self.fc_z = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers - 1)])
        self.fc_out_z = nn.Linear(hidden_dim, 1, bias=False)

        # Funzione di attivazione convessa e non-decrescente
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.fc_x[0](x))
        for i in range(self.num_layers - 1):
            z = self.act(self.fc_z[i](z) + self.fc_x[i + 1](x))

        out = self.fc_out_z(z) + self.fc_out_x(x)
        return out

    def enforce_convexity(self):
        # Proietta i pesi negativi a zero
        with torch.no_grad():
            for layer in self.fc_z:
                layer.weight.clamp_(min=0)
            self.fc_out_z.weight.clamp_(min=0)


def compute_grad_g(g_net, z):
    # Calcola T(z) = grad g(z) usando PyTorch autograd
    z_in = z.clone().detach().requires_grad_(True)
    g_val = g_net(z_in)

    grad = torch.autograd.grad(
        outputs=g_val.sum(),
        inputs=z_in,
        create_graph=True,
        retain_graph=True
    )[0]
    return grad