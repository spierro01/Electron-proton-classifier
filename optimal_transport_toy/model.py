import torch
import torch.nn as nn

class ICNN(nn.Module):
    """
    Input-Convex Neural Network (ICNN).
    Garantisce la convessità imponendo pesi non-negativi sui layer interni.
    """
    def __init__(self, in_dim=2, hidden_dim=64, num_layers=3):
        super(ICNN, self).__init__()
        self.num_layers = num_layers
        
        # Connessioni dirette dall'input x a ciascun layer (passthrough)
        self.w_x = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(num_layers)])
        
        # Connessioni tra layer nascosti consecutivi (devono avere pesi >= 0)
        self.w_z = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers - 1)])
        
        # Layer finale scalare
        self.w_out = nn.Linear(hidden_dim, 1, bias=False)
        self.w_x_out = nn.Linear(in_dim, 1)

        # Attivazione convessa e non-decrescente (CELU o Softplus)
        self.act = nn.CELU()

    def forward(self, x):
        z = self.act(self.w_x[0](x))
        for i in range(self.num_layers - 1):
            z = self.act(self.w_z[i](z) + self.w_x[i+1](x))
        
        out = self.w_out(z) + self.w_x_out(x)
        return out

    def clamp_weights(self):
        """Impone la non-negatività sui pesi per preservare la convessità."""
        with torch.no_grad():
            for w in self.w_z:
                w.weight.data.clamp_(min=0)
            self.w_out.weight.data.clamp_(min=0)

class NeuralOT(nn.Module):
    """Mappa di trasporto T(z) = grad g(z)"""
    def __init__(self, hidden_dim=64):
        super(NeuralOT, self).__init__()
        self.f = ICNN(in_dim=2, hidden_dim=hidden_dim)
        self.g = ICNN(in_dim=2, hidden_dim=hidden_dim)

    def transport(self, z):
        """Calcola T(z) = grad g(z) tramite autograd."""
        z = z.clone().detach().requires_grad_(True)
        g_val = self.g(z)
        grad_g = torch.autograd.grad(
            outputs=g_val.sum(),
            inputs=z,
            create_graph=True,
            retain_graph=True
        )[0]
        return grad_g