import os
import sys
import numpy as np
import torch
import torch.optim as optim

# Aggiunge la cartella dello script al percorso di Python per trovare models.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ICNN, compute_grad_g

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilizzato: {device}")

    # Caricamento dati training
    data_path = os.path.join("data", "toy_ot_data.npz")
    data = np.load(data_path)
    source_train = data["source_train"]
    target_train = data["target_train"]

    # Standardizzazione solo sul train set
    mean_s, std_s = source_train.mean(axis=0), source_train.std(axis=0)
    mean_t, std_t = target_train.mean(axis=0), target_train.std(axis=0)

    # Normalizzazione
    src_tr_norm = (source_train - mean_s) / std_s
    tgt_tr_norm = (target_train - mean_t) / std_t

    src_tensor = torch.tensor(src_tr_norm, dtype=torch.float32, device=device)
    tgt_tensor = torch.tensor(tgt_tr_norm, dtype=torch.float32, device=device)

    # Inizializzazione delle due ICNN
    f_net = ICNN(in_dim=2, hidden_dim=64, num_layers=3).to(device)
    g_net = ICNN(in_dim=2, hidden_dim=64, num_layers=3).to(device)

    # Learning rate
    opt_f = optim.Adam(f_net.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_g = optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.9))

    batch_size = 512
    epochs = 100
    n_f_updates = 5  # 5 aggiornamenti di f per ogni aggiornamento di g
    num_samples = len(source_train)

    print("Inizio Addestramento (con Gradient Clipping)...")
    for epoch in range(1, epochs + 1):
        perm_s = torch.randperm(num_samples)
        perm_t = torch.randperm(num_samples)

        epoch_loss_f = 0.0
        epoch_loss_g = 0.0
        steps = 0

        for i in range(0, num_samples, batch_size):
            idx_s = perm_s[i:i + batch_size]
            idx_t = perm_t[i:i + batch_size]
            if len(idx_s) < batch_size or len(idx_t) < batch_size:
                continue

            z_b = src_tensor[idx_s]
            y_b = tgt_tensor[idx_t]

            # Inner loop: aggiornamento rete f (max)
            for _ in range(n_f_updates):
                opt_f.zero_grad()
                grad_g_z = compute_grad_g(g_net, z_b)
                dot_term = (z_b * grad_g_z).sum(dim=1, keepdim=True)
                f_grad_g = f_net(grad_g_z)
                f_y = f_net(y_b)

                loss_f = f_y.mean() - (f_grad_g - dot_term).mean()
                loss_f.backward()

                torch.nn.utils.clip_grad_norm_(f_net.parameters(), max_norm=1.0)
                opt_f.step()
                f_net.enforce_convexity()

            # Outer step: aggiornamento rete g (min)
            opt_g.zero_grad()
            grad_g_z = compute_grad_g(g_net, z_b)
            dot_term = (z_b * grad_g_z).sum(dim=1, keepdim=True)
            f_grad_g = f_net(grad_g_z)

            loss_g = (f_grad_g - dot_term).mean()
            loss_g.backward()

            torch.nn.utils.clip_grad_norm_(g_net.parameters(), max_norm=1.0)
            opt_g.step()
            g_net.enforce_convexity()

            epoch_loss_f += loss_f.item()
            epoch_loss_g += loss_g.item()
            steps += 1

        if epoch % 10 == 0 or epoch == 1:
            avg_f = epoch_loss_f / steps if steps > 0 else 0
            avg_g = epoch_loss_g / steps if steps > 0 else 0
            print(f"Epoch {epoch:03d}/{epochs:03d} | Avg Loss F: {avg_f:.4f} | Avg Loss G: {avg_g:.4f}")

    # Salvataggio checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "g_state_dict": g_net.state_dict(),
        "f_state_dict": f_net.state_dict(),
        "stats": {
            "mean_s": mean_s, "std_s": std_s,
            "mean_t": mean_t, "std_t": std_t
        },
        "config": {"hidden_dim": 64, "num_layers": 3}
    }
    torch.save(checkpoint, "checkpoints/ot_model.pt")
    print("\nModello salvato con successo in checkpoints/ot_model.pt")

if __name__ == "__main__":
    main()