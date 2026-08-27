import os
import torch
import torch.optim as optim
from dataset import load_and_preprocess_data
from model import NeuralOT

def train_neural_ot(
    data_path="data/toy_ot_data.npz",
    epochs=40,
    batch_size=256,
    lr=1e-3,
    n_g_steps=1,
    n_f_steps=5,
    save_path="checkpoint.pt"
):
    # Fissa il seed casuale per la riproducibilità
    torch.manual_seed(42)
    
    # Se disponibile usa la GPU (CUDA), altrimenti CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # Carica e standardizza i dati
    source_loader, target_loader, stats = load_and_preprocess_data(data_path, batch_size)

    # Inizializza il modello
    model = NeuralOT(hidden_dim=64).to(device)
    
    # Ottimizzatori separati per i due potenziali
    opt_g = optim.Adam(model.g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_f = optim.Adam(model.f.parameters(), lr=lr, betas=(0.5, 0.999))

    print("Inizio addestramento...")

    for epoch in range(1, epochs + 1):
        loss_g_accum = 0.0
        loss_f_accum = 0.0
        steps = 0

        target_iter = iter(target_loader)

        for z in source_loader:
            try:
                y = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                y = next(target_iter)

            z = z.to(device)
            y = y.to(device)

            # ---------------------------------------------------------
            # 1. Aggiornamento del Potenziale g (Minimizzazione)
            # ---------------------------------------------------------
            for _ in range(n_g_steps):
                opt_g.zero_grad()
                
                # Transport map T(z) = grad g(z)
                grad_g = model.transport(z)
                
                # Objective per g: E[f(grad g(z)) - <z, grad g(z)>]
                f_grad_g = model.f(grad_g)
                dot_product = torch.sum(z * grad_g, dim=1, keepdim=True)
                
                loss_g = torch.mean(f_grad_g - dot_product)
                loss_g.backward()
                opt_g.step()
                
                # Mantiene la convessità di g
                model.g.clamp_weights()

            # ---------------------------------------------------------
            # 2. Aggiornamento del Potenziale f (Massimizzazione)
            # ---------------------------------------------------------
            for _ in range(n_f_steps):
                opt_f.zero_grad()
                
                with torch.no_grad():
                    grad_g = model.transport(z)
                
                f_grad_g = model.f(grad_g)
                f_y = model.f(y)
                
                # Maximise objective <=> Minimise -Objective
                dot_product = torch.sum(z * grad_g, dim=1, keepdim=True)
                loss_f = -torch.mean(f_grad_g - dot_product) + torch.mean(f_y)
                
                loss_f.backward()
                opt_f.step()
                
                # Mantiene la convessità di f
                model.f.clamp_weights()

            loss_g_accum += loss_g.item()
            loss_f_accum += loss_f.item()
            steps += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Loss G: {loss_g_accum/steps:.4f} | Loss F: {loss_f_accum/steps:.4f}")

    # Salvataggio del checkpoint e delle statistiche
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "source_mean": stats["source_stdizer"].mean,
        "source_std": stats["source_stdizer"].std,
        "target_mean": stats["target_stdizer"].mean,
        "target_std": stats["target_stdizer"].std,
    }
    torch.save(checkpoint, save_path)
    print(f"\nModello salvato con successo in '{save_path}'!")

if __name__ == "__main__":
    train_neural_ot()