import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Aggiunge la cartella dello script al percorso di Python per trovare models.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import ICNN, compute_grad_g

def load_transport_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    g_net = ICNN(in_dim=2, hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]).to(device)
    g_net.load_state_dict(checkpoint["g_state_dict"])
    g_net.eval()

    stats = checkpoint["stats"]
    return g_net, stats

def transport_samples(g_net, source_data, stats, device, batch_size=512):
    mean_s, std_s = stats["mean_s"], stats["std_s"]
    mean_t, std_t = stats["mean_t"], stats["std_t"]

    src_norm = (source_data - mean_s) / std_s
    src_tensor = torch.tensor(src_norm, dtype=torch.float32, device=device)

    transported_norm = []
    for i in range(0, len(src_tensor), batch_size):
        batch = src_tensor[i:i + batch_size]
        grad = compute_grad_g(g_net, batch)
        transported_norm.append(grad.detach().cpu().numpy())

    transported_norm = np.vstack(transported_norm)
    transported_orig = transported_norm * std_t + mean_t
    return transported_orig

def verify_reloading_invariance(g_net, stats, source_test, device):
    res1 = transport_samples(g_net, source_test[:100], stats, device)
    g_net_reloaded, stats_reloaded = load_transport_model("checkpoints/ot_model.pt", device)
    res2 = transport_samples(g_net_reloaded, source_test[:100], stats_reloaded, device)

    diff = np.max(np.abs(res1 - res2))
    print("\n--- VERIFICA RELOAD ---")
    print(f"Max differenza tra modello originale e ricaricato: {diff:.2e}")
    assert np.allclose(res1, res2, atol=1e-5), "ERRORE: I risultati differiscono dopo il reload!"
    print("VERIFICA SUPERATA: Il modello ricaricato produce risultati identici.")

def compute_roc_auc(X1, X2):
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(len(X1)), np.ones(len(X2))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

def plot_results_with_marginals(source_test, target_test, transported_test):
    fig = plt.figure(figsize=(15, 9))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(source_test[:, 0], source_test[:, 1], alpha=0.3, s=5, c='blue')
    ax1.set_title("Source Test")
    ax1.set_xlim(-5, 5); ax1.set_ylim(-5, 5)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(target_test[:, 0], target_test[:, 1], alpha=0.3, s=5, c='red')
    ax2.set_title("Target Test")
    ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(transported_test[:, 0], transported_test[:, 1], alpha=0.3, s=5, c='purple')
    ax3.set_title("Transported Source Test")
    ax3.set_xlim(-5, 5); ax3.set_ylim(-5, 5)

    ax4 = fig.add_subplot(2, 2, 3)
    ax4.hist(target_test[:, 0], bins=50, density=True, alpha=0.5, color='red', label='Target')
    ax4.hist(transported_test[:, 0], bins=50, density=True, alpha=0.5, color='purple', label='Transported')
    ax4.set_title("Marginale Dimensione 0 (x1)")
    ax4.legend()

    ax5 = fig.add_subplot(2, 2, 4)
    ax5.hist(target_test[:, 1], bins=50, density=True, alpha=0.5, color='red', label='Target')
    ax5.hist(transported_test[:, 1], bins=50, density=True, alpha=0.5, color='purple', label='Transported')
    ax5.set_title("Marginale Dimensione 1 (x2)")
    ax5.legend()

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/toy_ot_results.png")
    print("Grafico completo salvato in figures/toy_ot_results.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join("data", "toy_ot_data.npz")
    data = np.load(data_path)

    source_test = data["source_test"]
    target_test = data["target_test"]

    g_net, stats = load_transport_model("checkpoints/ot_model.pt", device)

    verify_reloading_invariance(g_net, stats, source_test, device)

    transported_test = transport_samples(g_net, source_test, stats, device)

    print("\n--- STATISTICHE PER COORDINATA ---")
    print(f"Source Test Mean:       {source_test.mean(axis=0)} | Std: {source_test.std(axis=0)}")
    print(f"Target Test Mean:       {target_test.mean(axis=0)} | Std: {target_test.std(axis=0)}")
    print(f"Transported Test Mean:  {transported_test.mean(axis=0)} | Std: {transported_test.std(axis=0)}")

    auc_before = compute_roc_auc(source_test, target_test)
    auc_after = compute_roc_auc(transported_test, target_test)

    print("\n--- RISULTATI CLASSIFICATORE ROC AUC ---")
    print(f"ROC AUC prima del trasporto (Source vs Target): {auc_before:.4f}")
    print(f"ROC AUC dopo il trasporto (Transported vs Target): {auc_after:.4f} (Obiettivo: < 0.62)")

    plot_results_with_marginals(source_test, target_test, transported_test)

if __name__ == "__main__":
    main()