import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from dataset import load_and_preprocess_data
from model import NeuralOT

def evaluate_model(checkpoint_path="checkpoint.pt", data_path="data/toy_ot_data.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica dati di test reali e trasformati
    _, _, stats = load_and_preprocess_data(data_path)
    source_test_scaled = stats["source_test"]
    target_test_raw = stats["target_test_raw"]
    
    # Carica il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NeuralOT(hidden_dim=64).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Trasporto di source_test
    source_test_tensor = torch.tensor(source_test_scaled, dtype=torch.float32).to(device)
    transported_scaled = model.transport(source_test_tensor).detach().cpu().numpy()

    # Inversione della standardizzazione per tornare nelle coordinate originali
    target_stdizer = stats["target_stdizer"]
    transported_raw = target_stdizer.inverse_transform(transported_scaled)

    # ---------------------------------------------------------
    # 1. Metriche Statistiche (Media e Std)
    # ---------------------------------------------------------
    print("\n=== CONFRONTO STATISTICO (Coordinate Reali) ===")
    print(f"Target Test   - Media: {target_test_raw.mean(axis=0)}, Std: {target_test_raw.std(axis=0)}")
    print(f"Transported   - Media: {transported_raw.mean(axis=0)}, Std: {transported_raw.std(axis=0)}")

    # ---------------------------------------------------------
    # 2. Due-Sample Classifier ROC AUC
    # ---------------------------------------------------------
    print("\n=== VALUTAZIONE ROC AUC ===")
    
    def compute_auc(X_gen, X_real):
        X = np.vstack([X_gen, X_real])
        y = np.hstack([np.zeros(len(X_gen)), np.ones(len(X_real))])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    auc_before = compute_auc(stats["source_test_raw"], target_test_raw)
    auc_after = compute_auc(transported_raw, target_test_raw)

    print(f"ROC AUC prima del trasporto (Source vs Target): {auc_before:.4f}")
    print(f"ROC AUC dopo il trasporto (Transported vs Target): {auc_after:.4f}")
    
    if auc_after < 0.62:
        print("-> OBIETTIVO RAGGIUNTO: ROC AUC inferiore a 0.62!")
    else:
        print("-> NOTA: L'AUC supera 0.62. Potrebbe servire un addestramento con più epoche.")

    # ---------------------------------------------------------
    # 3. Generazione dei Grafici di Confronto
    # ---------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

    axes[0].scatter(stats["source_test_raw"][:, 0], stats["source_test_raw"][:, 1], alpha=0.3, s=5, c="blue")
    axes[0].set_title("Source Test Originale")
    
    axes[1].scatter(target_test_raw[:, 0], target_test_raw[:, 1], alpha=0.3, s=5, c="red")
    axes[1].set_title("Target Test Reale")

    axes[2].scatter(transported_raw[:, 0], transported_raw[:, 1], alpha=0.3, s=5, c="green")
    axes[2].set_title("Source Trasportato T(z)")

    for ax in axes:
        ax.set_grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = "figures/eval_result.png"
    plt.savefig(plot_path)
    print(f"\nGrafico salvato in '{plot_path}'")

if __name__ == "__main__":
    evaluate_model()