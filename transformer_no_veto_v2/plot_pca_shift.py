import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def main():
    # 1. Caricamento dei vettori latenti
    z_sim_path = "../derived/z_sim.npy"
    z_real_path = "../derived/z_real.npy"

    print("Caricamento vettori latenti...")
    z_sim = np.load(z_sim_path)
    z_real = np.load(z_real_path)

    # 2. Subsampling per visualizzazione rapida (20.000 campioni ciascuno)
    n_samples = 20000
    np.random.seed(42)

    idx_sim = np.random.choice(len(z_sim), size=n_samples, replace=False)
    idx_real = np.random.choice(len(z_real), size=n_samples, replace=False)

    z_sim_sub = z_sim[idx_sim]
    z_real_sub = z_real[idx_real]

    # 3. Fit della PCA sullo spazio combinato
    print("Calcolo PCA 2D...")
    combined_data = np.vstack([z_sim_sub, z_real_sub])
    pca = PCA(n_components=2)
    pca.fit(combined_data)

    z_sim_pca = pca.transform(z_sim_sub)
    z_real_pca = pca.transform(z_real_sub)

    # 4. Generazione e visualizzazione del grafico
    os.makedirs("../derived", exist_ok=True)
    plt.figure(figsize=(9, 7))

    plt.scatter(
        z_sim_pca[:, 0],
        z_sim_pca[:, 1],
        alpha=0.3,
        s=4,
        c="tab:blue",
        label="Monte Carlo (Sim)",
    )
    plt.scatter(
        z_real_pca[:, 0],
        z_real_pca[:, 1],
        alpha=0.3,
        s=4,
        c="tab:orange",
        label="Test Beam (Real)",
    )

    var_expl = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PCA Component 1 ({var_expl[0]:.1f}% var)")
    plt.ylabel(f"PCA Component 2 ({var_expl[1]:.1f}% var)")
    plt.title("Domain Shift Visualization: Latent Space (Z_enc)")
    plt.legend(loc="upper right", markerscale=3)
    plt.grid(True, linestyle="--", alpha=0.5)

    output_fig = "../derived/pca_domain_shift.png"
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"Grafico salvato in: {output_fig}")

    # Mostra la finestra interattiva prima di chiudere
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()