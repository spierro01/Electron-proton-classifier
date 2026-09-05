import os
import inspect
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import pid_data
from pid_model import ParticleTransformer


def extract_z_enc(model, x_tensor, device, batch_size=256):
    """Estrae i vettori latenti z_enc facendoli passare nel modello."""
    model.eval()
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latents_list = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z_out = model.latents(batch_x)
            
            # Gestione output se z_out è un dizionario
            if isinstance(z_out, dict):
                # Cerca la chiave del vettore latente o prende il primo valore
                if "z_enc" in z_out:
                    z_tensor = z_out["z_enc"]
                elif "latents" in z_out:
                    z_tensor = z_out["latents"]
                else:
                    z_tensor = list(z_out.values())[0]
            else:
                z_tensor = z_out

            latents_list.append(z_tensor.cpu().numpy())

    return np.concatenate(latents_list, axis=0)


def preprocess_file(parquet_path, mean, std):
    """Legge il parquet, costruisce la matrice 3D e applica la normalizzazione."""
    print(f"Lettura e processamento file: {parquet_path}")
    cols = pid_data.required_columns()

    try:
        df = pd.read_parquet(parquet_path, columns=cols)
    except KeyError:
        feat_cols = [c for c in cols if c != "particle_type"]
        df = pd.read_parquet(parquet_path, columns=feat_cols)

    x = pid_data.build_matrix(df)

    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        std_arr = np.where(std_arr == 0, 1.0, std_arr)
        x = (x - mean_arr) / std_arr

    return torch.from_numpy(x).float()


def find_file(possible_paths):
    """Restituisce il primo percorso valido tra quelli forniti."""
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/pid_transformer/best.pt"

    print(f"Caricamento checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    raw_args = ckpt["args"]
    args_dict = raw_args if isinstance(raw_args, dict) else vars(raw_args)

    sig = inspect.signature(ParticleTransformer.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in args_dict.items() if k in valid_params}

    if "num_classes" not in model_kwargs or model_kwargs["num_classes"] is None:
        model_kwargs["num_classes"] = len(ckpt["classes"]) if "classes" in ckpt else 3

    model = ParticleTransformer(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])

    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)

    os.makedirs("../derived", exist_ok=True)

    # 1. Estrazione latenti Monte Carlo (Simulazione)
    mc_paths = [
        "../dumpMC_spectra.parquet",
        "dumpMC_spectra.parquet",
        "../optimal_transport_toy/dumpMC_spectra.parquet",
        "../optimal_transport_toy/data/dumpMC_spectra.parquet",
    ]
    mc_path = find_file(mc_paths)

    if mc_path:
        print("\n--- Processamento Monte Carlo ---")
        x_mc = preprocess_file(mc_path, mean, std)
        z_sim = extract_z_enc(model, x_mc, device)
        np.save("../derived/z_sim.npy", z_sim)
        print(f"-> Salvato ../derived/z_sim.npy | Shape: {z_sim.shape}")
    else:
        print("\n[ATTENZIONE] File Monte Carlo non trovato nei percorsi standard.")

    # 2. Estrazione latenti Test Beam (Dati Reali)
    tb_paths = [
        "../dumpTB.parquet",
        "dumpTB.parquet",
        "../optimal_transport_toy/dumpTB.parquet",
        "../optimal_transport_toy/data/dumpTB.parquet",
    ]
    tb_path = find_file(tb_paths)

    if tb_path:
        print("\n--- Processamento Test Beam ---")
        x_tb = preprocess_file(tb_path, mean, std)
        z_real = extract_z_enc(model, x_tb, device)
        np.save("../derived/z_real.npy", z_real)
        print(f"-> Salvato ../derived/z_real.npy | Shape: {z_real.shape}")
    else:
        print("\n[ATTENZIONE] File Test Beam non trovato nei percorsi standard.")


if __name__ == "__main__":
    main()
