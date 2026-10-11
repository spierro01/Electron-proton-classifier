import numpy as np

z_sim = np.load("../derived/z_sim.npy")
z_real = np.load("../derived/z_real.npy")

print(
    f"Sim - Mean: {z_sim.mean():.4f}, Std: {z_sim.std():.4f}, Shape: {z_sim.shape}"
)
print(
    f"Real - Mean: {z_real.mean():.4f}, Std: {z_real.std():.4f}, Shape: {z_real.shape}"
)