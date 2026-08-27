import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Standardizer:
    """Standardizza i dati usando media e deviazione standard del training set."""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # Evita divisioni per zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_and_preprocess_data(data_path="data/toy_ot_data.npz", batch_size=256):
    data = np.load(data_path)
    
    source_train = data["source_train"]
    target_train = data["target_train"]
    source_test = data["source_test"]
    target_test = data["target_test"]

    # Inizializza e calcola statistiche solo su TRAIN
    source_stdizer = Standardizer()
    target_stdizer = Standardizer()

    source_stdizer.fit(source_train)
    target_stdizer.fit(target_train)

    # Standardizza tutti i set
    source_train_scaled = source_stdizer.transform(source_train)
    target_train_scaled = target_stdizer.transform(target_train)
    source_test_scaled = source_stdizer.transform(source_test)
    target_test_scaled = target_stdizer.transform(target_test)

    # Converti in PyTorch Tensors
    source_train_tensor = torch.tensor(source_train_scaled, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train_scaled, dtype=torch.float32)

    # Dataloader per il training batching
    source_loader = DataLoader(source_train_tensor, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_train_tensor, batch_size=batch_size, shuffle=True)

    stats = {
        "source_stdizer": source_stdizer,
        "target_stdizer": target_stdizer,
        "source_test": source_test_scaled,
        "target_test": target_test_scaled,
        "source_test_raw": source_test,
        "target_test_raw": target_test,
    }

    return source_loader, target_loader, stats