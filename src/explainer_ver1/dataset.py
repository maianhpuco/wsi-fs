import os
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, pt_dirs, label_map):
        self.files = []
        self.labels = []
        for label_name, path in pt_dirs.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory {path} does not exist")
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]
            if not files:
                raise ValueError(f"No .pt files found in {path}")
            self.files.extend(files)
            self.labels.extend([label_map[label_name]] * len(files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx])
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.files[idx]}: {str(e)}")
        features = data['features']  # Tensor [N, D]
        if features.ndim != 2:
            raise ValueError(f"Expected features to have shape [N, D], got {features.shape} in {self.files[idx]}")
        coords = data.get('coords', None)
        meta = {'coords': coords, 'filename': os.path.basename(self.files[idx])}
        return features, self.labels[idx], meta