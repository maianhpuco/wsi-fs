import os
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np


class Generic_MIL_Dataset(Dataset):
    def __init__(self,
                 data_dir_s,
                 data_dir_l,
                 patient_ids,
                 slides,
                 labels,
                 label_dict,
                 seed=1,
                 print_info=False,
                 use_h5=True,
                 ignore=[],
                 **kwargs):
        self.data_dir_s = data_dir_s
        self.data_dir_l = data_dir_l
        self.label_dict = label_dict
        self.ignore = ignore
        self.seed = seed
        self.use_h5 = use_h5
        self.kwargs = kwargs

        self.slide_data = pd.DataFrame({
            'patient_id': patient_ids,
            'slide_id': slides,
            'label': labels
        })

        if print_info:
            print(f"Loaded {len(self.slide_data)} slides.")

    def __len__(self):
        return len(self.slide_data)

    def _resolve_subtype_path(self, slide_id, path_dict):
        for key in path_dict:
            if slide_id.startswith("TCGA") and key.lower() in slide_id.lower():
                return path_dict[key]
        raise ValueError(f"Cannot match slide_id '{slide_id}' to any subtype in {list(path_dict.keys())}")

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        slide_id = row['slide_id']
        label_str = row['label']
        label = self.label_dict[label_str]

        # Try each path and skip if file not found
        folder_s, folder_l = None, None
        for subtype in self.data_dir_s:
            if subtype.lower() in slide_id.lower():
                folder_s = self.data_dir_s[subtype]
                folder_l = self.data_dir_l[subtype]
                break
        if folder_s is None or folder_l is None:
            raise ValueError(f"Slide ID '{slide_id}' does not match any subtype in provided paths.")

        if self.use_h5:
            h5_path_s = os.path.join(folder_s, f"{slide_id}.h5")
            h5_path_l = os.path.join(folder_l, f"{slide_id}.h5")

            try:
                with h5py.File(h5_path_s, 'r') as f_s:
                    features_s = torch.from_numpy(f_s['features'][:])
                    coords_s = torch.from_numpy(f_s['coords'][:])
            except Exception:
                features_s, coords_s = None, None

            try:
                with h5py.File(h5_path_l, 'r') as f_l:
                    features_l = torch.from_numpy(f_l['features'][:])
                    coords_l = torch.from_numpy(f_l['coords'][:])
            except Exception:
                features_l, coords_l = None, None

            return features_s, coords_s, features_l, coords_l, label

        else:
            pt_path_s = os.path.join(folder_s, f"{slide_id}.pt")
            pt_path_l = os.path.join(folder_l, f"{slide_id}.pt")
            features_s = torch.load(pt_path_s, map_location='cpu')
            features_l = torch.load(pt_path_l, map_location='cpu')
            return features_s, None, features_l, None, label


def return_splits_custom(
    train_csv_path,
    val_csv_path,
    test_csv_path,
    data_dir_s,
    data_dir_l,
    label_dict,
    seed=1,
    print_info=False,
    use_h5=False,
    mode="transformer"
):
    def create_dataset(df):
        return Generic_MIL_Dataset(
            data_dir_s=data_dir_s,
            data_dir_l=data_dir_l,
            patient_ids=df["patient_id"].dropna().tolist(),
            slides=df["slide"].dropna().tolist(),
            labels=df["label"].dropna().tolist(),
            label_dict=label_dict,
            seed=seed,
            print_info=print_info,
            use_h5=use_h5,
            mode=mode
        )

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    return create_dataset(df_train), create_dataset(df_val), create_dataset(df_test)
