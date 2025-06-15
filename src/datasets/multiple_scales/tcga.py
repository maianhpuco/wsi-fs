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
                 use_h5=False,
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
            print("Loaded {} samples (low: {}, high: {})".format(
                len(self.slide_data), data_dir_s, data_dir_l))

    def load_from_h5(self, toggle: bool):
        self.use_h5 = toggle

    def __len__(self):
        return len(self.slide_data)
    def __getitem__(self, idx):
        slide_id = self.slide_data.iloc[idx]['slide_id']
        label_str = self.slide_data.iloc[idx]['label']
        label = self.label_dict[label_str]

        if not self.use_h5:
            # Use .pt files
            pt_path_s = os.path.join(self.data_dir_s, f"{slide_id}.pt")
            pt_path_l = os.path.join(self.data_dir_l, f"{slide_id}.pt")
            features_s = torch.load(pt_path_s, weights_only=True)
            features_l = torch.load(pt_path_l, weights_only=True)
            return features_s, None, features_l, None, label

        else:
            # Try to use .h5 files, but if they don't exist, fall back to returning None for coords
            h5_path_s = os.path.join(self.data_dir_s, 'h5_files', f"{slide_id}.h5")
            h5_path_l = os.path.join(self.data_dir_l, 'h5_files', f"{slide_id}.h5")

            try:
                with h5py.File(h5_path_s, 'r') as f_s:
                    features_s = torch.from_numpy(f_s['features'][:])
                    coords_s = torch.from_numpy(f_s['coords'][:])
            except FileNotFoundError:
                pt_path_s = os.path.join(self.data_dir_s, 'pt_files', f"{slide_id}.pt")
                features_s = torch.load(pt_path_s, weights_only=True)
                coords_s = None

            try:
                with h5py.File(h5_path_l, 'r') as f_l:
                    features_l = torch.from_numpy(f_l['features'][:])
                    coords_l = torch.from_numpy(f_l['coords'][:])
            except FileNotFoundError:
                pt_path_l = os.path.join(self.data_dir_l, 'pt_files', f"{slide_id}.pt")
                features_l = torch.load(pt_path_l, weights_only=True)
                coords_l = None

            return features_s, coords_s, features_l, coords_l, label
 
    # def __getitem__(self, idx):
    #     slide_id = self.slide_data['slide_id'].iloc[idx]
    #     label_str = self.slide_data['label'].iloc[idx]
    #     label = self.label_dict[label_str]

    #     if not self.use_h5:
    #         # Use .pt files if specified
    #         pt_path_s = os.path.join(self.data_dir_s, 'pt_files', f"{slide_id}.pt")
    #         pt_path_l = os.path.join(self.data_dir_l, 'pt_files', f"{slide_id}.pt")

    #         features_s = torch.load(pt_path_s, weights_only=True)
    #         features_l = torch.load(pt_path_l, weights_only=True)
    #         return features_s, None, features_l, None, label

    #     else:
    #         # Use .h5 files
    #         h5_path_s = os.path.join(self.data_dir_s, 'h5_files', f"{slide_id}.h5")
    #         h5_path_l = os.path.join(self.data_dir_l, 'h5_files', f"{slide_id}.h5")

    #         with h5py.File(h5_path_s, 'r') as f_s:
    #             features_s = torch.from_numpy(f_s['features'][:])
    #             coords_s = torch.from_numpy(f_s['coords'][:])

    #         with h5py.File(h5_path_l, 'r') as f_l:
    #             features_l = torch.from_numpy(f_l['features'][:])
    #             coords_l = torch.from_numpy(f_l['coords'][:])

    #         return features_s, coords_s, features_l, coords_l, label

    def get_features_by_slide_id(self, slide_id):
        row = self.slide_data[self.slide_data['slide_id'] == slide_id]
        if row.empty:
            raise ValueError(f"Slide ID {slide_id} not found in dataset.")
        label = row.iloc[0]['label']
        label_idx = self.label_dict[label]

        h5_path_s = os.path.join(self.data_dir_s, 'h5_files', f"{slide_id}.h5")
        h5_path_l = os.path.join(self.data_dir_l, 'h5_files', f"{slide_id}.h5")

        with h5py.File(h5_path_s, 'r') as f_s:
            features_s = torch.from_numpy(f_s['features'][:])
        with h5py.File(h5_path_l, 'r') as f_l:
            features_l = torch.from_numpy(f_l['features'][:])

        return features_s, features_l

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
    """Return train, val, test datasets with multi-level feature support."""

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

    train_dataset = create_dataset(df_train)
    val_dataset = create_dataset(df_val)
    test_dataset = create_dataset(df_test)

    return train_dataset, val_dataset, test_dataset
