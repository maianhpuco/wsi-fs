import os
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np

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
        self.data_dir_s = data_dir_s  # dict: {kirc: path, kirp: path, ...}
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
        # Better: use patient_id if available
        subtype = None
        for key in path_dict:
            if key.lower() in slide_id.lower():
                subtype = key
                break
        if subtype is None and 'patient_id' in self.slide_data.columns:
            row = self.slide_data[self.slide_data['slide_id'] == slide_id]
            pid = row['patient_id'].values[0]
            for key in path_dict:
                if key.lower() in pid.lower():
                    subtype = key
                    break
        if subtype is None:
            raise ValueError(f"Cannot match slide_id '{slide_id}' to any subtype in {list(path_dict.keys())}")
        return path_dict[subtype]

    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        slide_id = row['slide_id']
        label_str = row['label']
        label = self.label_dict[label_str]

        folder_s = self._resolve_subtype_path(slide_id, self.data_dir_s)
        folder_l = self._resolve_subtype_path(slide_id, self.data_dir_l)

        if not self.use_h5:
            pt_path_s = os.path.join(folder_s, f"{slide_id}.pt")
            pt_path_l = os.path.join(folder_l, f"{slide_id}.pt")
            features_s = torch.load(pt_path_s, weights_only=True)
            features_l = torch.load(pt_path_l, weights_only=True)
            return features_s, None, features_l, None, label
        else:
            h5_path_s = os.path.join(folder_s, 'h5_files', f"{slide_id}.h5")
            h5_path_l = os.path.join(folder_l, 'h5_files', f"{slide_id}.h5")

            try:
                with h5py.File(h5_path_s, 'r') as f_s:
                    features_s = torch.from_numpy(f_s['features'][:])
                    coords_s = torch.from_numpy(f_s['coords'][:])
            except Exception:
                pt_path_s = os.path.join(folder_s, f"{slide_id}.pt")
                features_s = torch.load(pt_path_s, weights_only=True)
                coords_s = None

            try:
                with h5py.File(h5_path_l, 'r') as f_l:
                    features_l = torch.from_numpy(f_l['features'][:])
                    coords_l = torch.from_numpy(f_l['coords'][:])
            except Exception:
                pt_path_l = os.path.join(folder_l, f"{slide_id}.pt")
                features_l = torch.load(pt_path_l, weights_only=True)
                coords_l = None

            return features_s, coords_s, features_l, coords_l, label

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
