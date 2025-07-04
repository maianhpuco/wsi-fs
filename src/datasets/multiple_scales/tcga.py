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
        # print("-------")
        # print(self.slide_data)
        if print_info:
            print(f"Loaded {len(self.slide_data)} slides.")

    def __len__(self):
        return len(self.slide_data)

    def _resolve_subtype_path(self, slide_id, path_dict):
        for key in path_dict:
            if key.lower() in slide_id.lower():
                return path_dict[key]
        raise ValueError(f"Cannot match slide_id '{slide_id}' to any subtype in {list(path_dict.keys())}")
    
    def __getitem__(self, idx):
        row = self.slide_data.iloc[idx]
        slide_id = row['slide_id']
        label_str = row['label']
        label = self.label_dict[label_str]

        folder_s = self.data_dir_s[label_str.lower()]
        folder_l = self.data_dir_l[label_str.lower()] 
        
        h5_path_s = os.path.join(folder_s, f"{slide_id}.h5")
        h5_path_l = os.path.join(folder_l, f"{slide_id}.h5")
        # print("h5 file large and small")
        # print(h5_path_l)
        # print(h5_path_s)
        with h5py.File(h5_path_s, 'r') as f_s:
            features_s = torch.from_numpy(f_s['features'][:])
            coords_s = torch.from_numpy(f_s['coords'][:])

        with h5py.File(h5_path_l, 'r') as f_l:
            features_l = torch.from_numpy(f_l['features'][:])
            coords_l = torch.from_numpy(f_l['coords'][:])
        # print(features_s, coords_s, features_l, coords_l, label)
        
        return features_s, coords_s, features_l, coords_l, label

import os
import pandas as pd
from collections import defaultdict

# from datasets.dataset_generic import Generic_MIL_Dataset

def return_splits_custom(
    train_csv_path,
    val_csv_path,
    test_csv_path,
    data_dir_s,
    data_dir_l,
    label_dict,
    seed=1,
    print_info=False,
    use_h5=True,
    mode="transformer",
    args=None,
):
    def filter_df(df, name):
        print("------------------")
        kept, missing = [], []
        kept_per_label = defaultdict(list)
        missing_s, missing_l = defaultdict(list), defaultdict(list)

        df = df.drop_duplicates(subset=["slide"])
        for _, row in df.iterrows():
            slide_id = row["slide"]
            label = row["label"].lower()

            try:
                path_s = os.path.join(data_dir_s[label], f"{slide_id}.h5")
                path_l = os.path.join(data_dir_l[label], f"{slide_id}.h5")

                exists_s = os.path.exists(path_s)
                exists_l = os.path.exists(path_l)

                if exists_s and exists_l:
                    kept.append(row)
                    kept_per_label[label].append(slide_id)
                else:
                    missing.append(row)
                    if not exists_s:
                        missing_s[label].append(slide_id)
                    if not exists_l:
                        missing_l[label].append(slide_id)

            except Exception as e:
                print(f"[WARN] {slide_id} → error: {e}")
                missing.append(row)
                missing_s[label].append(slide_id)
                missing_l[label].append(slide_id)

        df_kept = pd.DataFrame(kept).drop_duplicates(subset=["slide"])

        os.makedirs("logs", exist_ok=True)
        if missing_s:
            pd.DataFrame([(k, v) for k, lst in missing_s.items() for v in lst],
                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}_s.csv", index=False)
            print(f"[INFO] Saved data_dir_s missing slides → logs/missing_slides_{name}_s.csv")

        if missing_l:
            pd.DataFrame([(k, v) for k, lst in missing_l.items() for v in lst],
                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}_l.csv", index=False)
            print(f"[INFO] Saved data_dir_l missing slides → logs/missing_slides_{name}_l.csv")

        print(f"[INFO] {name.upper()}: Kept {len(df_kept)} / {len(df)}")

        if args is not None and getattr(args, "dataset_name", "") == "tcga_renal":
            labels = ['kich', 'kirc', 'kirp']
        elif args is not None and getattr(args, "dataset_name", "") == "tcga_lung":
            labels = ['luad', 'lusc']
        else:
            labels = df["label"].dropna().unique().tolist()

        for label in labels:
            count_s = len(missing_s[label])
            count_l = len(missing_l[label])
            available = len(kept_per_label[label])
            print(f"[SUMMARY - {name.upper()} | {label.upper()}] -- AVAILABLE: {available}  MISSING_s: {count_s}  MISSING_l: {count_l}")

        return df_kept

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

    df_train = filter_df(pd.read_csv(train_csv_path), "train")
    df_val   = filter_df(pd.read_csv(val_csv_path), "val")
    df_test  = filter_df(pd.read_csv(test_csv_path), "test")

    return create_dataset(df_train), create_dataset(df_val), create_dataset(df_test)
