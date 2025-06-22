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


                


# def return_splits_custom(
#     train_csv_path,
#     val_csv_path,
#     test_csv_path,
#     data_dir_s,
#     data_dir_l,
#     label_dict,
#     seed=1,
#     print_info=False,
#     use_h5=False,
#     mode="transformer"
# ):
#     def create_dataset(df):
#         return Generic_MIL_Dataset(
#             data_dir_s=data_dir_s,
#             data_dir_l=data_dir_l,
#             patient_ids=df["patient_id"].dropna().tolist(),
#             slides=df["slide"].dropna().tolist(),
#             labels=df["label"].dropna().tolist(),
#             label_dict=label_dict,
#             seed=seed,
#             print_info=print_info,
#             use_h5=use_h5,
#             mode=mode
#         )

#     df_train = pd.read_csv(train_csv_path)
#     df_val = pd.read_csv(val_csv_path)
#     df_test = pd.read_csv(test_csv_path)

#     return create_dataset(df_train), create_dataset(df_val), create_dataset(df_test)

# import os
# import pandas as pd
# # from datasets.dataset_generic import Generic_MIL_Dataset
# from collections import defaultdict
 
# def return_splits_custom(
#     train_csv_path,
#     val_csv_path,
#     test_csv_path,
#     data_dir_s,
#     data_dir_l,
#     label_dict,
#     seed=1,
#     print_info=False,
#     use_h5=True,
#     mode="transformer"
# ):
#     from collections import defaultdict

#     def filter_df(df, name):
#         kept_s, kept_l = [], []
#         missing_s, missing_l = defaultdict(list), defaultdict(list)

#         print(f"[INFO] Filtering {name} dataset...")
#         print(f"[INFO] Total slides in {name}: {len(df)}")

#         # Count unique slide IDs per label
#         unique_slide_counts = df.groupby("label")["slide"].nunique()
#         print(f"[INFO] Unique slide IDs per label in {name}:\n{unique_slide_counts}")

#         for _, row in df.iterrows():
#             slide_id = row["slide"]
#             label = row["label"].lower()
#             try:
#                 path_s = os.path.join(data_dir_s[label], f"{slide_id}.h5")
#                 path_l = os.path.join(data_dir_l[label], f"{slide_id}.h5")

#                 exists_s = os.path.exists(path_s)
#                 exists_l = os.path.exists(path_l)

#                 if exists_s:
#                     kept_s.append(row)
#                 else:
#                     missing_s[label].append(slide_id)

#                 if exists_l:
#                     kept_l.append(row)
#                 else:
#                     missing_l[label].append(slide_id)

#             except Exception as e:
#                 print(f"[WARN] {slide_id} → error: {e}")
#                 missing_s[label].append(slide_id)
#                 missing_l[label].append(slide_id)

#         # Save missing logs
#         os.makedirs("logs", exist_ok=True)
#         if missing_s:
#             pd.DataFrame([(k, v) for k, lst in missing_s.items() for v in lst],
#                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}_s.csv", index=False)
#             print(f"[INFO] Saved data_dir_s missing slides → logs/missing_slides_{name}_s.csv")

#         if missing_l:
#             pd.DataFrame([(k, v) for k, lst in missing_l.items() for v in lst],
#                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}_l.csv", index=False)
#             print(f"[INFO] Saved data_dir_l missing slides → logs/missing_slides_{name}_l.csv")

#         print(f"[INFO] {name.upper()}:")
#         print(f"  data_dir_s: Kept {len(kept_s)} / {len(df)}")
#         print(f"  data_dir_l: Kept {len(kept_l)} / {len(df)}")

#         # === Summary of missing files ===
#         for label in ['kich', 'kirc', 'kirp']:
#             count_s = len(missing_s[label])
#             count_l = len(missing_l[label])
#             print(f"\n[SUMMARY - {name.upper()} | {label.upper()}]")
#             print(f"  data_dir_s: {count_s} missing file(s)")
#             print(f"  data_dir_l: {count_l} missing file(s)")

#         df_kept_s = pd.DataFrame(kept_s).drop_duplicates(subset=["slide"])
#         df_kept_l = pd.DataFrame(kept_l).drop_duplicates(subset=["slide"])
#         return df_kept_s, df_kept_l
  
#     def create_dataset(df):
#         return Generic_MIL_Dataset(
#             data_dir_s=data_dir_s,
#             data_dir_l=data_dir_l,
#             patient_ids=df["patient_id"].dropna().tolist(),
#             slides=df["slide"].dropna().tolist(),
#             labels=df["label"].dropna().tolist(),
#             label_dict=label_dict,
#             seed=seed,
#             print_info=print_info,
#             use_h5=use_h5,
#             mode=mode
#         )

#     # Load and filter
#     df_train = filter_df(pd.read_csv(train_csv_path), "train")
#     df_val   = filter_df(pd.read_csv(val_csv_path), "val")
#     df_test  = filter_df(pd.read_csv(test_csv_path), "test")

#     return create_dataset(df_train), create_dataset(df_val), create_dataset(df_test)
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
    mode="transformer"
):
    def filter_df(df, name):
        kept, missing = [], []
        missing_s, missing_l = defaultdict(list), defaultdict(list)

        # print(f"[INFO] Filtering {name} dataset...")
        # print(f"[INFO] Total slides in {name}: {len(df)}")

        # unique_slide_counts = df.groupby("label")["slide"].nunique()
        # print(f"[INFO] Unique slide IDs per label in {name}:\n{unique_slide_counts}")

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

        # Save separate logs
        os.makedirs("logs", exist_ok=True)
        if missing_s:
            pd.DataFrame([(k, v) for k, lst in missing_s.items() for v in lst],
                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}_s.csv", index=False)
            # print(f"[INFO] Saved data_dir_s missing slides → logs/missing_slides_{name}_s.csv")

        if missing_l:
            pd.DataFrame([(k, v) for k, lst in missing_l.items() for v in lst],
                         columns=["label", "slide"]).to_csv(f"logs/missing_slides_{name}_l.csv", index=False)
            # print(f"[INFO] Saved data_dir_l missing slides → logs/missing_slides_{name}_l.csv")

        print(f"[INFO] {name.upper()}: Kept {len(df_kept)} / {len(df)}")

        for label in ['kich', 'kirc', 'kirp']:
            count_s = len(missing_s[label])
            count_l = len(missing_l[label])
            print(f"\n[SUMMARY - {name.upper()} | {label.upper()}]")
            print(f"  data_dir_s: {count_s} missing file(s)","  data_dir_l: {count_l} missing file(s)")

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
