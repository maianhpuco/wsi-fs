import os
import pandas as pd
import random

# === CONFIGURATION ===
feature_dir = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/features_fp/pt_files"
output_csv = "kich_split.csv"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
seed = 42

# === LOAD FILENAMES ===
all_files = [f for f in os.listdir(feature_dir) if f.endswith(".pt")]
ids = [os.path.splitext(f)[0] for f in all_files]  # remove .pt extension

# === SHUFFLE AND SPLIT ===
random.seed(seed)
random.shuffle(ids)

n = len(ids)
n_train = int(n * train_ratio)
n_val = int(n * val_ratio)
n_test = n - n_train - n_val

train_ids = ids[:n_train]
val_ids = ids[n_train:n_train + n_val]
test_ids = ids[n_train + n_val:]

# === PAD TO EQUAL LENGTHS FOR CSV FORMAT ===
max_len = max(len(train_ids), len(val_ids), len(test_ids))
train_ids += [""] * (max_len - len(train_ids))
val_ids += [""] * (max_len - len(val_ids))
test_ids += [""] * (max_len - len(test_ids))

# === WRITE CSV ===
df = pd.DataFrame({
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
})
df.to_csv(output_csv, index=False)
print(f"Saved split to {output_csv}")
