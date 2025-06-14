import os
import pandas as pd

# Define paths
data_dir_map = {
    'kich': "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/features_fp",
    'kirp': "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp",
    'kirc': "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp"
}

simea_root = "/home/mvu9/processing_datasets/processing_tcga_256"
transfer_script_path = "transfer_missing_files.sh"

train_csv_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/splits_csv_tcga_renal/fold_1/train.csv"
val_csv_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/splits_csv_tcga_renal/fold_1/val.csv"
test_csv_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/splits_csv_tcga_renal/fold_1/test.csv"

# Step 1: Check for missing files
def check_missing_files(csv_path, data_dir_map):
    df = pd.read_csv(csv_path)
    missing_files = []

    for _, row in df.iterrows():
        slide_id = row['slide']
        label = row['label'].lower()
        pt_path = os.path.join(data_dir_map[label], 'pt_files', f"{slide_id}.pt")
        h5_path = os.path.join(data_dir_map[label], 'h5_files', f"{slide_id}.h5")
        pt_missing = not os.path.exists(pt_path)
        h5_missing = not os.path.exists(h5_path)
        if pt_missing or h5_missing:
            missing_files.append((label, slide_id, pt_missing, h5_missing))
    
    return missing_files

# Collect from all splits
missing_all = (
    check_missing_files(train_csv_path, data_dir_map) +
    check_missing_files(val_csv_path, data_dir_map) +
    check_missing_files(test_csv_path, data_dir_map)
)

df_missing = pd.DataFrame(missing_all, columns=["label", "slide_id", "missing_pt", "missing_h5"])

# Step 2: Write SCP commands
scp_commands = []

for _, row in df_missing.iterrows():
    label = row["label"]
    slide_id = row["slide_id"]

    if row["missing_pt"]:
        remote_pt = f"mvu9@simea.ee.e.uh.edu:{simea_root}/{label}/features_fp/pt_files/{slide_id}.pt"
        local_pt = f"{data_dir_map[label]}/pt_files/"
        scp_commands.append(f"scp -r {remote_pt} {local_pt}")

    if row["missing_h5"]:
        remote_h5 = f"mvu9@simea.ee.e.uh.edu:{simea_root}/{label}/features_fp/h5_files/{slide_id}.h5"
        local_h5 = f"{data_dir_map[label]}/h5_files/"
        scp_commands.append(f"scp -r {remote_h5} {local_h5}")

# Step 3: Save to shell script
with open(transfer_script_path, 'w') as f:
    f.write("#!/bin/bash\n\n")
    for cmd in scp_commands:
        f.write(cmd + '\n')

print(f"✅ Found {len(df_missing)} missing slides")
print(f"✅ Saved {len(scp_commands)} SCP commands to: {transfer_script_path}")
