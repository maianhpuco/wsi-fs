import os
import pandas as pd

# Define parameters
data_dir_map = {
    'kich': "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/features_fp",
    'kirp': "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp",
    'kirc': "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp"
}

train_csv_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/splits_csv_tcga_renal/train.csv"
val_csv_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/splits_csv_tcga_renal/val.csv"
test_csv_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/splits_csv_tcga_renal/test.csv"

def check_missing_files(csv_path, data_dir_map):
    df = pd.read_csv(csv_path)
    missing_files = []

    for _, row in df.iterrows():
        slide_id = row['slide']
        label = row['label'].lower()
        pt_path = os.path.join(data_dir_map[label], 'pt_files', f"{slide_id}.pt")
        h5_path = os.path.join(data_dir_map[label], 'h5_files', f"{slide_id}.h5")
        if not os.path.exists(pt_path) or not os.path.exists(h5_path):
            missing_files.append((label, slide_id, not os.path.exists(pt_path), not os.path.exists(h5_path)))
    
    return missing_files

# Check all splits
missing_train = check_missing_files(train_csv_path, data_dir_map)
missing_val = check_missing_files(val_csv_path, data_dir_map)
missing_test = check_missing_files(test_csv_path, data_dir_map)

missing_all = missing_train + missing_val + missing_test
import pandas as pd
df_missing = pd.DataFrame(missing_all, columns=["label", "slide_id", "missing_pt", "missing_h5"])

