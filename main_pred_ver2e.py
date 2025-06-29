import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F 
from datetime import datetime
from utils.file_utils import save_pkl
from utils.core_utils import train  # Make sure this expects model as the first arg

sys.path.append(os.path.join("src"))  
sys.path.append("src")
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'
from explainer_ver2 import Ver2e
import ml_collections

# === PATH SETUP ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'src')))

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_dataset(args, fold_id):
    if args.dataset_name == 'tcga_renal':
        from src.datasets.multiple_scales.tcga import return_splits_custom
        patch_size = args.patch_size
        data_dir_s_mapping= args.paths['data_folder_s']
        data_dir_l_mapping =args.paths['data_folder_l'] 
        
        train_dataset, val_dataset, test_dataset = return_splits_custom(
            train_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/train.csv"),
            val_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/val.csv"),
            test_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/test.csv"),
            data_dir_s=data_dir_s_mapping,
            data_dir_l=data_dir_l_mapping,
            label_dict=args.label_dict,
            seed=args.seed,
            use_h5=True,
        )
        print(len(train_dataset))
        return train_dataset, val_dataset, test_dataset  
    else:
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' not supported.")

import json 
def validate_and_save_desc(model, loader, desc_dir, args):
    device = args.device
    model.eval()
    os.makedirs(desc_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (data_s, coord_s, data_l, coord_l, label) in enumerate(loader):
            data_s = data_s.to(device)
            coord_s = coord_s.to(device)
            data_l = data_l.to(device)
            coord_l = coord_l.to(device)
            label = label.to(device)

            # Forward pass to get patch features
            B, N, D = data_s.shape
            data_s_proj = F.normalize(model.visual_proj(F.normalize(data_s, dim=-1)), dim=-1)
            patch_summaries = model.get_patchwise_desc_summary(data_s_proj, threshold=0.25)

            slide_id = loader.dataset.slide_data.iloc[batch_idx]["slide_id"]
            out_path = os.path.join(desc_dir, f"desc_summary_{slide_id}.json")
            print(f"[✓] Processing slide {slide_id} ({batch_idx + 1}/{len(loader)})")
            print(f"Patch summary length: {patch_summaries}")
            # Save as JSON (ensure all keys/values are serializable)
            with open(out_path, "w") as f:
                json.dump(patch_summaries[0], f, indent=2)

            print(f"[✓] Saved summary for slide {slide_id} at {out_path}")

def main(args):
    import json

    with open(args.text_prompts_path, "r") as f:
        args.text_prompts = json.load(f)
    print(args.text_prompts)    
    
    seed_torch(args.seed)

    all_test_auc, all_val_auc, all_test_acc, all_val_acc, all_test_f1, folds = [], [], [], [], [], []

    for i in range(args.k_start, args.k_end + 1):
        datasets = prepare_dataset(args, i)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.results_dir = os.path.join(args.paths['results_dir'], f"resuls_fold{i}_timestamp_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True) 
        
        print(f"\n=========== Fold {i} ===========")
        seed_torch(args.seed)
        folds.append(i)

        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompts = args.text_prompts
        config.device = args.device
        config.prototype_number = args.prototype_number
        config.weight_path = "hf_hub:MahmoodLab/conch"

        model = Ver2e(config=config, num_classes=args.n_classes).cuda()
        cur = 1 
        # ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
        ckpt_path = args.ckpt_path 
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device)) 
      

        desc_dir = os.path.join(args.results_dir, "desc") 
        os.makedirs(desc_dir, exist_ok=True)
        test_dataset = datasets[2]  # test_dataset 
        # ===== Return descriptions for patches in test set ===== 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2) 
        validate_and_save_desc(model, test_loader , desc_dir, args)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    print("################# SETTINGS ###################")
    for k, v in vars(args).items():
        if k != 'paths':
            print(f"{k}: {v}")
    print("##############################################")

    main(args)
