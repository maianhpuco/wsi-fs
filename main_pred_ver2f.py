import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from utils.file_utils import save_pkl
from utils.core_utils import train  # Make sure this expects model as the first arg

sys.path.append(os.path.join("src"))  
sys.path.append("src")
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'
from explainer_ver2 import Ver2f
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

def main(args):
    import json

    with open(args.text_prompts_path, "r") as f:
        args.text_prompts = json.load(f)
    print(args.text_prompts)    
    
    seed_torch(args.seed)

    all_test_auc, all_val_auc, all_test_acc, all_val_acc, all_test_f1, folds = [], [], [], [], [], []

    for i in range(args.k_start, args.k_end + 1):
        datasets = prepare_dataset(args, i)

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = 'final'
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

        model = Ver2f(config=config, num_classes=args.n_classes).cuda()

        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(model, datasets, cur=i, args=args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_f1.append(test_f1)

        save_pkl(os.path.join(args.results_dir, f'split_{i}_results.pkl'), results)

        desc_dir = os.path.join(args.results_dir, "desc") 
        os.makedirs(desc_dir, exist_ok=True)
        # ===== Return descriptions for patches in test set =====
        for idx, batch in enumerate(datasets[2]):  # test_dataset
            model.eval()
            with torch.no_grad():
                x_s, coord_s, x_l, coord_l, label, slide_id = batch['features_s'].cuda(), batch['coords_s'], batch['features_l'].cuda(), batch['coords_l'], batch['label'].cuda(), batch['slide_id']
                x_s_proj = model.visual_proj(F.normalize(x_s.unsqueeze(0), dim=-1))
                x_s_proj = F.normalize(x_s_proj, dim=-1)

                # === Description Summary by Attention Threshold ===
                summary = model.get_patchwise_desc_summary(x_s_proj, threshold=0.5)
                import json

                def convert(o):
                    if isinstance(o, torch.Tensor):
                        return o.tolist()
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    return o

                with open(os.path.join(desc_dir, f"desc_summary_{slide_id}.json"), "w") as f:
                    json.dump(summary, f, indent=2, default=convert) 
                
                # save_pkl(os.path.join(args.results_dir, f"desc_summary_{slide_id}.pkl"), summary)

    # Save summary
    summary_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'test_acc': all_test_acc,
        'test_f1': all_test_f1
    })

    result_df = pd.DataFrame({
        'metric': ['mean', 'std'],
        'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
        'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
        'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)]
    })

    args.k = args.k_end - args.k_start + 1
    suffix = f"partial_{folds[0]}_{folds[-1]}" if len(folds) != args.k else "full"
    summary_df.to_csv(os.path.join(args.results_dir, f"summary_{suffix}.csv"), index=False)
    result_df.to_csv(os.path.join(args.results_dir, f"result_{suffix}.csv"), index=False)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--k_start', type=int, required=True)
    parser.add_argument('--k_end', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=42)
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
