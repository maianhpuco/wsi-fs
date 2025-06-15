import os
import sys  
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime 
from utils.file_utils import save_pkl
'''this utils is only for the current testing model, ''' 
from utils.core_utils import train # replace by a 


# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))  #  
_path = os.path.abspath(os.path.join(current_dir,'src')) # 
'''sys.path.append(os.path.join("src/externals/ViLA-MIL")) instead if you want to use their code'''
 
sys.path.append(_path)
'''from utils.core_utils import train --> use this if you want to use ViLa-MIL train function'''  

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_dataset(args):
    if args.dataset_name == 'tcga_renal':
        from src.datasets.multiple_scales.tcga import return_splits_custom

        patch_size = args.patch_size
        mag_s = '5x'
        mag_l = '10x'
        tissue_type = args.tissue_type

        key_s = f"patch_{patch_size}x{patch_size}_{mag_s}"
        key_l = f"patch_{patch_size}x{patch_size}_{mag_l}"

        base_s = args.paths['data_folder_s'][tissue_type]
        base_l = args.paths['data_folder_l'][tissue_type]
        data_dir_s = os.path.join(base_s, key_s)
        data_dir_l = os.path.join(base_l, key_l)

        train_csv_path = os.path.join(args.paths['split_folder'], "train.csv")
        val_csv_path = os.path.join(args.paths['split_folder'], "val.csv")
        test_csv_path = os.path.join(args.paths['split_folder'], "test.csv")

        return return_splits_custom(
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            test_csv_path=test_csv_path,
            data_dir_s=data_dir_s,
            data_dir_l=data_dir_l,
            label_dict=args.label_dict,
            seed=args.seed,
            use_h5=True,
        )
    else:
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' is not supported.")

def main(args):
    args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
    seed_torch(args.seed)

    train_dataset, val_dataset, test_dataset = prepare_dataset(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}_{timestamp}")
    os.makedirs(args.results_dir, exist_ok=True) 


    folds = np.arange(args.k_start if args.k_start != -1 else 0,
                      args.k_end if args.k_end != -1 else args.k)

    all_test_auc, all_val_auc, all_test_acc, all_val_acc, all_test_f1 = [], [], [], [], []

    for i in folds:
        seed_torch(args.seed)
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_f1.append(test_f1)

        save_pkl(os.path.join(args.results_dir, f'split_{i}_results.pkl'), results)

    summary_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'test_acc': all_test_acc,
        'test_f1': all_test_f1
    })
    result_df = pd.DataFrame({
        'metric': ['mean', 'var'],
        'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
        'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)],
        'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)]
    })

    suffix = f"partial_{folds[0]}_{folds[-1]}" if len(folds) != args.k else "full"
    summary_df.to_csv(os.path.join(args.results_dir, f"summary_{suffix}.csv"), index=False)
    result_df.to_csv(os.path.join(args.results_dir, f"result_{suffix}.csv"), index=False)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Load config from YAML and merge into args
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        setattr(args, key, value)

    # Setup and print config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    print("################# SETTINGS ###################")
    for k, v in vars(args).items():
        if k != 'paths':
            print(f"{k}: {v}")
    print(f"[INFO] Max Epochs set to: {args.max_epochs}")
    print("##############################################")

    main(args)
