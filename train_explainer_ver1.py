import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(args):
    
    if args.dataset_name == 'tcga_renal':
        from src.datasets.multiple_scales.tcga import return_splits_custom  
        train_dataset, val_dataset, test_dataset = return_splits_custom(
            train_csv_path="splits/train.csv",
            val_csv_path="splits/val.csv",
            test_csv_path="splits/test.csv",
            data_dir_s="/data/10x",
            data_dir_l="/data/20x",
            label_dict=args.label_dict, 
            seed=42,
            use_h5=True,
        )
    
    
    
    
    
def main(config_path):
    args = load_config(config_path)
    args['text_prompt'] = np.array(pd.read_csv(args['text_prompt_path'], header=None)).squeeze()
    seed_torch(args['seed'])

    dataset = prepare_dataset(args)

    args['results_dir'] = os.path.join(args['results_dir'], f"{args['exp_code']}_s{args['seed']}")
    os.makedirs(args['results_dir'], exist_ok=True)

    if args['split_dir'] is None:
        args['split_dir'] = os.path.join('splits', f"{args['task']}_{int(args['label_frac'] * 100)}")
    else:
        args['split_dir'] = os.path.join('splits', args['split_dir'])

    assert os.path.isdir(args['split_dir'])

    with open(os.path.join(args['results_dir'], f"experiment_{args['exp_code']}.txt"), 'w') as f:
        print(args, file=f)

    print("########## CONFIG ##########")
    for k, v in args.items():
        print(f"{k}: {v}")

    start = 0 if args['k_start'] == -1 else args['k_start']
    end = args['k'] if args['k_end'] == -1 else args['k_end']
    folds = np.arange(start, end)

    all_test_auc, all_val_auc, all_test_acc, all_val_acc, all_test_f1 = [], [], [], [], []

    for i in folds:
        seed_torch(args['seed'])
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path=os.path.join(args['split_dir'], f'splits_{i}.csv')
        )
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, _, test_f1 = train(datasets, i, argparse.Namespace(**args))

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_f1.append(test_f1)

        save_pkl(os.path.join(args['results_dir'], f'split_{i}_results.pkl'), results)

    # Save results summary
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

    suffix = f"partial_{folds[0]}_{folds[-1]}" if len(folds) != args['k'] else ""
    summary_df.to_csv(os.path.join(args['results_dir'], f"summary_{suffix or 'full'}.csv"), index=False)
    result_df.to_csv(os.path.join(args['results_dir'], f"result_{suffix or 'full'}.csv"), index=False)

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge YAML config into args
    for key, value in config.items():
        setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    print("################# Settings ###################")
    settings = vars(args).copy()
    settings.pop('paths', None)
    for key, val in settings.items():
        print(f"{key}: {val}")
    print(f"[INFO] Max Epochs set to: {args.max_epochs}")
    main(args) 