import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from utils.file_utils import save_pkl
import ml_collections
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Set up path
sys.path.append(os.path.join("src"))   

os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'

from explainer_ver1 import CONCH_ZeroShot_Model

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
        train_dataset, val_dataset, test_dataset = return_splits_custom(
            train_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/train.csv"),
            val_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/val.csv"),
            test_csv_path=os.path.join(args.paths['split_folder'], f"fold_{fold_id}/test.csv"),
            data_dir_s=args.paths['data_folder_s'],
            data_dir_l=args.paths['data_folder_l'],
            label_dict=args.label_dict,
            seed=args.seed,
            use_h5=True,
        )
        print(f"Test dataset size for fold {fold_id}: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError(f"[\u2717] Dataset '{args.dataset_name}' not supported.")

def evaluate(model, dataset, args):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for data in dataset:
            x_s, coord_s, x_l, coords_l, label = (
                data['x_s'].to(args.device),
                data['coord_s'].to(args.device),
                data['x_l'].to(args.device),
                data['coords_l'].to(args.device),
                data['label'].to(args.device)
            )
            Y_prob, Y_hat, _ = model(x_s, coord_s, x_l, coords_l, label)
            all_probs.append(Y_prob.cpu().numpy())
            all_preds.append(Y_hat.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if args.n_classes > 2 else roc_auc_score(all_labels, all_probs[:, 1])
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {'auc': auc, 'acc': acc, 'f1': f1}

def main(args):
    args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
    seed_torch(args.seed)

    all_test_auc, all_test_acc, all_test_f1, folds = [], [], [], []

    for i in range(args.k_start, args.k_end + 1):
        print(f"\n=========== Fold {i} ===========")
        datasets = prepare_dataset(args, i)
        _, _, test_dataset = datasets

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_result_dir = os.path.join(args.paths['results_dir'], f"zeroshot_fold{i}_timestamp_{timestamp}")
        os.makedirs(fold_result_dir, exist_ok=True)

        folds.append(i)

        config = ml_collections.ConfigDict()
        config.input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.device = args.device
        config.prototype_number = args.prototype_number
        config.weight_path = "hf_hub:MahmoodLab/conch" #args.paths.get('weight_path', '')

        model = CONCH_ZeroShot_Model(config=config, num_classes=args.n_classes).to(args.device)

        results = evaluate(model, test_dataset, args)

        all_test_auc.append(results['auc'])
        all_test_acc.append(results['acc'])
        all_test_f1.append(results['f1'])

        save_pkl(os.path.join(fold_result_dir, f'split_{i}_results.pkl'), results)

    summary_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'test_acc': all_test_acc,
        'test_f1': all_test_f1
    })

    result_df = pd.DataFrame({
        'metric': ['mean', 'std'],
        'test_auc': [np.mean(all_test_auc), np.std(all_test_auc)],
        'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
        'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)]
    })

    suffix = f"partial_{folds[0]}_{folds[-1]}" if len(folds) != (args.k_end - args.k_start + 1) else "full"
    summary_df.to_csv(os.path.join(args.paths['results_dir'], f"summary_{suffix}.csv"), index=False)
    result_df.to_csv(os.path.join(args.paths['results_dir'], f"result_{suffix}.csv"), index=False)

    print("Zero-shot evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for k, v in config.items():
        setattr(args, k, v)

    if 'paths' in config:
        args.paths = config['paths']
        for subkey, val in args.paths.items():
            if isinstance(val, dict):
                args.paths[subkey] = {k: os.path.abspath(v) for k, v in val.items()}
            else:
                args.paths[subkey] = os.path.abspath(val)

    if not os.path.exists(args.paths.get('weight_path', '')):
        os.makedirs(args.paths['weight_path'], exist_ok=True)
        print(f"[✓] Created weight folder: {args.paths['weight_path']}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("################# SETTINGS ###################")
    for k, v in vars(args).items():
        if k != 'paths':
            print(f"{k}: {v}")
    print("##############################################")

    seed_torch(args.seed)
    main(args)
