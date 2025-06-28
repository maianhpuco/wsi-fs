import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import ml_collections

# Setup paths
sys.path.append("src")
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'

from explainer_ver1 import CONCH_ZeroShot_Model_TopjPooling_MoreText 
from utils.file_utils import save_pkl

def seed_torch(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_dataset(args, fold_id):
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
        args=args, 
    )
    print(f"[✓] Fold {fold_id} | Test set size: {len(test_dataset)}")
    return test_dataset

def run_fold_evaluation(fold_id, args):
    test_dataset = prepare_dataset(args, fold_id)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.text_prompt = args.text_prompt
    config.device = args.device
    config.prototype_number = args.prototype_number
    config.weight_path = "hf_hub:MahmoodLab/conch"

    model = CONCH_ZeroShot_Model_TopjPooling_MoreText(config=config, num_classes=args.n_classes).to(args.device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():  
        for batch in tqdm(test_loader):
            x_s, coord_s, x_l, coord_l, label = batch

            # Handle if dataset returns unbatched tensors
            if x_s.ndim == 2:
                x_s = x_s.unsqueeze(0)
                x_l = x_l.unsqueeze(0)
                coord_s = coord_s.unsqueeze(0)
                coord_l = coord_l.unsqueeze(0)
                label = label.unsqueeze(0)

            x_s = x_s.to(args.device)
            x_l = x_l.to(args.device)
            coord_s = coord_s.to(args.device)
            coord_l = coord_l.to(args.device)
            label = label.to(args.device)

            Y_prob, Y_hat, loss = model(x_s, coord_s, x_l, coord_l, label)
        

            all_preds.append(Y_hat.cpu().item())
            all_probs.append(Y_prob.detach().cpu().numpy())
            all_labels.append(label.cpu().item())
                # ----- Per-class accuracy report -----
                
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    
    n_classes = args.n_classes
    class_results = {}
    for cls in range(n_classes):
        cls_mask = (all_labels == cls)
        total = cls_mask.sum()
        correct = ((all_preds == cls) & cls_mask).sum()
        acc_cls = correct / total if total > 0 else 0.0
        print(f"Class {cls}: acc {acc_cls:.4f}, correct {correct}/{total}")
     
    if all_probs.shape[1] == 2:
        roc_input = all_probs[:, 1]
        roc_args = {}
    else:
        roc_input = all_probs
        roc_args = {'multi_class': 'ovo', 'average': 'macro'}

    try:
        auc = roc_auc_score(all_labels, roc_input, **roc_args)
    except ValueError:
        auc = np.nan

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"[✓] Fold {fold_id}: ACC={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.paths['results_dir'], f"zeroshot_fold{fold_id}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    results = {
        'roc_auc': auc,
        'acc': acc,
        'weighted_f1': f1,
        'preds': all_preds.tolist(),
        'labels': all_labels.tolist()
    }

    save_pkl(os.path.join(result_dir, f"split_{fold_id}_results.pkl"), results)
    return {'fold': fold_id, 'test_auc': auc, 'test_acc': acc, 'test_f1': f1}

def main(args):
    import json

    with open(args.text_prompt_path, "r") as f:
        args.text_prompt = json.load(f)
    print(args.text_prompt)
    # args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze().tolist()
    seed_torch(args.seed)

    summary = []
    for fold_id in range(args.k_start, args.k_end + 1):
        print(f"\n========== Running Fold {fold_id} ==========")
        result = run_fold_evaluation(fold_id, args)
        summary.append(result)

    summary_df = pd.DataFrame(summary)
    suffix = f"partial_{summary[0]['fold']}_{summary[-1]['fold']}" if len(summary) < (args.k_end - args.k_start + 1) else "full"

    result_df = pd.DataFrame({
        'metric': ['mean', 'std'],
        'test_auc': [summary_df.test_auc.mean(), summary_df.test_auc.std()],
        'test_acc': [summary_df.test_acc.mean(), summary_df.test_acc.std()],
        'test_f1': [summary_df.test_f1.mean(), summary_df.test_f1.std()],
    })

    summary_df.to_csv(os.path.join(args.paths['results_dir'], f"summary_{suffix}.csv"), index=False)
    result_df.to_csv(os.path.join(args.paths['results_dir'], f"result_{suffix}.csv"), index=False)

    print("\n✅ Zero-shot evaluation complete.")

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

    main(args)
