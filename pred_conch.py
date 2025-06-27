import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from utils.file_utils import save_pkl
from utils.core_utils import Accuracy_Logger, calculate_error
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import ml_collections

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
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' not supported.")


def run_evaluation(model, loader, args, epoch=0):
    device = args.device
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)

    all_probs, all_preds, all_labels = [], [], []
    val_loss = val_error = 0.

    with torch.no_grad():
        for batch_idx, (x_s, coord_s, x_l, coords_l, label) in enumerate(loader):
            x_s, coord_s, x_l, coords_l, label = [d.to(device, non_blocking=True) for d in (x_s, coord_s, x_l, coords_l, label)]
            Y_prob, Y_hat, loss = model(x_s, coord_s, x_l, coords_l, label)

            acc_logger.log(Y_hat, label)
            all_probs.append(Y_prob.cpu().numpy())
            all_preds.append(Y_hat.cpu())
            all_labels.append(label.cpu())

            val_loss += loss.item()
            val_error += calculate_error(Y_hat, label)

    all_probs = np.concatenate(all_probs)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if args.n_classes > 2 else roc_auc_score(all_labels, all_probs[:, 1])
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"[Evaluation] AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
    for i in range(args.n_classes):
        acc_i, correct, count = acc_logger.get_summary(i)
        print(f"  - Class {i}: acc={acc_i:.4f}, correct={correct}/{count}")

    return {'auc': auc, 'acc': acc, 'f1': f1}


def main(args):
    args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()
    seed_torch(args.seed)

    all_test_auc, all_test_acc, all_test_f1, folds = [], [], [], []

    for fold_id in range(args.k_start, args.k_end + 1):
        print(f"\n=========== Fold {fold_id} ===========")
        _, _, test_dataset = prepare_dataset(args, fold_id)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_result_dir = os.path.join(args.paths['results_dir'], f"zeroshot_fold{fold_id}_timestamp_{timestamp}")
        os.makedirs(fold_result_dir, exist_ok=True)
        folds.append(fold_id)

        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.device = args.device
        config.prototype_number = args.prototype_number
        config.weight_path = "hf_hub:MahmoodLab/conch"

        model = CONCH_ZeroShot_Model(config=config, num_classes=args.n_classes).to(args.device)

        # Wrap dataset in a DataLoader for consistency
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        results = run_evaluation(model, test_loader, args)

        all_test_auc.append(results['auc'])
        all_test_acc.append(results['acc'])
        all_test_f1.append(results['f1'])

        save_pkl(os.path.join(fold_result_dir, f'split_{fold_id}_results.pkl'), results)

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

    seed_torch(args.seed)
    main(args)
