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

sys.path.append("src")
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'

from explainer_ver1 import CONCH_Finetuning_Model_TopjPooling_MoreText
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
    print(f"[✓] Fold {fold_id} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def train_and_save(model, train_loader, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x_s, coord_s, x_l, coord_l, label = [b.to(args.device) for b in batch]
            if x_s.ndim == 2:
                x_s = x_s.unsqueeze(0)
                x_l = x_l.unsqueeze(0)
                coord_s = coord_s.unsqueeze(0)
                coord_l = coord_l.unsqueeze(0)
                label = label.unsqueeze(0)

            optimizer.zero_grad()
            _, _, loss = model(x_s, coord_s, x_l, coord_l, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[✓] Epoch {epoch+1} - Avg Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(args.paths['weight_path'], f"conch_model.pt"))
    print(f"[✓] Model saved to {args.paths['weight_path']}/conch_model.pt")


def evaluate(model, test_loader, args, fold_id):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x_s, coord_s, x_l, coord_l, label = [b.to(args.device) for b in batch]
            if x_s.ndim == 2:
                x_s = x_s.unsqueeze(0)
                x_l = x_l.unsqueeze(0)
                coord_s = coord_s.unsqueeze(0)
                coord_l = coord_l.unsqueeze(0)
                label = label.unsqueeze(0)
            Y_prob, Y_hat, _ = model(x_s, coord_s, x_l, coord_l, label)
            all_preds.append(Y_hat.cpu().item())
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().item())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    except:
        auc = np.nan

    print(f"[✓] Fold {fold_id} Evaluation: ACC={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.paths['results_dir'], f"train_fold{fold_id}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    results = {
        'roc_auc': auc,
        'acc': acc,
        'weighted_f1': f1,
        'preds': all_preds.tolist(),
        'labels': all_labels.tolist()
    }
    save_pkl(os.path.join(result_dir, f"fold_{fold_id}_results.pkl"), results)


def main(args):
    import json
    with open(args.text_prompts_path, "r") as f:
        args.text_prompts = json.load(f)

    seed_torch(args.seed)
    summary = []

    for fold_id in range(args.k_start, args.k_end + 1):
        print(f"\n========== Fold {fold_id} ==========")
        train_dataset, val_dataset, test_dataset = prepare_dataset(args, fold_id)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompts = args.text_prompts
        config.device = args.device
        config.prototype_number = args.prototype_number
        config.weight_path = "hf_hub:MahmoodLab/conch"

        model = CONCH_Finetuning_Model_TopjPooling_MoreText(config=config, num_classes=args.n_classes).to(args.device)
        train_and_save(model, train_loader, args)
        result = evaluate(model, test_loader, args, fold_id)
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

    print("\n✅ Training and evaluation complete.")


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
