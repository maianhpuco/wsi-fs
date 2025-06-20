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
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, 'src')))

# Assume CONCH is available in a custom module
from conch.model import CONCHVisionLanguageModel  # Placeholder for CONCH model
from conch.tokenizer import CONCHTokenizer  # Placeholder for CONCH tokenizer

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
        data_dir_s_mapping = args.paths['data_folder_s']
        data_dir_l_mapping = args.paths['data_folder_l']
        
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
        print(f"Test dataset size for fold {fold_id}: {len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset
    else:
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' not supported.")

class CONCH_ZeroShot_Model(torch.nn.Module):
    def __init__(self, config, num_classes=3):
        super(CONCH_ZeroShot_Model, self).__init__()
        self.device = config.device
        self.num_classes = num_classes
        
        # Load CONCH model (replace with actual path to CONCH weights)
        self.conch_model = CONCHVisionLanguageModel.from_pretrained(
            "path/to/conch/weights", device_map=self.device
        )
        self.tokenizer = CONCHTokenizer()  # CONCH-specific tokenizer
        self.text_prompt = config.text_prompt
        
        # Attention mechanism for aggregating patch features
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D), torch.nn.Tanh()
        ).to(self.device)
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D), torch.nn.Sigmoid()
        ).to(self.device)
        self.attention_weights = torch.nn.Linear(self.D, self.K).to(self.device)

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.conch_model.encode_text(tokenized['input_ids'], tokenized['attention_mask'])
        return text_features

    def forward(self, x_s, coord_s, x_l, coords_l, label=None):
        # Process low magnification patches
        M = x_s.float()  # Shape: [batch, num_patches, input_size]
        A_V = self.attention_V(M)
        A_U = self.attention_U(M)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = torch.nn.functional.softmax(A, dim=1)
        image_features_low = torch.mm(A, M)  # Aggregated low-mag features

        # Process high magnification patches
        M_high = x_l.float()
        A_V_high = self.attention_V(M_high)
        A_U_high = self.attention_U(M_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = torch.nn.functional.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, M_high)  # Aggregated high-mag features

        # Encode text prompts
        text_features = self.encode_text(self.text_prompt)
        
        # Zero-shot prediction: compute logits for low and high magnification
        logits_low = image_features_low @ text_features[:self.num_classes].T
        logits_high = image_features_high @ text_features[self.num_classes:].T
        logits = logits_low + logits_high

        Y_prob = torch.nn.functional.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        loss = None
        if label is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, label)

        return Y_prob, Y_hat, loss

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
        args.results_dir = os.path.join(args.paths['results_dir'], f"zeroshot_fold{i}_timestamp_{timestamp}")
        os.makedirs(args.results_dir, exist_ok=True)

        seed_torch(args.seed)
        folds.append(i)

        config = ml_collections.ConfigDict()
        config.input_size = 1024  # Adjust based on CONCH feature size
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.device = args.device
        config.prototype_number = args.prototype_number
        model = CONCH_ZeroShot_Model(config=config, num_classes=args.n_classes).to(args.device)

        # Evaluate on test set
        results = evaluate(model, test_dataset, args)

        all_test_auc.append(results['auc'])
        all_test_acc.append(results['acc'])
        all_test_f1.append(results['f1'])

        save_pkl(os.path.join(args.results_dir, f'split_{i}_results.pkl'), results)

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
        'test_acc': [np.mean(all_test_acc), np.std(all_test_acc)],
        'test_f1': [np.mean(all_test_f1), np.std(all_test_f1)]
    })

    args.k = args.k_end - args.k_start + 1
    suffix = f"partial_{folds[0]}_{folds[-1]}" if len(folds) != args.k else "full"
    summary_df.to_csv(os.path.join(args.results_dir, f"summary_{suffix}.csv"), index=False)
    result_df.to_csv(os.path.join(args.results_dir, f"result_{suffix}.csv"), index=False)
    print("Zero-shot evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--k_start', type=int, required=True)
    parser.add_argument('--k_end', type=int, required=True)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--prototype_number', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
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