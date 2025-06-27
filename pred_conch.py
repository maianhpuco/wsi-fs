import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import ml_collections

# Setup paths
sys.path.append("src")
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'

from explainer_ver1 import CONCH_ZeroShot_Model
sys.path.append("src/externals/CONCH")

 
from conch.downstream.zeroshot_path import zero_shot_classifier, run_mizero
from utils.file_utils import save_pkl


def seed_torch(seed=42):
    """Set all seeds for reproducibility."""
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
    """Prepare TCGA Renal dataset splits for a given fold."""
    if args.dataset_name != 'tcga_renal':
        raise NotImplementedError(f"[✗] Dataset '{args.dataset_name}' not supported.")

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
    print(f"[✓] Fold {fold_id} | Test set size: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def run_fold_evaluation(fold_id, args):
    """Run evaluation for a single fold."""
    _, _, test_dataset = prepare_dataset(args, fold_id)
    
    # Model config
    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.text_prompt = args.text_prompt
    config.device = args.device
    config.prototype_number = args.prototype_number
    config.weight_path = "hf_hub:MahmoodLab/conch"

    # Initialize model
    model = CONCH_ZeroShot_Model(config=config, num_classes=args.n_classes).to(args.device)
    model.eval()

    # Setup classifier weights from prompts
    classnames = [[cls] for cls in args.text_prompt[:args.n_classes]]
    templates = ["a photo of CLASSNAME."]
    classifier = zero_shot_classifier(model, classnames, templates, tokenizer=model.tokenizer, device=args.device)

    # DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Evaluate
    results, dump = run_mizero(
        model=model,
        classifier=classifier,
        dataloader=test_loader,
        device=args.device,
        topj=(1, 5, 10, 50, 100)
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.paths['results_dir'], f"zeroshot_fold{fold_id}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    save_pkl(os.path.join(result_dir, f"split_{fold_id}_results.pkl"), results)

    return {
        'fold': fold_id,
        'test_auc': results['roc_auc'][1],
        'test_acc': results['acc'][1],
        'test_f1': results['weighted_f1'][1]
    }


def main(args):
    args.text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze().tolist()
    seed_torch(args.seed)

    summary = []

    for fold_id in range(args.k_start, args.k_end + 1):
        print(f"\n========== Running Fold {fold_id} ==========")
        result = run_fold_evaluation(fold_id, args)
        summary.append(result)

    # Save summary CSVs
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

    # Normalize paths
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
