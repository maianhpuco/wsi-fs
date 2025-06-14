import os
import argparse
import yaml
import torch
import sys
from datetime import datetime

# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
_path = os.path.abspath(os.path.join(current_dir,'src'))
sys.path.append(_path)

from explainer.explainer_utils import train_prototype_module
from explainer.prototype import ViLaPrototypeTrainer
from datasets.tcga import return_splits_custom


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # === Load prototype training hyperparameters ===
    proto_cfg = config['prototype_training']
    input_size = proto_cfg['input_size']
    hidden_size = proto_cfg['hidden_size']
    prototype_number = proto_cfg['prototype_number']
    num_classes = proto_cfg.get('num_classes', 3)
    epochs = proto_cfg.get('epochs', 10)
    lr = proto_cfg.get('lr', 1e-4)
    batch_size = proto_cfg.get('batch_size', 4)

    # === Load path and label map ===
    pt_dirs = config['paths']['pt_files_dir']
    out_path = config['paths']['prototype_out_dir']
    label_dict = config['label_dict']

    # Create dummy train/val/test CSVs (assumes you've pre-split and saved them)
    split_folder = config['paths'].get('split_folder', None)
    if not split_folder:
        raise ValueError("split_folder not found in config['paths']")

    train_csv_path = os.path.join(split_folder, 'train.csv')
    val_csv_path = os.path.join(split_folder, 'val.csv')
    test_csv_path = os.path.join(split_folder, 'test.csv')

    for p in [train_csv_path, val_csv_path, test_csv_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")

    # === Load datasets ===
    train_dataset, val_dataset, test_dataset = return_splits_custom(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        data_dir_map=pt_dirs,
        label_dict=label_dict,
        seed=1,
        print_info=True,
        use_h5=False
    ) 

    # === Build model ===
    model = ViLaPrototypeTrainer(
        input_size=input_size,
        hidden_size=hidden_size,
        prototype_number=prototype_number,
        num_classes=num_classes
    )

    # === Train ===
    train_prototype_module(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_path=out_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=args.device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device, type=str, help='Device to use')
    args = parser.parse_args()
    main(args)
