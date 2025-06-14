import os
import argparse
import yaml
import torch
import sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
explainer_path = os.path.abspath(os.path.join(current_dir, 'src'))
sys.path.append(explainer_path)

from datasets.dataset import return_splits_custom
from explainer.explainer_utils import train_prototype_module
from explainer.prototype import ViLaPrototypeTrainer

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    proto_cfg = config['prototype_training']
    data_cfg = config['data']
    out_path = config['paths']['prototype_out_dir']

    # Extract parameters
    input_size = proto_cfg['input_size']
    hidden_size = proto_cfg['hidden_size']
    prototype_number = proto_cfg['prototype_number']
    num_classes = proto_cfg.get('num_classes', 3)  # Default to 3 classes
    epochs = proto_cfg.get('epochs', 10)
    lr = proto_cfg.get('lr', 1e-4)
    batch_size = proto_cfg.get('batch_size', 1)
    use_h5 = data_cfg.get('use_h5', False)

    # Load data configuration
    data_dir_map = data_cfg['data_dir_map']
    label_dict = data_cfg.get('label_dict', {'kich': 0, 'kirp': 1, 'kirc': 2})
    train_csv_path = data_cfg['train_csv_path']
    val_csv_path = data_cfg['val_csv_path']
    test_csv_path = data_cfg['test_csv_path']

    # Validate file paths
    for path in [train_csv_path, val_csv_path, test_csv_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
    for label, data_dir in data_dir_map.items():
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found for {label}: {data_dir}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = return_splits_custom(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        data_dir_map=data_dir_map,
        label_dict=label_dict,
        seed=1,
        print_info=True,
        use_h5=use_h5
    )

    # Initialize model
    model = ViLaPrototypeTrainer(
        input_size=input_size,
        hidden_size=hidden_size,
        prototype_number=prototype_number,
        num_classes=num_classes
    )

    # Train
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
    parser.add_argument('--device', default=default_device, type=str, help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    main(args)