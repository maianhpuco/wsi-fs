import os
import argparse
import yaml
import torch
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
explainer_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'externals', 'explainer'))
sys.path.append(explainer_path) 

# from torch.utils.data import Dataset
from explainer.dataset import FeatureDataset 
from explainer.explainer_utils  import train_prototype_module
from explainer.prototype import ViLaPrototypeTrainer 

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    proto_cfg = config['prototype_training']
    pt_dirs = config['paths']['pt_files_dir']
    out_path = config['paths']['prototype_out_dir']

    # Extract parameters
    input_size = proto_cfg['input_size']
    hidden_size = proto_cfg['hidden_size']
    prototype_number = proto_cfg['prototype_number']
    num_classes = proto_cfg.get('num_classes', 3)  # Default to 3 classes
    epochs = proto_cfg.get('epochs', 10)
    lr = proto_cfg.get('lr', 1e-4)
    batch_size = proto_cfg.get('batch_size', 1)

    # Load label map from config for flexibility
    label_map = config.get('label_map', {'kich': 0, 'kirp': 1, 'kirc': 2})

    # Initialize dataset and model
    dataset = FeatureDataset(pt_dirs, label_map)
    model = ViLaPrototypeTrainer(
        input_size=input_size,
        hidden_size=hidden_size,
        prototype_number=prototype_number,
        num_classes=num_classes
    )

    # Train
    train_prototype_module(
        model, dataset, out_path,
        epochs=epochs, lr=lr, batch_size=batch_size, device=args.device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')

    # Automatically use GPU if available
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device, type=str, help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    main(args)