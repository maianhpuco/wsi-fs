import os
import torch
import argparse
import numpy as np
import yaml
import sys 
# from dataloader import return_splits_custom 

sys.path.append("src/externals/wsi_caption")
sys.path.append("src/externals_modified") 
from wsi_caption_tokenizer_origin import Tokenizer
# from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel

def load_wsi_feature_from_dataset(args):
    """
    Use your MIL dataset and config to retrieve a test sample feature.
    """
    from src.datasets.single_scale.tcga import return_splits_custom
 
    dataset_name = args.dataset_name.lower()

    if dataset_name in ['tcga_renal', 'tcga_lung']:
        # Parse YAML config
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        data_dir_map = config['data_dir_map']
        label_dict = {label: idx for idx, label in enumerate(data_dir_map.keys())}

        split_folder = config['split_dir']
        fold = args.fold  # which fold to use
        fold_dir = os.path.join(split_folder, f"fold_{fold}")
        train_csv_path = os.path.join(fold_dir, "train.csv")
        val_csv_path = os.path.join(fold_dir, "val.csv")
        test_csv_path = os.path.join(fold_dir, "test.csv")

        _, _, test_dataset = return_splits_custom(
            train_csv_path,
            val_csv_path,
            test_csv_path,
            data_dir_map=data_dir_map,
            label_dict=label_dict,
            seed=42,
            print_info=False,
            use_h5=False
        )

        # Load feature for the given slide_id
        feature_tensor = test_dataset.get_features_by_slide_id(args.slide_id)
        return feature_tensor.unsqueeze(0)  # shape [1, N, d_vf]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported yet.")

@torch.no_grad()
def generate_caption(model, feature_tensor, tokenizer, args):
    model.eval()

    att_mask = torch.ones(feature_tensor.shape[:2], dtype=torch.long).to(args.device)
    memory = model._encode(None, feature_tensor.to(args.device), att_mask)

    seq = torch.full((1, 1), args.bos_idx, dtype=torch.long).to(args.device)
    state = []
    mask = att_mask.unsqueeze(1)

    for _ in range(args.max_seq_length):
        logit, state = model.core(seq[:, -1], None, None, memory, state, mask)
        logit = torch.log_softmax(logit, dim=-1)
        _, next_word = torch.max(logit, dim=-1)
        seq = torch.cat([seq, next_word.unsqueeze(1)], dim=1)
        if next_word.item() == args.eos_idx:
            break

    caption = tokenizer.decode(seq.squeeze(0).tolist())
    return caption

def main(args):

    # Tokenizer and model
    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer).to(args.device)
    
    
    print(f"Loading model from: {args.checkpoint_path}")
    
    state_dict = torch.load(
        args.checkpoint_path, 
        map_location=args.device)['state_dict']
    model.load_state_dict(state_dict)

    # Load feature from dataset
    feature_tensor = load_wsi_feature_from_dataset(args, config)

    # Generate caption
    caption = generate_caption(model, feature_tensor, tokenizer, args)
    print(f"\nSlide ID: {args.slide_id}")
    print("Generated caption:")
    print(caption)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--slide_id', type=str, required=True, help='Slide ID to generate caption for')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    # parser.add_argument('--dataset_name', type=str, required=True, help='e.g. tcga_renal')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    
    args = parser.parse_args()
    slide_id =  'TCGA-UW-A7GY-11Z-00-DX1.7410A3EA-BFFD-4744-8DB2-66A409C0BFA9'
    # === Load and inject config ===
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)  # shallow merge (top-level only)

    # === Setup device and seed ===
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.anno_path = '/project/hnguyen2/mvu9/datasets/PathText/TCGA-KICH'
    # Set model args (hardcoded)
    args.d_model = 512
    args.d_ff = 512
    args.d_vf = 1024
    args.num_heads = 4
    args.num_layers = 3
    args.dropout = 0.1
    args.max_seq_length = 600
    args.bos_idx = 0
    args.eos_idx = 0
    args.pad_idx = 0
    
    main(args)