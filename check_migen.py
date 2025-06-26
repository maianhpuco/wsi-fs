import os
import torch
import argparse
import numpy as np
import sys

sys.path.append("src/externals/wsi_caption")
sys.path.append("src/externals_modified") 

from wsi_caption_tokenizer_origin import Tokenizer
from models.r2gen import R2GenModel

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
    state_dict = torch.load(args.checkpoint_path, map_location=args.device)['state_dict']
    model.load_state_dict(state_dict)

    # Load pre-extracted feature
    feature_path = args.feature_path
    feature_tensor = torch.load(feature_path, map_location=args.device)  # shape [N, d_vf]
    if len(feature_tensor.shape) == 2:
        feature_tensor = feature_tensor.unsqueeze(0)  # make it [1, N, d_vf]

    print(f"Loaded features from {feature_path} with shape {feature_tensor.shape}")

    # Generate caption
    caption = generate_caption(model, feature_tensor, tokenizer, args)
    print(f"\nSlide ID: {args.slide_id}")
    print("Generated caption:")
    print(caption)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True, help='Path to the .pt WSI feature file')
    parser.add_argument('--slide_id', type=str, required=True, help='Slide ID for logging only')
    args = parser.parse_args()

    # Hardcoded model + tokenizer config
    args.ann_path = '/project/hnguyen2/mvu9/datasets/PathText/TCGA-BRCR'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    args.threshold = 0.5
    args.n_classes = 2
    args.drop_prob_lm = 0.5
    args.use_bn = True
    args.checkpoint_path = '/project/hnguyen2/mvu9/pretrained_checkpoints/mi-gen/model_best.pth'

    main(args)
