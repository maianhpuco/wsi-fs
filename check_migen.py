import os
import torch
import argparse
import numpy as np
import glob
import sys

sys.path.append("src/externals/wsi_caption")
sys.path.append("src/externals_modified")

from wsi_caption_tokenizer_origin import Tokenizer
from models.r2gen import R2GenModel


@torch.no_grad()
def generate_caption(model, feature_tensor, tokenizer, args):
    model.eval()

    # Add learnable prompt
    att_feats = torch.cat([model.prompt, feature_tensor.to(args.device)], dim=1)
    fc_feats = torch.sum(att_feats, dim=1)

    # Encode features
    memory = model.encoder_decoder(fc_feats, att_feats, mode='train')

    # Greedy decoding loop
    seq = torch.full((1, 1), args.bos_idx, dtype=torch.long).to(args.device)
    state = []
    mask = torch.ones(feature_tensor.shape[:2], dtype=torch.long).unsqueeze(1).to(args.device)

    for _ in range(args.max_seq_length):
        logit, state = model.encoder_decoder.core(seq[:, -1], None, None, memory, state, mask)
        logit = torch.log_softmax(logit, dim=-1)
        _, next_word = torch.max(logit, dim=-1)
        seq = torch.cat([seq, next_word.unsqueeze(1)], dim=1)
        if next_word.item() == args.eos_idx:
            break

    caption = tokenizer.decode(seq.squeeze(0).tolist())
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--ann_path', type=str, default='/project/hnguyen2/mvu9/datasets/PathText/TCGA-BRCA')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_length', type=int, default=600)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--dataset_name', type=str, default='BRCA')

    args = parser.parse_args()

    # Auto-pick feature file if not provided
    if args.feature_path is None:
        pt_files = glob.glob("*.pt")
        if not pt_files:
            raise FileNotFoundError("No .pt file found.")
        args.feature_path = pt_files[0]

    print(f"[INFO] Using feature: {args.feature_path}")

    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer).to(args.device)
    print("[INFO] Initialized model with random weights.")

    # Load feature
    feature_tensor = torch.load(args.feature_path, map_location=args.device)
    if len(feature_tensor.shape) == 2:
        feature_tensor = feature_tensor.unsqueeze(0)
    print(f"[INFO] Feature shape: {feature_tensor.shape}")

    # Run captioning
    caption = generate_caption(model, feature_tensor, tokenizer, args)
    print(f"[OUTPUT] Caption:\n{caption}")


if __name__ == '__main__':
    main()
