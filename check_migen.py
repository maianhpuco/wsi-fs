import os
import torch
import argparse
import numpy as np
import sys
import glob

sys.path.append("src/externals/wsi_caption")
sys.path.append("src/externals_modified") 

from wsi_caption_tokenizer_origin import Tokenizer
from models.r2gen import R2GenModel



@torch.no_grad()
def generate_caption(model, feature_tensor, tokenizer, args):
    model.eval()

    att_feats = torch.cat([model.prompt, feature_tensor.to(args.device)], dim=1)
    fc_feats = torch.sum(att_feats, dim=1)
    memory = model.encoder_decoder(fc_feats, att_feats, mode='encode')

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

# @torch.no_grad()
# def generate_caption(model, feature_tensor, tokenizer, args):
#     model.eval()
#     att_mask = torch.ones(feature_tensor.shape[:2], dtype=torch.long).to(args.device)
#     memory = model._encode(None, feature_tensor.to(args.device), att_mask)

#     seq = torch.full((1, 1), args.bos_idx, dtype=torch.long).to(args.device)
#     state = []
#     mask = att_mask.unsqueeze(1)

#     for _ in range(args.max_seq_length):
#         logit, state = model.core(seq[:, -1], None, None, memory, state, mask)
#         logit = torch.log_softmax(logit, dim=-1)
#         _, next_word = torch.max(logit, dim=-1)
#         seq = torch.cat([seq, next_word.unsqueeze(1)], dim=1)
#         if next_word.item() == args.eos_idx:
#             break

#     caption = tokenizer.decode(seq.squeeze(0).tolist())
#     return caption

def main(args):
    # Initialize tokenizer and model with random weights
    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer).to(args.device)
    print("Initialized model without checkpoint (random weights).")

    # Load any available .pt feature file
    args.feature_path = glob.glob("*.pt")[0]
    print(f"Using feature file: {args.feature_path}")
    feature_tensor = torch.load(args.feature_path, map_location=args.device)
    if len(feature_tensor.shape) == 2:
        feature_tensor = feature_tensor.unsqueeze(0)

    print(f"Loaded feature shape: {feature_tensor.shape}")

    # Run forward pass
    caption = generate_caption(model, feature_tensor, tokenizer, args)
    print("Generated caption:")
    print(caption)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Hardcoded config
    args.ann_path = '/project/hnguyen2/mvu9/datasets/PathText/TCGA-BRCA'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.d_model = 512
    args.d_ff = 512
    args.d_vf = 1764 #1024
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
    args.dataset_name = 'BRCA'

    main(args)
