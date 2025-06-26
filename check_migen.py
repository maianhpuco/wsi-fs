import os
import torch
import argparse
import yaml
import sys
import numpy as np

sys.path.append("src/externals/wsi_vqa")

from models.r2gen import VQA_model
from modules.text_extractor import create_text_extractor
from transformers import BertTokenizerFast, AutoTokenizer

# === Build Tokenizer ===
def Build_Tokenizer(args):
    if args.text_extractor == 'bioclinicalbert':
        model_path = '/project/hnguyen2/mvu9/pretrained_checkpoints/bioclinicalbert'
        tokenizer = BertTokenizerFast.from_pretrained(model_path, tokenizer_class=AutoTokenizer)

    elif args.text_extractor == 'pubmedbert':
        model_path = '/project/hnguyen2/mvu9/pretrained_checkpoints/pubmedbert'
        tokenizer = BertTokenizerFast.from_pretrained(model_path, tokenizer_class=AutoTokenizer)

    elif args.text_extractor == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("/output/path")

    else:
        from modules.tokenizers import Tokenizer
        tokenizer = Tokenizer(args)

    # Combine transformer tokenizer with your custom tokenizer
    from modules.tokenizers import Tokenizer, MixedTokenizer
    tokenizer_a = Tokenizer(args)
    return MixedTokenizer(tokenizer, tokenizer_a)

# === Caption Generation ===
@torch.no_grad()
def generate_caption(model, feature_tensor, tokenizer, args):
    model.eval()

    question = "What is the histologic subtype?"
    question_ids = tokenizer.encode_input(question)
    question_ids = torch.tensor(question_ids).unsqueeze(0).to(args.device)  # [1, T]
    question_mask = (question_ids > 0).long()

    feature_tensor = feature_tensor.to(args.device)
    memory = model.encoder(feature_tensor, question_ids, question_mask)

    seq = torch.full((1, 1), args.bos_idx, dtype=torch.long).to(args.device)
    state = []

    for _ in range(args.max_seq_length):
        logits, state = model.decoder(seq[:, -1], memory, state)
        next_token = torch.argmax(logits, dim=-1)
        seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == args.eos_idx:
            break

    caption_ids = seq.squeeze(0).tolist()[1:]  # remove <bos>
    caption = tokenizer.decode(caption_ids)
    return caption

# === Load Features ===
def load_feature(args):
    slide_path = args.slide_path
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"Slide feature not found: {slide_path}")
    feat = torch.load(slide_path)
    feat = feat.unsqueeze(0)  # [1, N, d_vf]
    return feat

# === Main ===
def main(args):
    tokenizer = Build_Tokenizer(args)
    text_extractor = create_text_extractor(args.text_extractor, override_image_size=None)
    model = VQA_model(args, tokenizer, text_extractor).to(args.device)

    print(f"Loading model from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)['state_dict']
    model.load_state_dict(checkpoint, strict=False)

    feature_tensor = load_feature(args)
    caption = generate_caption(model, feature_tensor, tokenizer, args)
    
    print(f"\nSlide ID: {args.slide_id}")
    print("Generated caption:", caption)

# === Entry Point ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    # Load YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    # Device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters
    args.d_model = 512
    args.d_ff = 512
    args.d_vf = 1024
    args.num_heads = 4
    args.num_layers = 3
    args.dropout = 0.1
    args.max_seq_length = 60
    args.bos_idx = 1243
    args.eos_idx = 0
    args.pad_idx = 0
    args.threshold = 1.0
    args.text_extractor = 'bioclinicalbert'

    args.ann_path = '/project/hnguyen2/mvu9/folder_04_ma/wsi-fs/src/externals/wsi_vqa/dataset/WSI_captions'
    # Hardcoded inputs
    args.slide_id = 'TCGA-UW-A7GP-11Z-00-DX1.C89DD837-4B77-4CB5-8FAB-DF9315892B9B'
    args.slide_path = os.path.join(
        "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/features_fp/pt_files",
        f"{args.slide_id}.pt"
    )
    args.checkpoint_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/wsi-vqa/W2T_resnet.pth"

    main(args)
