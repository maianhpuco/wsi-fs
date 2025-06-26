import torch
import sys

sys.path.append("src/externals_modified/wsi_caption")
from dataloaders import R2DataLoader 

sys.path.append("src/externals/wsi_caption")

import os
import argparse
import numpy as np
import random
import warnings

from modules.tokenizers import Tokenizer
# from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='.../pt_files')
    parser.add_argument('--ann_path', type=str, default='.../TCGA_BRCA')
    parser.add_argument('--split_path', type=str, default='../ocr/dataset_csv/splits_0.csv')
    
    # Data loader
    parser.add_argument('--dataset_name', type=str, default='TCGA', choices=['TCGA'])
    parser.add_argument('--max_fea_length', type=int, default=10000)
    parser.add_argument('--max_seq_length', type=int, default=600)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)

    # Model settings
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--n_classes', type=int, default=2)

    # Sampling
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=1)
    parser.add_argument('--suppress_UNK', type=int, default=1)
    parser.add_argument('--block_trigrams', type=int, default=1)

    # Trainer
    parser.add_argument('--n_gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--epochs_val', type=int, default=2)
    parser.add_argument('--start_val', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='results/BRCA')
    parser.add_argument('--record_dir', type=str, default='records/')
    parser.add_argument('--save_period', type=int, default=1)
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')
    parser.add_argument('--early_stop', type=int, default=20)

    # Optimizer
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr_ed', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--amsgrad', type=bool, default=True)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Misc
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--mode', type=str, default='Test', choices=['Train', 'Test'])
    parser.add_argument('--debug', type=str, default='False')
    parser.add_argument('--seed', type=int, default=9233)
    parser.add_argument('--resume', type=str, help='resume from checkpoint')

    args = parser.parse_args()
    for k, v in vars(args).items():
        if v == 'True': setattr(args, k, True)
        elif v == 'False': setattr(args, k, False)
    return args


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.n_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_seeds(args.seed)

    tokenizer = Tokenizer(args)

    train_loader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_loader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_loader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    model = R2GenModel(args, tokenizer).to(device)

    if args.mode == 'Test':
        resume_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
        print(f"Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)['state_dict']
        model.load_state_dict({k: v for k, v in checkpoint.items() if k in model.state_dict()})

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    criterion = compute_loss
    metrics = compute_scores

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                      train_loader, val_loader, test_loader)

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'Train':
        trainer.train()
    else:
        trainer.test()


if __name__ == '__main__':
    args = parse_args()
    main(args)
