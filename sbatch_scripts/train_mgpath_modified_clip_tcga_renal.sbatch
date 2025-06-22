#!/bin/bash

#SBATCH --job-name=mg_clip_tcga_renal
#SBATCH --output=logs/mg_clip_tcga_renal.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting Explainer v1 training for TCGA-Renal"

python train_mgpath_modified.py --config configs_maui/mgpath_modified_clip_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 100

echo "Training complete"
