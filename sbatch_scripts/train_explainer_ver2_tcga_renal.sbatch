#!/bin/bash

#SBATCH --job-name=ev2_tcga_renal
#SBATCH --output=logs/ev2_tcga_renal.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2

echo "Starting Explainer v2 training for TCGA-Renal"
python train_explainer_ver2.py --config configs_maui/explainer_ver1_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 100
echo "Training complete"
