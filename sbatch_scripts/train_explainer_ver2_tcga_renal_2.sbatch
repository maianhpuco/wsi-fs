#!/bin/bash

#SBATCH --job-name=ev2c_tcga_renal
#SBATCH --output=logs/ev2c_tcga_renal.log
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2

echo "Starting Explainer v21c training for TCGA-Renal"
python main_train_ver2c.py --config configs_maui/ver2c_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20
echo "Training complete"
