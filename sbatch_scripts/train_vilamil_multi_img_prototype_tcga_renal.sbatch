#!/bin/bash

#SBATCH --job-name=vila_mproto_tcga_renal
#SBATCH --output=logs/vila_mproto_tcga_renal.log
#SBATCH --nodelist=compute-0-6
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting Explainer v1b training for TCGA-Renal"
python train_vilamil_multi_img_prototype.py --config configs_maui/vilamil_multi_img_prototype_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 100
echo "Training complete"
