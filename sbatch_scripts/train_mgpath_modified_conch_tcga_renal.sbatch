#!/bin/bash

#SBATCH --job-name=mg_conch_tcga_renal
#SBATCH --output=logs/mg_conch_tcga_renal.log
#SBATCH --nodelist=compute-0-4
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting Explainer v1 training for TCGA-Renal"

python train_mgpath_modified.py --config configs_maui/mgpath_modified_conch_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 100

echo "Training complete"
