#!/bin/bash

#SBATCH --job-name=pcp
#SBATCH --output=sbatchs/logs/preprocessing_camelyon16_patching.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00

python preprocessing/camelyon16/patching.py 