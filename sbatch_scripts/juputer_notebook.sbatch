#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=logs/jupyter.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:0  # if needed, adjust or remove

# Load your conda env or module
source ~/.bashrc
conda activate jupyter_notebook_env  # or module load python

# Launch Jupyter
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
