jpt:
	sbatch sbatch_scripts/juputer_notebook.sbatch

check_proto:
	python train_prototype.py --config configs_maui/prototype_tcga_renal.yaml