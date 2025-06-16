jpt:
	sbatch sbatch_scripts/juputer_notebook.sbatch

train_proto_tcga_renal:
	python train_prototype.py --config configs_maui/prototype_tcga_renal.yaml

train_explainer_ver1_tcga_renal: 
	python train_explainer_ver1.py --config configs_maui/explainer_ver1_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20
train_explainer_ver1_tcga_renal: 
	python train_vilamil.py --config configs_maui/vilamil_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20  