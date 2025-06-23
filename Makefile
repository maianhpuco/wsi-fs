jpt:
	sbatch sbatch_scripts/juputer_notebook.sbatch

train_proto_tcga_renal:
	python train_prototype.py --config configs_maui/prototype_tcga_renal.yaml

train_explainer_ver1_tcga_renal: 
	python train_explainer_ver1.py --config configs_maui/explainer_ver1_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20

# train_vilamil_tcga_renal: 
# 	python train_vilamil.py --config configs_maui/vilamil_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20  

pred_conch_tcga_renal:
	python pred_conch.py --config configs_maui/conch_tcga_renal.yaml 

mgpath_modified_tcga_renal:
	python train_mgpath_modified.py --config configs_maui/mgpath_modified_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 10

train_vilamil_tcga_renal:
	python train_explainer_ver1b.py --config configs_maui/explainer_ver1b_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 100



# ==================== SBATCH COMMANDS ==================== 
sbatch_train_explainer_ver1_tcga_renal: # use wsi-fs-2 environment 
	sbatch sbatch_scripts/train_explainer_ver1_tcga_renal.sbatch 	
sbatch_train_explainer_ver1b_tcga_renal: # use wsi-fs-2 environment 
	sbatch sbatch_scripts/train_explainer_ver1b_tcga_renal.sbatch 	

# sbatch_train_explainer_ver1c_tcga_renal: # use wsi-fs-2 environment 
# 	sbatch sbatch_scripts/train_explainer_ver1c_tcga_renal.sbatch 	
 
# sbatch_train_explainer_ver1d_tcga_renal: # use wsi-fs-2 environment 
# 	sbatch sbatch_scripts/train_explainer_ver1d_tcga_renal.sbatch 	
# modify mg path
sbatch_train_mgpath_modified_clip_tcga_renal: # use wsi-fs-2 environment 
	sbatch sbatch_scripts/train_mgpath_modified_clip_tcga_renal.sbatch  

sbatch_train_mgpath_modified_conch_tcga_renal: # use wsi-fs-2 environment 
	sbatch sbatch_scripts/train_mgpath_modified_conch_tcga_renal.sbatch 	

sbatch_train_mgpath_modified_quilt_tcga_renal: # use wsi-fs-2 environment 
	sbatch sbatch_scripts/train_mgpath_modified_quilt_tcga_renal.sbatch 		
	
sbatch_train_vilamil_multi_img_prototype_tcga_renal: # use wsi-fs-2 environment 
	sbatch sbatch_scripts/train_vilamil_multi_img_prototype_tcga_renal.sbatch 