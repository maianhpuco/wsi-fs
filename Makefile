jpt:
	sbatch sbatch_scripts/juputer_notebook.sbatch

train_proto_tcga_renal:
	python train_prototype.py --config configs_maui/prototype_tcga_renal.yaml

train_explainer_ver1_tcga_renal: 
	python train_explainer_ver1.py --config configs_maui/explainer_ver1_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20

# train_vilamil_tcga_renal: 
# 	python train_vilamil.py --config configs_maui/vilamil_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 20  





# ======================================= 
pred_conch_meanpooling_tcga_renal:
	python main_pred_conch_meanpooling.py --config configs_maui/conch_tcga_renal.yaml 
pred_conch_topjpooling_tcga_renal:
	python main_pred_conch_topjpooling.py --config configs_maui/conch_tcga_renal.yaml 
pred_conch_topjpooling_tcga_renal_nodesc:
	python main_pred_conch_topjpooling.py --config configs_maui/conch_tcga_renal_no_desc.yaml 
pred_conch_topjpooling_tcga_renal_more_text:
	python main_pred_conch_topjpooling_more_text.py --config configs_maui/conch_tcga_renal_more_text.yaml 
pred_conch_topjpooling_tcga_renal_more_text:
	python main_pred_conch_topjpooling_more_text.py --config configs_maui/conch_tcga_renal_more_text.yaml 
 
# =======================================  
train_conch_topjpooling_tcga_renal_more_text:
	python main_train_conch_topjpooling_more_text.py --config configs_maui/conch_finetune_tcga_renal_more_text.yaml --k_start 1 --k_end 1 --max_epochs 20 
train_conch_topjpooling_tcga_renal_less_text:
	python main_train_conch_topjpooling_more_text.py --config configs_maui/conch_finetune_tcga_renal_less_text.yaml --k_start 1 --k_end 1 --max_epochs 20 





# ======================================= 
pred_conch_meanpooling_tcga_lung:
	python main_pred_conch_meanpooling.py --config configs_maui/conch_tcga_lung.yaml 
pred_conch_topjpooling_tcga_lung:
	python main_pred_conch_topjpooling.py --config configs_maui/conch_tcga_lung.yaml 

 

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

train_all: sbatch_train_mgpath_modified_clip_tcga_renal sbatch_train_mgpath_modified_conch_tcga_renal sbatch_train_mgpath_modified_quilt_tcga_renal sbatch_train_vilamil_multi_img_prototype_tcga_renal 


# export HF_HOME=/project/hnguyen2/mvu9/model_cache
run_migen:
	python check_migen.py 

run_caption:
	python main_wsi_caption.py  --image_dir /project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/features_fp/pt_files \
	--ann_path /project/hnguyen2/mvu9/datasets/PathText/TCGA-KICH \
	--split_path kich_split.csv 

run_caption_test:
	python main_wsi_caption_test.py  --image_dir /project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/features_fp/pt_files \
	--ann_path /project/hnguyen2/mvu9/datasets/PathText/TCGA-KICH \
	--split_path kich_split.csv \
	--checkpoint_dir ./results/BRCA \
	--mode Test

#/project/hnguyen2/mvu9/folder_04_ma/wsi-fs/src/externals/wsi_caption/ocr/dataset_csv/splits_0.csv 
#--config configs_maui/migen_tcga_renal.yaml --fold 1 