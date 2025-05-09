# Default target
all: metadata_kich

# Target to generate metadata for TCGA-KICH
metadata_kich:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_kich.yaml

# Optional: Clean target if you want to remove generated Excel files
clean_kich:
	rm -f /project/hnguyen2/mvu9/datasets/TGCA-metadata/KICH/metadata/*.xlsx

patching_kich:
	python scripts/preprocessing/tcga/patching.py --config configs/data_tcga_kich.yaml

.PHONY: all metadata_kich clean_kich


