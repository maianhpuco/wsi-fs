import torch
import pickle
import os
import sys

# Add TITAN submodule path to Python path
TITAN_SRC_PATH = os.path.abspath("src/externals/TITAN")
sys.path.append(TITAN_SRC_PATH)

# TITAN-specific imports
from modeling_titan import TITANForReportGeneration, TITANConfig
from transformers import AutoTokenizer

# === Path setup ===
TITAN_MODEL_PATH = "/project/hnguyen2/mvu9/pretrained_checkpoints/TITAN"
FEATURE_PKL_PATH = os.path.join(TITAN_MODEL_PATH, "TCGA_TITAN_features.pkl")

# === Load tokenizer (optional, depending on TITAN usage) ===
tokenizer = AutoTokenizer.from_pretrained(TITAN_MODEL_PATH)

# === Load TITAN model from config and weights ===
config = TITANConfig.from_pretrained(TITAN_MODEL_PATH)
model = TITANForReportGeneration.from_pretrained(
    TITAN_MODEL_PATH, config=config
).eval().cuda()

# === Load precomputed TCGA slide features ===
with open(FEATURE_PKL_PATH, "rb") as f:
    slide_features_dict = pickle.load(f)

# === Select a sample slide ===
sample_slide_id = list(slide_features_dict.keys())[0]
slide_feature = slide_features_dict[sample_slide_id]  # shape: [n_patches, feature_dim]

# === Convert features to torch tensor ===
slide_tensor = torch.tensor(slide_feature).unsqueeze(0).cuda()  # shape: [1, n_patches, d]

# === Generate pathology report ===
with torch.no_grad():
    generated_report = model.generate_report(slide_tensor)

# === Output ===
print(f"\nSlide ID: {sample_slide_id}")
print("Generated Report:\n")
print(generated_report)
