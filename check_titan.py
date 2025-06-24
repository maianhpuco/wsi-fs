import torch
import pickle
import os
import sys

# Add TITAN repo to path
sys.path.append(os.path.abspath("src/externals/TITAN"))

# TITAN imports
from modeling_titan import TITANForReportGeneration, TITANConfig
from transformers import AutoTokenizer

# Path to your downloaded TITAN model
TITAN_MODEL_PATH = "/project/hnguyen2/mvu9/pretrained_checkpoints/TITAN"

# Load tokenizer (if used by report generation)
tokenizer = AutoTokenizer.from_pretrained(TITAN_MODEL_PATH)

# Load TITAN model directly using TITAN class
config = TITANConfig.from_pretrained(TITAN_MODEL_PATH)
model = TITANForReportGeneration.from_pretrained(TITAN_MODEL_PATH, config=config).eval().cuda()

# Load sample features
feature_pkl_path = os.path.join(TITAN_MODEL_PATH, "TCGA_TITAN_features.pkl")
with open(feature_pkl_path, "rb") as f:
    features = pickle.load(f)

# Pick a sample slide
sample_slide_id = list(features.keys())[0]
slide_feature = features[sample_slide_id]

# Convert to tensor
slide_tensor = torch.tensor(slide_feature).unsqueeze(0).cuda()

# Generate report
with torch.no_grad():
    report = model.generate_report(slide_tensor)

# Display result
print(f"- Slide ID: {sample_slide_id}")
print("- Generated Report:")
print(report)
