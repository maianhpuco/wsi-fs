import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModel

# Path to your downloaded TITAN model
TITAN_MODEL_PATH = "/project/hnguyen2/mvu9/pretrained_checkpoints/TITAN"

# Load TITAN model with custom code
model = AutoModel.from_pretrained(
    TITAN_MODEL_PATH,
    trust_remote_code=True
).eval().cuda()

# Load tokenizer if needed (optional depending on model.generate_report())
tokenizer = AutoTokenizer.from_pretrained(TITAN_MODEL_PATH)

# Path to a sample feature file from TCGA (e.g., pre-extracted features)
feature_pkl_path = os.path.join(TITAN_MODEL_PATH, "TCGA_TITAN_features.pkl")

# Load features
with open(feature_pkl_path, "rb") as f:
    features = pickle.load(f)

# Select a sample slide (use the key from the dict)
sample_slide_id = list(features.keys())[0]
slide_feature = features[sample_slide_id]

# Convert to torch tensor
slide_tensor = torch.tensor(slide_feature).unsqueeze(0).cuda()

# Generate report using TITAN
with torch.no_grad():
    report = model.generate_report(slide_tensor)

# Print output
print(f"🧪 Slide ID: {sample_slide_id}")
print("📝 Generated Report:")
print(report)
