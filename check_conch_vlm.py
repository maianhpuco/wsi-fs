import os
import sys
import torch

# Add CONCH source directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
conch_path = os.path.abspath(os.path.join(current_dir, "../../src/externals/CONCH"))
sys.path.append(conch_path)

from conch.model import CONCHVisionLanguageModel
from conch.tokenizer import CONCHTokenizer

# Optional: specify a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CONCHVisionLanguageModel.from_pretrained(
    pretrained_model_name_or_path="hf_hub:MahmoodLab/conch",
    device_map=device
)
model = model.to(device)
model.eval()

# Load tokenizer
tokenizer = CONCHTokenizer()

# Dummy text
text = ["A WSI of clear cell renal cell carcinoma"]

# Tokenize
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

# Encode text
with torch.no_grad():
    text_features = model.encode_text(inputs['input_ids'], inputs['attention_mask'])

print("Text encoding successful.")
print("Text feature shape:", text_features.shape)
print("First 5 values of first vector:", text_features[0][:5])
