import os
import sys
import torch
import numpy as np
from PIL import Image

# Add CONCH module
current_dir = os.path.dirname(os.path.abspath(__file__))
conch_path = os.path.abspath(os.path.join(current_dir, "../../src/externals/CONCH"))
sys.path.append(conch_path)

from conch.open_clip_custom import create_model_from_pretrained, tokenize

# Hugging Face cache and token
os.environ["HF_HOME"] = "/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface"
hf_token = os.environ.get("HF_TOKEN")
if hf_token is None:
    raise ValueError("Please set HF_TOKEN in your environment.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and preprocess
model, preprocess = create_model_from_pretrained(
    "conch_ViT-B-16",
    "hf_hub:MahmoodLab/conch",
    hf_auth_token=hf_token
)
model = model.to(device)
model.eval()

# === Encode Image ===
image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.inference_mode():
    image_features = model.encode_image(image_tensor, proj_contrast=True, normalize=True)

print("✅ Image feature shape:", image_features.shape)

# === Encode Text ===
text_prompt = ["A WSI of clear cell renal cell carcinoma"]
tokenized = tokenize(text_prompt).to(device)

with torch.no_grad():
    text_features = model.encode_text(tokenized)

print("✅ Text feature shape:", text_features.shape)
print("First 5 values of text embedding:", text_features[0][:5])
