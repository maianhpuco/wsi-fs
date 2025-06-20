import os
import sys
import torch
import numpy as np
from PIL import Image

# === Setup path ===
conch_path = os.path.abspath("src/externals/CONCH")  # adjust if needed
sys.path.append(conch_path)

from conch.open_clip_custom import create_model_from_pretrained, tokenize

# === HF cache and device ===
os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/cache_folder/.cache/huggingface'
hf_token = os.environ.get("HF_TOKEN")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load CONCH ===
model, preprocess = create_model_from_pretrained(
    model_cfg="conch_ViT-B-16",
    checkpoint_path="hf_hub:MahmoodLab/conch",
    device=device,
    hf_auth_token=hf_token
)
model = model.to(device)
model.eval()

# === Create dummy image (224x224 RGB) ===
random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(random_array)
image_tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# === Create dummy text prompts ===
prompts = ["A WSI of tumor", "A WSI of normal tissue"]
tokens = tokenize(prompts).to(device)

# === Encode image and text ===
with torch.no_grad():
    image_feat = model.encode_image(image_tensor, proj_contrast=True, normalize=True)  # [1, D]
    text_feat = model.encode_text(tokens)  # [2, D]

    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    similarity = image_feat @ text_feat.T  # [1, 2]
    pred = similarity.argmax(dim=1).item()

# === Output ===
print("Dummy image vs prompts similarity:", similarity.cpu().numpy())
print("Predicted class index:", pred)
print("Predicted prompt:", prompts[pred])
