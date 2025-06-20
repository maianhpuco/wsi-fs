import os
import sys
import torch
import numpy as np
from PIL import Image

# === Setup path ===
conch_path = os.path.abspath("src/externals/CONCH")  # adjust if needed
sys.path.append(conch_path)

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

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

# === Inputs ===
classes = ['invasive ductal carcinoma', 'invasive lobular carcinoma']
prompts = [f"an H&E image of {cls}" for cls in classes]

# === Load tokenizer and tokenize
tokenizer = get_tokenizer()
tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)

# === Encode
with torch.inference_mode():
    image_embedings = model.encode_image(image_tensor, proj_contrast=True, normalize=True)
    text_embedings = model.encode_text(tokenized_prompts, proj_contrast=True, normalize=True)
    sim_scores = (image_embedings @ text_embedings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

# === Output
print("Predicted class:", classes[sim_scores.argmax()])
print("Normalized similarity scores:", [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])]) 