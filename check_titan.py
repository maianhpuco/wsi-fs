from transformers import AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(
    "MahmoodLab/TITAN",
    trust_remote_code=True
).to(device)

# Load example features from HuggingFace Hub
from huggingface_hub import hf_hub_download
import h5py

demo_h5_path = hf_hub_download(
    "MahmoodLab/TITAN",
    filename="TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5",
)
file = h5py.File(demo_h5_path, 'r')
features = torch.from_numpy(file['features'][:]).to(device)
coords = torch.from_numpy(file['coords'][:]).to(device)
patch_size_lv0 = file['coords'].attrs['patch_size_level0']

with torch.autocast('cuda', torch.float16), torch.inference_mode():
    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
    report = model.generate_report(slide_embedding)

print("📝 Report:\n", report)
