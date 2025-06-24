from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os

# === Configuration ===
model_id = "wisdomik/Quilt-Llava-v1.5-7b"
image_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/png_patches/patch_256x256_5x/TCGA-UW-A7GY-11Z-00-DX1.7410A3EA-BFFD-4744-8DB2-66A409C0BFA9/30179_44129.png"  # <-- Update this path
prompt = "### Explain this pathology slide"
max_new_tokens = 200

# === Load Model and Processor ===
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Load and preprocess image ===
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at: {image_path}")

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    raise ValueError(f"Failed to open image: {e}")

# === Prepare input ===
inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.float16)

# === Generate response ===
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(output, skip_special_tokens=True)[0]

# === Display output ===
print("\n Model Response:")
print(response)
