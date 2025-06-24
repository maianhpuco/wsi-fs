import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
import sys
import os
sys.path.append("src/externals/quilt-llava") 

def run_quilt_llava(image_path, prompt="### Explain this pathology patches in detail, is there any abnormality?"):
    model_id = "wisdomik/Quilt-Llava-v1.5-7b"

    # Load model + processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.float16)

    # Generate
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return response

# Example usage
if __name__ == "__main__":
    img_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/png_patches/patch_256x256_5x/TCGA-UW-A7GY-11Z-00-DX1.7410A3EA-BFFD-4744-8DB2-66A409C0BFA9/30179_43105.png"
    print(run_quilt_llava(img_path))
