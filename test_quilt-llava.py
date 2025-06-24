import torch
import sys
import os
sys.path.append("src/externals/quilt-llava")
 
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

def run_quilt_llava(image_path, prompt="### Explain this pathology patch, is there any abnormality?"):
    model_path = "wisdomik/Quilt-Llava-v1.5-7b"  # or local path

    # Load model, tokenizer, processor
    tokenizer, model, processor = load_pretrained_model(
        model_path, model_base=None, model_name="llava"
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], processor, model.config).to(model.device, torch.float16)

    # Prepend image token
    prompt = tokenizer_image_token(prompt, tokenizer, model.config)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.update({"images": image_tensor})

    output_ids = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    img_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/png_patches/patch_256x256_5x/TCGA-UW-A7GY-11Z-00-DX1.7410A3EA-BFFD-4744-8DB2-66A409C0BFA9/30179_43105.png"
    print(run_quilt_llava(img_path))
