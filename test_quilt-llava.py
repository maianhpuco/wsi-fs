import os
import sys
import torch
from PIL import Image

# Add the Quilt-LLaVA submodule to the Python path
sys.path.append("src/externals/quilt-llava")

# Import required functions and modules from Quilt-LLaVA
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

def run_quilt_llava(image_path, prompt="### Explain this pathology patch, is there any abnormality?"):
    """
    Run inference using the locally saved Quilt-LLaVA model on a given image.

    Args:
        image_path (str): Path to the input image (e.g., pathology patch).
        prompt (str): Text prompt to guide the model's response.

    Returns:
        str: Generated explanation text from the model.
    """
    model_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b"

    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava",
            load_8bit=False,
            load_4bit=False,
            device_map="auto",
            device="cuda"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Ensure vision tower is loaded
    try:
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, "load_model") and not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=model.device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
    except Exception as e:
        print(f"Error loading vision tower: {e}")
        return None

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    # Format the prompt
    try:
        formatted_prompt = tokenizer_image_token(prompt, tokenizer, model.config)
        if isinstance(formatted_prompt, list):
            formatted_prompt = "".join(formatted_prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        inputs.update({"images": image_tensor})
    except Exception as e:
        print(f"Error preparing inputs: {e}")
        return None

    # Generate prediction
    try:
        output_ids = model.generate(**inputs, max_new_tokens=200)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

if __name__ == "__main__":
    img_path = "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/kich/png_patches/patch_256x256_5x/TCGA-UW-A7GY-11Z-00-DX1.7410A3EA-BFFD-4744-8DB2-66A409C0BFA9/30179_43105.png"
    result = run_quilt_llava(img_path)
    if result:
        print("Generated explanation:")
        print(result)
    else:
        print("Inference failed.")
