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
    # Path to the local directory containing the full Quilt-LLaVA model
    model_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b/"

    # Load model, tokenizer, and image processor
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

    # Verify model and vision tower
    if not hasattr(model, 'vision_tower') or model.vision_tower is None:
        print("Error: Model does not have a vision tower or vision tower is None.")
        return None

    # Load and preprocess the input image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    # Debug: Check image tensor shape
    print(f"Image tensor shape: {image_tensor.shape}")

    # Tokenize the prompt with image tokens
    try:
        input_ids = tokenizer_image_token(prompt, tokenizer, model.config, return_tensors="pt").to(model.device)
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        return None

    # Debug: Check input_ids shape
    print(f"Input IDs shape: {input_ids.shape}")

    # Ensure input_ids is 2D
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension if missing
    print(f"Adjusted Input IDs shape: {input_ids.shape}")

    # Prepare attention mask
    attention_mask = torch.ones_like(input_ids, device=model.device)

    # Prepare inputs for the model
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": image_tensor,
    }

    # Debug: Print input details
    print(f"Inputs keys: {inputs.keys()}")
    print(f"Input IDs: {input_ids}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Images shape: {image_tensor.shape}")

    # Generate a response
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
        print(result)
    else:
        print("Inference failed.")