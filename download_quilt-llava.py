import os
import sys
from PIL import Image
import torch

# Add Quilt-LLaVA to path
sys.path.append("src/externals/quilt-llava")

# Import Quilt-LLaVA-specific loader
from llava.model.builder import load_pretrained_model

def download_quilt_llava(destination_path):
    """
    Download the Quilt-LLaVA model and processor using its internal loading logic.
    """
    model_name = "llava"  # required by Quilt-LLaVA
    model_path = "wisdomik/Quilt-Llava-v1.5-7b"

    # Make sure destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    print(f"Downloading model and tokenizer to: {destination_path}")

    # Download and cache to specified location
    tokenizer, model, image_processor = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        cache_dir=destination_path
    )

    print("Quilt-LLaVA download complete.")

if __name__ == "__main__":
    target_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b"
    download_quilt_llava(target_dir)
