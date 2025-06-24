import os
import sys
import torch

# Add Quilt-LLaVA submodule path
sys.path.append("src/externals/quilt-llava")

# Import loader from submodule
from llava.model.builder import load_pretrained_model

def download_quilt_llava(destination_path):
    """
    Download the Quilt-LLaVA model, tokenizer, and image processor to a local directory.
    """
    model_id = "wisdomik/Quilt-Llava-v1.5-7b"
    model_name = "llava"

    os.makedirs(destination_path, exist_ok=True)

    print(f"Downloading '{model_id}' into Transformers cache (default location).")

    # NOTE: load_pretrained_model does not support cache_dir explicitly
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda"
    )

    print("Download complete.")
    print(f"Model context length: {context_len}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Image processor: {image_processor.__class__.__name__}")

    # Optional: Save dummy state dict to confirm it's loaded
    torch.save(model.state_dict(), os.path.join(destination_path, "check_loaded_model.pt"))
    print("Saved model checkpoint (dummy) to verify model loading.")

if __name__ == "__main__":
    target_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b"
    download_quilt_llava(target_dir)
