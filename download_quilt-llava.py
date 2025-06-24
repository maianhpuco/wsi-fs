import os
import sys
import torch

# Add Quilt-LLaVA submodule path
sys.path.append("src/externals/quilt-llava")

# Import loader
from llava.model.builder import load_pretrained_model

def download_quilt_llava(destination_path):
    """
    Download the Quilt-LLaVA model, tokenizer, and image processor to a local directory.
    """
    # Quilt-LLaVA HF model ID
    model_id = "wisdomik/Quilt-Llava-v1.5-7b"
    model_name = "llava"

    os.makedirs(destination_path, exist_ok=True)

    print(f"🚀 Downloading '{model_id}' to cache dir: {destination_path}")

    # Use builder with cache_dir
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device="cuda",  # or "cpu" if needed
        cache_dir=destination_path  # <-- Save everything here
    )

    print("\n✅ Download complete!")
    print(f"📦 Model saved in: {destination_path}")
    print(f"🧠 Context length: {context_len}")
    print(f"🧾 Tokenizer vocab size: {len(tokenizer)}")
    print(f"📸 Image processor: {image_processor.__class__.__name__}")

    # Optional: save dummy output to test it's working
    torch.save(model.state_dict(), os.path.join(destination_path, "check_loaded_model.pt"))
    print("💾 Dummy model checkpoint saved.")

if __name__ == "__main__":
    target_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b"
    download_quilt_llava(target_dir)
