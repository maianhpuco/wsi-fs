import os
from transformers import AutoProcessor, AutoModelForVision2Seq
import sys
import os
sys.path.append("src/externals/quilt-llava") 

def download_quilt_llava(destination_path):
    """
    Download the Quilt-LLaVA model and processor to a specified local directory.

    Args:
        destination_path (str): Local path where the model and processor will be saved.
    """
    model_id = "wisdomik/Quilt-Llava-v1.5-7b"

    # Make sure the directory exists
    os.makedirs(destination_path, exist_ok=True)

    print(f"Downloading processor to {destination_path}...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=destination_path)

    print(f"Downloading model to {destination_path}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        cache_dir=destination_path
    )

    print("Download completed successfully.")

if __name__ == "__main__":
    target_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b"
    download_quilt_llava(target_dir)
