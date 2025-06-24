import os
from huggingface_hub import snapshot_download

def download_quilt_llava(local_dir):
    """
    Download the full Quilt-LLaVA model repository from Hugging Face
    to a specified local directory using snapshot_download.

    Args:
        local_dir (str): Target directory to store the full model.
    """
    print(f"Downloading Quilt-LLaVA to: {local_dir}")

    snapshot_download(
        repo_id="wisdomik/Quilt-Llava-v1.5-7b",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print("✅ Download complete.")
    print(f"Model files are saved at: {local_dir}")

if __name__ == "__main__":
    target_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/Quilt-Llava-v1.5-7b"
    os.makedirs(target_dir, exist_ok=True)
    download_quilt_llava(target_dir)
