import os
from huggingface_hub import snapshot_download

# Make sure your Hugging Face token is exported as an environment variable
# Example before running: export HUGGINGFACE_HUB_TOKEN=your_token_here

def download_titan_model():
    try:
        local_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/TITAN"
        repo_id = "MahmoodLab/TITAN"
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

        if token is None:
            raise ValueError("HUGGINGFACE_HUB_TOKEN not found in environment variables.")

        snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"TITAN model successfully downloaded to {local_dir}")
    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_titan_model()
