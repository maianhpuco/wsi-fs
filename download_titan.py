from huggingface_hub import snapshot_download

# Set target directory where the model will be saved
save_dir = "/project/hnguyen2/mvu9/pretrained_checkpoints/TITAN"

# Download TITAN model weights
snapshot_download(
    repo_id="MahmoodLab/TITAN",
    local_dir=save_dir,
    local_dir_use_symlinks=False  # Ensure files are actually copied
)

print(f"TITAN model downloaded to: {save_dir}")
