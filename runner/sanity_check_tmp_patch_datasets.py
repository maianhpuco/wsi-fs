import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(PROJECT_ROOT)
from src.datasets.camelyon16 import TmpPatchesDataset 


# Define any transforms you want
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize patches to a common size
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
])

root_patch_dir = '/project/hnguyen2/mvu9/miccai_25/camelyon16/patch_paths'
subdirs = sorted([d for d in os.listdir(root_patch_dir) if os.path.isdir(os.path.join(root_patch_dir, d))])

print(f"Found {len(subdirs)} folders in patch_dir")

for subdir in subdirs:
    subdir_path = os.path.join(root_patch_dir, subdir)
    dataset = TmpPatchesDataset(patch_dir=subdir_path, transform=transform)

    if len(dataset) == 0:
        print(f"Skipping {subdir}: no .png patches found.")
        continue

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    print(f"\n=== DataLoader for folder: {subdir} (total patches: {len(dataset)}) ===")
    for batch in loader:
        images = batch['image']
        patch_infos = batch['patch_info']
        print(f"  Batch size: {images.size()}")
        print(f"  First patch info: {patch_infos}")
        break  # Remove this if you want to process the entire folder