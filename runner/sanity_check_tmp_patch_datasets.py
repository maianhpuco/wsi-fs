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

# Initialize dataset
patch_dir = '/project/hnguyen2/mvu9/miccai_25/camelyon16/patch_paths'  # Should match what's used inside TmpPatchesDataset
dataset = TmpPatchesDataset(patch_dir=patch_dir, transform=transform)

# Wrap with DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Iterate through a few batches
for batch in loader:
    images = batch['image']
    patch_infos = batch['patch_info']

    print(f"Batch size: {images.size()}")
    print(f"Patch info (first item): {patch_infos[0]}")
    break  # Remove this to loop through the entire dataset
