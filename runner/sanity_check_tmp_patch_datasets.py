import os
from torch.utils.data import DataLoader

# Assuming TmpPatchesDataset and transform are already defined

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
        print(f"  First patch info: {patch_infos[0]}")
        break  # Remove this if you want to process the entire folder
