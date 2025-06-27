import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

# Assumed: Custom dataset + trained model from CONCH_Prototype_Model
# Imports for your data/model setup
from explainer_ver1 import CONCH_Prototype_Model
from utils.file_utils import load_pkl
from src.datasets.multiple_scales.tcga import return_splits_custom


def save_topk_patches(model, dataset, save_dir, k=20):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    all_features = []
    all_coords = []
    all_slide_ids = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting patch features"):
            x_s = batch['features_s'].squeeze(0).to(model.device)  # [N, D]
            coords_s = batch['coords_s'].squeeze(0).cpu().numpy()  # [N, 2]
            slide_id = batch['slide_id'][0]

            # Forward project features
            x_proj = model.forward_project(x_s)  # [N, D]
            x_proj = F.normalize(x_proj, dim=-1).cpu()

            all_features.append(x_proj)
            all_coords.append(coords_s)
            all_slide_ids.extend([slide_id] * len(x_proj))

    all_features = torch.cat(all_features, dim=0)  # [M, D]
    all_coords = np.concatenate(all_coords, axis=0)  # [M, 2]

    # Compute similarity
    prototypes = F.normalize(model.prototypes.detach().cpu(), dim=-1)  # [P, D]
    all_features = F.normalize(all_features, dim=-1)  # [M, D]

    sim_matrix = torch.matmul(prototypes, all_features.T)  # [P, M]
    topk_indices = torch.topk(sim_matrix, k=k, dim=1).indices  # [P, K]

    # Save top-k per prototype
    for proto_id in range(prototypes.size(0)):
        indices = topk_indices[proto_id].tolist()
        coords = all_coords[indices]
        slide_ids = [all_slide_ids[i] for i in indices]

        save_path = os.path.join(save_dir, f"prototype_{proto_id}_top{k}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'prototype_id': proto_id,
                'coords': coords,
                'slide_ids': slide_ids,
                'indices': indices,
            }, f)
        print(f"[✓] Saved top-{k} for prototype {proto_id} to {save_path}")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # Load dataset split
    dataset = return_splits_custom(
        train_csv_path=os.path.join(config['paths']['split_folder'], f"fold_{args.fold_id}/train.csv"),
        val_csv_path=os.path.join(config['paths']['split_folder'], f"fold_{args.fold_id}/val.csv"),
        test_csv_path=os.path.join(config['paths']['split_folder'], f"fold_{args.fold_id}/test.csv"),
        data_dir_s=config['paths']['data_folder_s'],
        data_dir_l=config['paths']['data_folder_l'],
        label_dict=config['label_dict'],
        seed=config['seed'],
        use_h5=True
    )[0]  # only use train set for prototype mining

    # Init model
    model = CONCH_Prototype_Model(config=ml_collections.ConfigDict(config), num_classes=config['n_classes'])
    model.load_state_dict(torch.load(config['paths']['checkpoint_path'], map_location=device))
    model = model.to(device)

    # Save top-k patches
    save_topk_patches(model, dataset, config['paths']['prototype_topk_dir'], k=args.k)

    print("✅ Top-k prototype patch extraction complete.")
