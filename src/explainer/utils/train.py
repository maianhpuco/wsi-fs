import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def train_prototype_module(model, dataset, out_path, epochs=5, lr=1e-4, batch_size=1, device='cuda'):
    """
    Trains the ViLaPrototypeTrainer and saves learned prototypes and selected image features.

    Args:
        model (nn.Module): ViLaPrototypeTrainer model.
        dataset (torch.utils.data.Dataset): Dataset returning (features, label, metadata).
        out_path (str): Folder to save outputs (prototypes and selection results).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (str): Device (e.g., 'cuda' or 'cpu').
    """
    model = model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(out_path, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for i, (features, label, meta) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            features = features.to(device)  # (N, D)

            # Forward
            image_feats, _ = model(features)

            # Optional: use contrastive or CE loss depending on setting
            # Dummy supervision here: you can attach contrastive / clustering / reconstruction loss
            loss = torch.norm(image_feats, p=2).mean()  # Placeholder

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # Save final prototypes
    torch.save(model.learnable_image_center.detach().cpu(), os.path.join(out_path, "prototypes.pt"))

    print("✅ Prototypes saved.")

