import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime


def train_prototype_module(model, train_dataset, val_dataset, out_path, epochs=5, lr=1e-4, batch_size=1, device='cuda'):
    """
    Trains the ViLaPrototypeTrainer and saves learned prototypes and model.

    Args:
        model (nn.Module): ViLaPrototypeTrainer model.
        train_dataset (Dataset): Training dataset (Generic_MIL_Dataset).
        val_dataset (Dataset): Validation dataset (Generic_MIL_Dataset).
        out_path (str): Folder to save outputs (prototypes and model).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (str): Device (e.g., 'cuda' or 'cpu').
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.train()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create output directory
    os.makedirs(out_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels, *extra in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            features = features.to(device)  # (B, N, D)
            labels = labels.to(device)     # (B,)

            # Forward
            logits, image_feats, _ = model(features)
            if logits is None:
                raise ValueError("Model did not return logits. Ensure num_classes is set in ViLaPrototypeTrainer.")

            # Compute loss
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels, *extra in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                features = features.to(device)  # (B, N, D)
                labels = labels.to(device)     # (B,)

                logits, image_feats, _ = model(features)
                if logits is None:
                    raise ValueError("Model did not return logits during validation.")

                loss = criterion(logits, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save prototypes and model
    torch.save(model.learnable_image_center.detach().cpu(), os.path.join(out_path, f"prototypes_{timestamp}.pt"))
    torch.save(model.state_dict(), os.path.join(out_path, f"model_{timestamp}.pt"))

    print(f"✅ Prototypes and model saved to {out_path}")