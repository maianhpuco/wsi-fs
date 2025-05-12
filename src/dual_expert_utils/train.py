import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import json
from tqdm import tqdm
from models.dual_expert import DualExpert
from utils.patch_selection import select_patches
from utils.data_loader import get_data_loader
from utils.contrastive_loss import supervised_contrastive_loss

def train(config):
    # Load config
    device = torch.device(config['training']['device'])
    num_shots = config['data']['num_shots']
    num_wsis = config['data']['num_wsis']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    
    # Load class descriptions
    with open(config['data']['class_descriptions'], 'r') as f:
        class_descriptions = json.load(f)
    
    # Initialize model
    model = DualExpert(
        clip_model_name=config['model']['clip_model'],
        vqgan_codebook=torch.randn(config['model']['vqgan_codebook_size'], config['model']['feature_dim']),  # Placeholder
        prototypes_per_class=config['model']['prototypes_per_class'],
        n_classes=config['model']['n_classes'],
        feature_dim=config['model']['feature_dim'],
        lora_rank=config['model']['lora_rank'],
        class_descriptions=class_descriptions
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Data loaders (placeholder)
    train_loader = get_data_loader(config['data']['wsi_dir'], batch_size=batch_size)
    
    # K-shot patches (placeholder)
    k_shot_patches = torch.rand(num_shots * config['model']['n_classes'], 3, 224, 224)
    k_shot_labels = torch.randint(0, config['model']['n_classes'], (num_shots * config['model']['n_classes'],))
    
    # Build cache
    model.build_cache(k_shot_patches.to(device), k_shot_labels.to(device))
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for patches, slide_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            patches, slide_labels = patches.to(device), slide_labels.to(device)
            
            # Select patches
            selected_patches = select_patches(patches, model.clip, k_shot_patches, k_shot_labels, 
                                             M=config['data']['patches_per_wsi'])
            
            # Forward pass
            patch_logits, slide_logits, augmented_features = model(selected_patches, slide_labels)
            
            # Losses
            loss_ce = F.cross_entropy(slide_logits, slide_labels)
            loss_supcon = supervised_contrastive_loss(augmented_features, slide_labels)
            loss = loss_ce + 0.5 * loss_supcon
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "pavwa_ltcp_model.pth")

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train(config)