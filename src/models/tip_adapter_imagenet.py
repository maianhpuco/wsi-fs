import os
import sys 
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(PROJECT_DIR) 
sys.path.append(os.path.join(PROJECT_DIR, 'src', 'includes', 'Tip-Adapter'))

import random
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.imagenet import ImageNet

import clip
from utils import cls_acc, build_cache_model, pre_load_features, clip_classifier, search_hp 

# --------------------------- Argument Parsing ---------------------------
def get_arguments():
    """Parse command-line arguments for Tip-Adapter configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args


# --------------------------- Tip-Adapter Functions ---------------------------
def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    """Run Tip-Adapter evaluation on test set."""
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)
    
    
# --------------------------- Tip-Adapter-F Functions ---------------------------
def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    """Run Tip-Adapter-F with fine-tuning on test set."""
    # Initialize learnable adapter
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    # Training loop
    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print(f'Train Epoch: {train_idx} / {cfg["train_epoch"]}')

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)
            acc = cls_acc(tip_logits, target)
            
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f'LR: {current_lr:.6f}, Acc: {correct_samples / all_samples:.4f} '
              f'({correct_samples}/{all_samples}), Loss: {sum(loss_list)/len(loss_list):.4f}')

        # Evaluation
        adapter.eval()
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity).exp()) @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print(f"**** Tip-Adapter-F's test accuracy: {acc:.2f}. ****\n")
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + f"/best_F_{cfg['shots']}shots.pt")
    
    # Load best model
    adapter.weight = torch.load(cfg['cache_dir'] + f"/best_F_{cfg['shots']}shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, "
          f"at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, adapter=adapter)

# --------------------------- Main Function ---------------------------
def main():
    """Main function to run Tip-Adapter and Tip-Adapter-F."""
    # Load configuration
    args = get_arguments()
    assert os.path.exists(args.config), "Configuration file does not exist"
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs:")
    print(cfg, "\n")

    # Initialize CLIP model
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Set random seeds
    random.seed(1)
    torch.manual_seed(1)
    
    # Prepare ImageNet dataset
    print("Preparing ImageNet dataset...")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)
    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Get textual features
    print("\nGetting textual features as CLIP's classifier...")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Build cache model
    print("\nConstructing cache model by few-shot visual features and labels...")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Load test features
    print("\nLoading visual features and labels from test set...")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # Run Tip-Adapter
    print("\n---------- Running Tip-Adapter ----------")
    run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

    # # Run Tip-Adapter-F
    # print("\n---------- Running Tip-Adapter-F ----------")
    # run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, 
    #                  clip_model, train_loader_F)

# --------------------------- Entry Point ---------------------------
if __name__ == '__main__':
    main()