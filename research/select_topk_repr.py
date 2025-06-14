import torch.nn.functional as F

def select_topk_representative_patches(features, prototypes, k=5):
    """
    Selects top-k patches most similar to the learned prototypes.

    Args:
        features (Tensor): Shape (N, D), patch features from a WSI.
        prototypes (Tensor): Shape (P, D), learned prototype embeddings.
        k (int): Number of patches to select.

    Returns:
        topk_indices (Tensor): Indices of top-k representative patches.
    """
    if prototypes.dim() == 3:
        prototypes = prototypes.squeeze(1)  # (P, D)

    # Cosine similarity between patches and all prototypes
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    similarity = features @ prototypes.T  # (N, P)

    max_sim, _ = similarity.max(dim=1)    # Best matching prototype for each patch
    topk_indices = torch.topk(max_sim, k=k).indices

    return topk_indices


# After training is done
prototypes = torch.load("prototypes.pt")  # shape [P, 1, D]
for features, label, meta in dataloader:
    topk_idx = select_topk_representative_patches(features.squeeze(0), prototypes, k=5)
    selected_coords = meta['coords'][topk_idx]  # assuming dataset returns patch coords
    # → Save selected_coords or visualize
