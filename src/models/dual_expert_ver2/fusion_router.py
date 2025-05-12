import torch
import torch.nn as nn

class FusionRouter(nn.Module):
    def __init__(self, beta=0.6):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
    
    def forward(self, patch_logits, prior_logits, clip_logits, cache_logits):
        fused_patch_logits = clip_logits + 10.0 * cache_logits
        agg_patch_logits = patch_logits.mean(dim=0)
        slide_logits = self.beta * agg_patch_logits + (1 - self.beta) * prior_logits
        return fused_patch_logits, slide_logits