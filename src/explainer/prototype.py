import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from torch.nn import MultiheadAttention


# ========== Truncated Normal Initialization ==========
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ========== ViLa Prototype Module ==========
class ViLaPrototypeTrainer(nn.Module):
    """
    Learns visual prototypes using cross-attention and attention-based aggregation.
    Optionally supports class-specific prototypes.
    """

    def __init__(self, input_size, hidden_size, prototype_number, num_classes=None):
        """
        Args:
            input_size (int): Dimensionality of input features (e.g., 1024 or 512).
            hidden_size (int): Size of hidden projection in attention.
            prototype_number (int): Number of learnable prototype vectors (per class if num_classes is given).
            num_classes (int, optional): If set, each class gets its own prototype set.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prototype_number = prototype_number
        self.num_classes = num_classes

        total_prototypes = prototype_number * num_classes if num_classes else prototype_number

        # Attention projection heads
        self.attention_V = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_size, 1)

        # Learnable prototype tokens
        self.learnable_image_center = nn.Parameter(torch.Tensor(total_prototypes, 1, input_size))
        trunc_normal_(self.learnable_image_center, std=0.02)

        self.norm = nn.LayerNorm(input_size)

        # Cross-attention module
        self.cross_attention = MultiheadAttention(embed_dim=input_size, num_heads=1, batch_first=False)

    def forward(self, x):
        """
        Args:
            x (Tensor): Bag features of shape (N, D), where N is number of patches.

        Returns:
            image_features (Tensor): Aggregated feature.
            prototypes (Tensor): All learned prototype vectors.
        """
        M = x.float()
        M = M.unsqueeze(1) if M.ndim == 2 else M  # (N, 1, D)

        # Cross-attention between prototypes and image patches
        compents, _ = self.cross_attention(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)

        H = compents.squeeze(1)  # (P, D)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)  # (P, 1)
        A = F.softmax(A.T, dim=1)              # (1, P)
        image_features = torch.mm(A, H)        # (1, D)

        return image_features, self.learnable_image_center.detach()
