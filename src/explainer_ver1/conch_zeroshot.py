import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 

sys.path.append("src/externals/CONCH") 
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

class CONCH_ZeroShot_Model(nn.Module):
    def __init__(self, config, num_classes=None):
        """
        Args:
            config: configuration object with fields like .device, .weight_path
            num_classes: optional, used for logging or slicing prompts
        """
        super(CONCH_ZeroShot_Model, self).__init__()
        self.device = config.device
        self.num_classes = num_classes

        # Load CONCH base model (e.g., ViT-B/16 + projection)
        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )

        self.tokenizer = get_tokenizer()
        self.logit_scale = self.model.logit_scale
        self.visual = self.model.visual  # for patch features

    def encode_text(self, prompts):
        """
        Encode a list of text prompts.
        Returns:
            torch.Tensor: [N, D] normalized
        """
        tokenized = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokenized)
        return F.normalize(text_features, dim=-1)

    def encode_image(self, images):
        """
        Encode images directly.
        Args:
            images: [B, C, H, W]
        Returns:
            torch.Tensor: [B, D] normalized
        """
        image_features = self.model.encode_image(images)
        return F.normalize(image_features, dim=-1)

    def forward_project(self, patch_features):
        """
        Project patch-level features through visual projection.
        Args:
            patch_features: [N, D]
        Returns:
            torch.Tensor: [N, D] normalized
        """
        projected = self.visual.forward_project(patch_features)
        return F.normalize(projected, dim=-1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use encode_text(), encode_image(), or forward_project() methods directly.")
