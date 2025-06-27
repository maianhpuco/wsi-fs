import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

class CONCH_Prototype_Model(nn.Module):
    def __init__(self, config, num_classes):
        super(CONCH_Prototype_Model, self).__init__()
        self.device = config.device
        self.num_classes = num_classes
        self.text_prompt = config.text_prompt

        assert isinstance(self.text_prompt, list) and len(self.text_prompt) == 2 * self.num_classes, \
            f"Expected 2 * num_classes text prompts, got {len(self.text_prompt)}"

        # Load CONCH encoder
        self.model, _ = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )

        self.tokenizer = get_tokenizer()
        self.visual = self.model.visual
        self.logit_scale = self.model.logit_scale
        self.loss_ce = nn.CrossEntropyLoss()

        self.hidden_size = config.hidden_size
        self.num_prototypes = config.num_prototypes

        # Trainable prototype vectors (shared across classes)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.hidden_size))
        nn.init.xavier_uniform_(self.prototypes)

        # Text embeddings (frozen)
        self.text_features_low, self.text_features_high = self.init_text_features()

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(tokenized)
        return F.normalize(text_features, dim=-1)

    def forward_project(self, patch_features):
        projected = self.visual.forward_project(patch_features)
        return F.normalize(projected, dim=-1)

    def init_text_features(self):
        low_text = self.text_prompt[:self.num_classes]
        high_text = self.text_prompt[self.num_classes:]
        text_features_low = self.encode_text(low_text)
        text_features_high = self.encode_text(high_text)
        return text_features_low, text_features_high

    def attention_pool(self, patch_features):
        """
        Args:
            patch_features: [B, N, D]
        Returns:
            pooled_features: [B, D] (via prototype attention)
        """
        B, N, D = patch_features.size()
        prototypes = F.normalize(self.prototypes, dim=-1)  # [P, D]

        # Attention scores: [B, P, N]
        attn_scores = torch.einsum('bnd,pd->bpn', patch_features, prototypes) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, P, N]

        # Attend over patch features: [B, P, D]
        attended = torch.einsum('bpn,bnd->bpd', attn_weights, patch_features)

        # Aggregate across prototypes
        pooled_features = attended.mean(dim=1)  # [B, D]
        return F.normalize(pooled_features, dim=-1)

    def forward(self, x_s, coord_s, x_l, coord_l, label):
        """
        Args:
            x_s: [B, N, D] low-res features
            x_l: [B, N, D] high-res features
            label: [B] class label
        Returns:
            Y_prob: softmax output [B, C]
            Y_hat: predicted class index [B]
            loss: CE loss
        """
        B = x_s.size(0)

        # Project each patch feature
        x_s_proj = self.forward_project(x_s.view(-1, x_s.size(-1))).view(B, -1, self.hidden_size)
        x_l_proj = self.forward_project(x_l.view(-1, x_l.size(-1))).view(B, -1, self.hidden_size)

        # Use prototype attention to pool image features
        image_features_low = self.attention_pool(x_s_proj)   # [B, D]
        image_features_high = self.attention_pool(x_l_proj)  # [B, D]

        # Compute logits with text features
        logits_low = image_features_low @ self.text_features_low.T.cuda()     # [B, C]
        logits_high = image_features_high @ self.text_features_high.T.cuda()  # [B, C]
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)

        return Y_prob, Y_hat, loss
