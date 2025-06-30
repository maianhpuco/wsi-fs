import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer


class GatedAttentionPooling(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, feats, scores):
        A_V = self.attention_V(feats)
        A_U = self.attention_U(feats)
        A = self.attention_weights(A_V * A_U).squeeze(-1) * scores
        A = F.softmax(A, dim=1)
        return torch.sum(A.unsqueeze(-1) * feats, dim=1)

# Define a module to compute attention scores between patch features and class descriptions
class DescriptionHead(nn.Module): 
    def __init__(self, desc_feats): 
        super().__init__()
        # Store description features as non-trainable parameters
        self.desc_feats = nn.Parameter(desc_feats.clone(), requires_grad=False)  # [num_desc, D]
        # Scaling factor for dot-product attention (1/sqrt(feature_dim))
        self.scale = desc_feats.shape[1] ** -0.5

    def forward(self, patch_feats):
        """
        Args:
            patch_feats: [B, N, D] - batch of N patch features with dimension D
        Returns:
            scores: [B, N, num_desc] - attention scores for each patch and description
        """
        # Compute similarity between patch features and description features
        # Matrix multiplication: [B, N, D] x [D, num_desc] → [B, N, num_desc]
        sim = torch.matmul(patch_feats, self.desc_feats.T) * self.scale
        # Apply softmax to get attention weights over descriptions
        return F.softmax(sim, dim=-1)


class Ver2c(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        # Store device and number of classes from config
        self.device = config.device
        self.num_classes = num_classes
        # Store text prompts (class descriptions) from config
        self.text_prompts = dict(config.text_prompts)
        # Initialize tokenizer for text processing
        self.tokenizer = get_tokenizer()

        # Validate text prompts: ensure each class has a list of descriptions
        for class_name, desc_list in self.text_prompts.items():
            assert isinstance(desc_list, list), f"Descriptions for '{class_name}' must be a list."
            # Ensure each description fits within tokenizer's length limit
            for desc in desc_list:
                assert len(self.tokenizer.encode(desc)) <= 77, f"Prompt too long: '{desc}'"

        # Load pretrained CONCH model
        self.model, _ = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )

        # Extract visual encoder and logit scale from pretrained model
        self.visual = self.model.visual
        self.logit_scale = self.model.logit_scale
        # Define cross-entropy loss for classification
        self.loss_ce = nn.CrossEntropyLoss()

        # Set feature and hidden dimensions from config (default to 512 and 256)
        feat_dim = getattr(config, "input_size", 512)
        hidden_dim = getattr(config, "hidden_size", 256)
        # Linear projection layer for visual features
        self.visual_proj = nn.Linear(feat_dim, feat_dim)
        # Create one attention pooling module per class
        self.attn_pooling = nn.ModuleList([
            GatedAttentionPooling(feat_dim, hidden_dim) for _ in range(self.num_classes)
        ])
        # Linear layer to combine class-specific features into a single slide feature
        self.class_combiner = nn.Linear(feat_dim * num_classes, feat_dim)

        # Initialize text features for descriptions and class-to-description mapping
        self.desc_text_features, self.class_to_desc_idx = self.init_text_features()
        self.desc_text_features = self.desc_text_features.detach()

        # Create description heads for each class
        self.desc_heads = nn.ModuleList([
            DescriptionHead(self.desc_text_features[start:end])
            for class_id, (start, end) in self.class_to_desc_idx.items()
        ])

        # Aggregate class-level text features for final classification
        self.text_features = self.aggregate_class_features().detach()

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        text_features = self.model.encode_text(tokenized["input_ids"])

        if text_features.shape[-1] != self.visual_proj.in_features:
            projection = getattr(self.model, "text_projection", None)
            if projection is None and hasattr(self.model, "text"):
                projection = getattr(self.model.text, "text_projection", None)
            if projection is not None:
                text_features = text_features @ projection

        return F.normalize(text_features, dim=-1)

    def init_text_features(self):
        # Initialize lists to store description features and class-to-description mappings
        desc_feats, class_to_desc = [], {}
        start_idx = 0
        # Process each class and its descriptions
        for class_id, (class_name, desc_list) in enumerate(self.text_prompts.items()):
            # Encode descriptions into features
            features = self.encode_text(desc_list)
            end_idx = start_idx + features.size(0)
            # Store features and update index mapping
            desc_feats.append(features)
            class_to_desc[class_id] = (start_idx, end_idx)
            start_idx = end_idx
        # Concatenate all description features
        return torch.cat(desc_feats, dim=0), class_to_desc

    def aggregate_class_features(self):
        # Initialize tensor to store class-level features
        C, D = self.num_classes, self.desc_text_features.size(1)
        class_feats = torch.zeros(C, D, device=self.device)
        # Compute mean feature for each class's descriptions
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_feats[class_id] = self.desc_text_features[start:end].mean(dim=0)
        # Normalize class features
        return F.normalize(class_feats, dim=-1)

    def compute_patch_scores_per_class(self, patch_feats):
        # Compute attention scores for each class's descriptions
        class_scores = []
        for head in self.desc_heads:
            # Get attention scores for each description [B, N, num_desc]
            scores = head(patch_feats)
            # Average scores across descriptions for this class [B, N, 1]
            class_scores.append(scores.mean(dim=-1, keepdim=True))
        # Concatenate scores for all classes [B, N, C]
        return torch.cat(class_scores, dim=-1)
    
    def forward(self, x_s, coord_s, x_l, coord_l, label=None):
        # Ensure inputs have batch dimension
        if x_s.ndim == 2:
            x_s = x_s.unsqueeze(0)
            x_l = x_l.unsqueeze(0)
            coord_s = coord_s.unsqueeze(0)
            coord_l = coord_l.unsqueeze(0)

        # Get input dimensions
        B, N, D = x_s.shape

        # Project and normalize patch features for small and large patches
        x_s_proj = F.normalize(self.visual_proj(F.normalize(x_s, dim=-1)), dim=-1)
        x_l_proj = F.normalize(self.visual_proj(F.normalize(x_l, dim=-1)), dim=-1)

        # Compute attention scores for each class
        class_scores_s = self.compute_patch_scores_per_class(x_s_proj)  # [B, N, C]
        class_scores_l = self.compute_patch_scores_per_class(x_l_proj)  # [B, N, C]

        # Compute class-specific slide features
        class_feats_s = []
        class_feats_l = []
        for c in range(self.num_classes):
            # Pool patch features using class-specific attention scores
            feat_s = self.attn_pooling[c](x_s_proj, class_scores_s[:, :, c])  # [B, D]
            feat_l = self.attn_pooling[c](x_l_proj, class_scores_l[:, :, c])  # [B, D]
            class_feats_s.append(feat_s)
            class_feats_l.append(feat_l)

        # Stack class-specific features into [B, C, D]
        class_feats_s = torch.stack(class_feats_s, dim=1)
        class_feats_l = torch.stack(class_feats_l, dim=1)

        # Average small and large patch features
        class_feats = (class_feats_s + class_feats_l) / 2  # [B, C, D]

        # Combine class-specific features into a single slide feature
        slide_feat = self.class_combiner(class_feats.view(B, -1))  # [B, D]
        slide_feat = F.normalize(slide_feat, dim=-1)

        # Compute logits by comparing slide feature with class text features
        logits = slide_feat @ self.text_features.T

        # Compute loss if labels are provided
        loss = self.loss_ce(logits, label) if label is not None else None

        # Compute probabilities and predicted class
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)

        return Y_prob, Y_hat, loss