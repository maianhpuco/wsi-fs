import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

# Define a gated attention pooling module to aggregate patch features
class GatedAttentionPooling(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        # Linear layer followed by Tanh for attention value transformation
        self.attention_V = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        # Linear layer followed by Sigmoid for attention gate transformation
        self.attention_U = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        # Final linear layer to compute attention weights
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, feats, scores):
        # Compute attention values: transform features to hidden_dim and apply Tanh
        A_V = self.attention_V(feats)
        # Compute attention gates: transform features to hidden_dim and apply Sigmoid
        A_U = self.attention_U(feats)
        # Combine value and gate, then compute attention weights, modulated by input scores
        A = self.attention_weights(A_V * A_U).squeeze(-1) * scores
        # Apply softmax to normalize attention weights across patches
        A = F.softmax(A, dim=1)
        # Compute weighted sum of features using attention weights
        return torch.sum(A.unsqueeze(-1) * feats, dim=1)
    
class DescriptionAttentionPooling(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

    def forward(self, desc_attn_scores):
        """
        Args:
            desc_attn_scores: [B, N, num_desc]
            patch_feats: [B, N, D]
        Returns:
            slide_desc_scores: [B, num_desc] - average concept score per slide
        """
        # Average over patches for each concept
        return desc_attn_scores.mean(dim=1)  # [B, num_desc] 

# Define a module to compute attention scores between patch features and class descriptions
class DescriptionHead(nn.Module): 
    def __init__(self, desc_feats): 
        super().__init__()
        # Store description features as non-trainable parameters
        self.desc_feats = nn.Parameter(desc_feats.clone(), requires_grad=False)  # [num_desc, D]
        # Scaling factor for dot-product attention (1/sqrt(feature_dim))
        self.scale = desc_feats.shape[1] ** -0.5
        # Linear layers for query and key projections
        self.query_proj = nn.Linear(desc_feats.shape[1], desc_feats.shape[1])
        self.key_proj = nn.Linear(desc_feats.shape[1], desc_feats.shape[1])

    def forward(self, patch_feats):
        """
        Args:
            patch_feats: [B, N, D] - batch of N patch features with dimension D
        Returns:
            scores: [B, N, num_desc] - attention scores for each patch and description
        """
        # Project patch features to query space: [B, N, D]
        queries = self.query_proj(patch_feats)
        # print(f"queries shape: {queries.shape}")  # Debugging output
        # Project description features to key space: [num_desc, D]
        keys = self.key_proj(self.desc_feats)
        # Compute attention scores: [B, N, D] x [D, num_desc] -> [B, N, num_desc]
        attn_scores = torch.matmul(queries, keys.T) * self.scale
        # Apply softmax to get attention weights over descriptions
        attn_weights = F.softmax(attn_scores, dim=-1)
        print(f"attn_weights shape: {attn_weights}")  # Debugging output
        return attn_weights

# Define the main model class
class Ver2f(nn.Module):
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
        # Attention pooling for description scores 
        
        self.desc_pooling = nn.ModuleList([
            DescriptionAttentionPooling(hidden_dim) for _ in range(self.num_classes)
        ])

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

        # Hyperparameters for submodular loss
        self.alpha = 0.5  # Weight for discriminability
        self.beta = 0.5   # Weight for coverage

    def encode_text(self, prompts):
        # Tokenize input prompts with padding
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        # Move tokenized inputs to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        # Encode text using the pretrained model's text encoder
        text_features = self.model.encode_text(tokenized["input_ids"])

        # Project text features to match visual feature dimension if needed
        if text_features.shape[-1] != self.visual_proj.in_features:
            projection = getattr(self.model, "text_projection", None)
            if projection is None and hasattr(self.model, "text"):
                projection = getattr(self.model.text, "text_projection", None)
            if projection is not None:
                text_features = text_features @ projection

        # Normalize text features
        return F.normalize(text_features, dim=-1)

    def init_text_features(self):
        # Initialize lists to store description features and class-to-description mapping
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
        slide_desc_scores = []
        for head, desc_pool in zip(self.desc_heads, self.desc_pooling):
            # Get attention scores for each description [B, N, num_desc]
            scores = head(patch_feats)
            print(f"scores : {scores}")  # Debugging output
            # Average scores across descriptions for this class [B, N, 1]
            class_scores.append(scores.mean(dim=-1, keepdim=True))
            # Compute slide-level scores for each description using attention pooling [B, num_desc]
            slide_scores = desc_pool(scores)
            slide_desc_scores.append(slide_scores)
        # Concatenate scores for all classes [B, N, C]
        class_scores = torch.cat(class_scores, dim=-1)
        # Concatenate slide-level description scores [B, num_desc * C]
        slide_desc_scores = torch.cat(slide_desc_scores, dim=1)
        # print(f"slide_desc_scores shape: {slide_desc_scores}")  # Debugging output
        return class_scores, slide_desc_scores

    def get_concept_scores(self, slide_desc_scores):
        """
        Args:
            slide_desc_scores: [B, num_desc * C] - slide-level description scores
        Returns:
            concept_scores: List of B dictionaries, each mapping class names to lists of
                           (description, score) tuples
        """
        # Initialize output list for batch
        concept_scores = []
        # Process each slide in the batch
        for b in range(slide_desc_scores.shape[0]):
            slide_dict = {}
            # Split scores by class using class_to_desc_idx
            for class_id, (class_name, desc_list) in enumerate(self.text_prompts.items()):
                start_idx, end_idx = self.class_to_desc_idx[class_id]
                # Extract scores for this class's descriptions
                class_scores = slide_desc_scores[b, start_idx:end_idx].tolist()
                # Pair each score with its corresponding description
                slide_dict[class_name] = list(zip(desc_list, class_scores))
            concept_scores.append(slide_dict)
        return concept_scores

    def compute_submodular_loss(self, slide_desc_scores):
        """
        Compute the submodular loss based on slide-level description scores.
        Args:
            slide_desc_scores: [B, num_desc * C] - slide-level description scores
        Returns:
            loss: Submodular loss value
        """
        B, total_desc = slide_desc_scores.shape
        num_classes = len(self.class_to_desc_idx)
        desc_per_class = total_desc // num_classes

        # Initialize total loss
        total_loss = 0.0

        for b in range(B):
            # Split scores by class
            class_scores = []
            for class_id in range(num_classes):
                start_idx, end_idx = self.class_to_desc_idx[class_id]
                class_scores.append(slide_desc_scores[b, start_idx:end_idx])

            # Compute discriminability score D(c) for each concept
            disc_loss = 0.0
            for c_idx in range(total_desc):
                class_idx = c_idx // desc_per_class
                c_scores = slide_desc_scores[b]  # [num_desc * C]
                sim_y_c = c_scores[c_idx]  # Use the score as Sim(y,c) approximation
                sim_y_c_norm = sim_y_c / (torch.sum(c_scores) + 1e-10)  # Normalize across all scores
                D_c = -torch.sum(sim_y_c_norm * torch.log(sim_y_c_norm + 1e-10))
                disc_loss += D_c
            disc_loss = disc_loss / total_desc  # Average over concepts

            # Compute coverage term using intra-concept similarity
            cov_loss = 0.0
            for c1_idx in range(total_desc):
                for c2_idx in range(total_desc):
                    if c1_idx != c2_idx:
                        # Approximate phi(c1, c2) using cosine similarity of description features
                        c1_feat = self.desc_text_features[c1_idx].unsqueeze(0)
                        c2_feat = self.desc_text_features[c2_idx].unsqueeze(0)
                        phi_c1_c2 = F.cosine_similarity(c1_feat, c2_feat)  # Keep as tensor
                        cov_loss += torch.relu(phi_c1_c2)  # Use ReLU for max(0, phi)
            cov_loss = cov_loss / (total_desc * (total_desc - 1))  # Normalize

            # Combine discriminability and coverage with hyperparameters
            submod_loss = self.alpha * disc_loss + self.beta * cov_loss
            total_loss += submod_loss

        return total_loss / B  # Average over batch
    
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

        # Compute attention scores for each class and slide-level description scores
        class_scores_s, slide_desc_scores_s = self.compute_patch_scores_per_class(x_s_proj)  # [B, N, C], [B, num_desc * C]
        class_scores_l, slide_desc_scores_l = self.compute_patch_scores_per_class(x_l_proj)  # [B, N, C], [B, num_desc * C]

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

        # Compute cross-entropy loss if labels are provided
        ce_loss = self.loss_ce(logits, label) if label is not None else None

        # Compute submodular loss
        submod_loss = self.compute_submodular_loss((slide_desc_scores_s + slide_desc_scores_l) / 2) if label is not None else 0.0

        # Combine losses (only during training)
        loss = ce_loss + submod_loss if ce_loss is not None else None

        # Compute probabilities and predicted class
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)

        # Average slide-level description scores from small and large patches
        slide_desc_scores = (slide_desc_scores_s + slide_desc_scores_l) / 2

        # Return slide-level description scores alongside existing outputs
        return Y_prob, Y_hat, loss, slide_desc_scores