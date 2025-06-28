import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer


class CONCH_ZeroShot_Model_TopjPooling_MoreText(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.device = config.device
        self.num_classes = num_classes
        self.text_prompts = dict(config.text_prompts)  # {class_name: [desc1, desc2, ...]}
        self.topj = getattr(config, "topj", 100)

        # Validate text prompts
        assert isinstance(self.text_prompts, dict), "text_prompts must be a dictionary"
        for class_name, prompts in self.text_prompts.items():
            assert isinstance(prompts, list), f"Expected list for {class_name}, got {type(prompts)}"
            for prompt in prompts:
                assert len(self.tokenizer.encode(prompt)) <= 77, f"Prompt exceeds 77 tokens: {prompt}"

        # Load pretrained CONCH model
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

        # Encode all textual descriptions and track index mapping
        self.desc_text_features, self.class_to_desc_idx = self.init_text_features()

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        text_features = self.model.encode_text(tokenized["input_ids"])

        # Apply text projection if not already projected
        if text_features.shape[-1] == 768:
            if hasattr(self.model, "text") and hasattr(self.model.text, "text_projection"):
                text_features = text_features @ self.model.text.text_projection
            elif hasattr(self.model, "text_projection"):
                text_features = text_features @ self.model.text_projection

        return F.normalize(text_features, dim=-1)

    def init_text_features(self):
        desc_features = []
        class_to_desc_idx = {}
        start_idx = 0

        for class_id, (class_name, desc_list) in enumerate(self.text_prompts.items()):
            features = self.encode_text(desc_list)  # [#desc, D]
            end_idx = start_idx + features.size(0)
            desc_features.append(features)
            class_to_desc_idx[class_id] = (start_idx, end_idx)
            start_idx = end_idx

        return torch.cat(desc_features, dim=0), class_to_desc_idx  # [T, D], {class_id: (start, end)}

    def forward(self, x_s, coord_s, x_l, coord_l, label=None, topj=100):
        """
        Args:
            x_s: low-res patch features [B, N, D]
            coord_s: low-res patch coordinates [B, N, 2]
            x_l: high-res patch features [B, N, D]
            coord_l: high-res patch coordinates [B, N, 2]
            label: ground-truth label [B] (optional)
            topj: number of top patches to consider for slide-level pooling
        Returns:
            Y_prob: softmax probabilities [B, C]
            Y_hat: predicted label [B]
            top_descriptions: list of top-matching descriptions for each sample
            loss: cross-entropy loss (if label provided)
        """
        B, N, D = x_s.shape

        # Normalize patch features
        x_s_proj = F.normalize(x_s, dim=-1)
        x_l_proj = F.normalize(x_l, dim=-1)

        # Compute cosine similarity to all textual descriptions
        desc_feats = self.desc_text_features.to(self.device)  # [T, D]
        logits_s = torch.matmul(x_s_proj, desc_feats.T)  # [B, N, T]
        logits_l = torch.matmul(x_l_proj, desc_feats.T)  # [B, N, T]
        print(f"[CONCH] Logits shapes: low-res {logits_s.shape}, high-res {logits_l.shape}")

        # Combine low-res and high-res patches
        x_proj = torch.cat([x_s_proj, x_l_proj], dim=1)  # [B, 2N, D]
        logits = torch.cat([logits_s, logits_l], dim=1)  # [B, 2N, T]
        print(f"[CONCH] Combined logits shape: {logits.shape}")

        # For each patch, select top description per class
        patch_scores = torch.zeros(B, 2 * N, device=self.device)  # [B, 2N]
        best_desc_per_class = torch.zeros(B, 2 * N, self.num_classes, dtype=torch.long, device=self.device)  # [B, 2N, C]

        for b in range(B):
            patch_logits = logits[b]  # [2N, T]
            for class_id, (start, end) in self.class_to_desc_idx.items():
                class_desc_logits = patch_logits[:, start:end]  # [2N, num_desc]
                max_scores, max_indices = class_desc_logits.max(dim=1)  # [2N], [2N]
                patch_scores[b] = torch.maximum(patch_scores[b], max_scores)  # Update with max score across classes
                best_desc_per_class[b, :, class_id] = start + max_indices  # Absolute desc indices

        # Get top-j indices per slide
        topj = min(topj, 2 * N)
        top_idx = torch.topk(patch_scores, topj, dim=1)[1]  # [B, j]

        # Gather top-j patch embeddings
        def gather_top_patches(feats, idx):
            B, N, D = feats.shape
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, j, D]
            return torch.gather(feats, dim=1, index=idx_exp)  # [B, j, D]

        top_feats = gather_top_patches(x_proj, top_idx)  # [B, j, D]
        image_features = F.normalize(top_feats.mean(dim=1), dim=-1)  # [B, D]

        # Compute slide-level logits using averaged text features per class
        class_text_features = torch.zeros(self.num_classes, desc_feats.shape[-1], device=self.device)  # [C, D]
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_text_features[class_id] = desc_feats[start:end].mean(dim=0)  # Average features for class

        class_logits = image_features @ class_text_features.T  # [B, C]
        class_logits = class_logits * self.logit_scale.exp()  # Apply logit scale

        # Compute softmax and predictions
        Y_prob = F.softmax(class_logits, dim=1)  # [B, C]
        Y_hat = Y_prob.argmax(dim=1)  # [B]

        # Extract best description for each predicted class
        all_descriptions = sum(self.text_prompts.values(), [])  # Flatten to list[str]
        top_descriptions = []
        for b in range(B):
            pred_cls = Y_hat[b].item()
            # Get the top patch for this sample
            top_patch_idx = top_idx[b, 0].item()  # Use the top-1 patch for description
            desc_idx = best_desc_per_class[b, top_patch_idx, pred_cls].item()
            top_descriptions.append(all_descriptions[desc_idx])

        # Compute loss if label is provided
        loss = self.loss_ce(class_logits, label) if label is not None else None

        return Y_prob, Y_hat, loss