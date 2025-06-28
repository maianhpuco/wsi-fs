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

        assert isinstance(self.text_prompts, dict)
        for prompts in self.text_prompts.values():
            assert isinstance(prompts, list)

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

        # Only apply projection if text_features not yet projected
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
        B, N, D = x_s.shape
        x_s_proj = F.normalize(x_s, dim=-1)
        x_l_proj = F.normalize(x_l, dim=-1)

        # Compute cosine similarity to all descriptions
        desc_feats = self.desc_text_features.to(self.device)  # [T, D]
        logits_s = torch.matmul(x_s_proj, desc_feats.T)  # [B, N, T]
        logits_l = torch.matmul(x_l_proj, desc_feats.T)  # [B, N, T]

        # Concatenate low-res and high-res logits
        logits = torch.cat([logits_s, logits_l], dim=1)  # [B, 2N, T]

        # Max over patches
        logits_max = logits.max(dim=1)[0]  # [B, T]

        # Aggregate per class (max over its description indices)
        class_logits = torch.zeros(B, self.num_classes, device=self.device)
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_logits[:, class_id] = logits_max[:, start:end].max(dim=1)[0]

        # Classification outputs
        Y_prob = F.softmax(class_logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)
        loss = self.loss_ce(class_logits, label) if label is not None else None

        # Optional: return top-matching description
        best_desc_idx = logits_max.argmax(dim=1)
        all_descriptions = sum(self.text_prompts.values(), [])
        top_descriptions = [all_descriptions[i] for i in best_desc_idx.tolist()]

        return Y_prob, Y_hat, loss, top_descriptions
