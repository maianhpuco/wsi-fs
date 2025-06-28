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

    def forward(self, x_s, coord_s, x_l, coord_l, label=None, topj=10):
        B, N, D = x_s.shape
        x_s_proj = F.normalize(x_s, dim=-1)
        x_l_proj = F.normalize(x_l, dim=-1)

        desc_feats = self.desc_text_features.to(self.device)  # [T, D]

        # --------- 1. Compute patch-to-description similarity ---------
        logits_s = torch.matmul(x_s_proj, desc_feats.T)  # [B, N, T]
        logits_l = torch.matmul(x_l_proj, desc_feats.T)  # [B, N, T]

        # --------- 2. Patch-level class score: max over descriptions in each class ---------
        def get_patch_class_scores(logits):  # [B, N, T] → [B, N, C]
            B, N, T = logits.shape
            C = self.num_classes
            class_scores = torch.zeros(B, N, C, device=logits.device)
            for class_id, (start, end) in self.class_to_desc_idx.items():
                class_scores[:, :, class_id] = logits[:, :, start:end].max(dim=2)[0]  # max over descriptions
            return class_scores  # [B, N, C]

        class_scores_s = get_patch_class_scores(logits_s)  # [B, N, C]
        class_scores_l = get_patch_class_scores(logits_l)  # [B, N, C]

        # --------- 3. For each patch, select best class score ---------
        patch_scores_s, _ = class_scores_s.max(dim=2)  # [B, N]
        patch_scores_l, _ = class_scores_l.max(dim=2)  # [B, N]

        # --------- 4. Select top-j patches based on discriminative power ---------
        topj = min(topj, N)
        top_idx_s = torch.topk(patch_scores_s, topj, dim=1)[1]  # [B, j]
        top_idx_l = torch.topk(patch_scores_l, topj, dim=1)[1]  # [B, j]

        def gather_top_patches(feats, idx):
            return torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)))

        top_feat_s = gather_top_patches(x_s_proj, top_idx_s)  # [B, j, D]
        top_feat_l = gather_top_patches(x_l_proj, top_idx_l)  # [B, j, D]

        # --------- 5. Slide-level representation ---------
        image_features_low = F.normalize(top_feat_s.mean(dim=1), dim=-1)   # [B, D]
        image_features_high = F.normalize(top_feat_l.mean(dim=1), dim=-1)  # [B, D]

        # --------- 6. Compute logits using class-level features ---------
        logits_low = image_features_low @ self.text_features_low.T.cuda()     # [B, C]
        logits_high = image_features_high @ self.text_features_high.T.cuda()  # [B, C]
        logits = logits_low + logits_high

        # --------- 7. Classification outputs ---------
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)
        loss = self.loss_ce(logits, label) if label is not None else None

        return Y_prob, Y_hat, loss
