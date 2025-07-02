import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer


class Ver2f(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.device = config.device
        self.num_classes = num_classes
        self.text_prompts = dict(config.text_prompts)
        self.tokenizer = get_tokenizer()

        for class_name, desc_list in self.text_prompts.items():
            assert isinstance(desc_list, list), f"Descriptions for '{class_name}' must be a list."
            for desc in desc_list:
                assert len(self.tokenizer.encode(desc)) <= 77, f"Prompt too long: '{desc}'"

        self.model, _ = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )

        self.visual = self.model.visual
        self.logit_scale = self.model.logit_scale
        self.loss_ce = nn.CrossEntropyLoss()

        feat_dim = getattr(config, "input_size", 512)
        self.visual_proj = nn.Linear(feat_dim, feat_dim)
        # self.attn_pooling = AttentionPooling(feat_dim)

        self.desc_text_features, self.class_to_desc_idx = self.init_text_features()
        self.desc_text_features = self.desc_text_features.detach()

        self.text_features_low = self.aggregate_class_features().detach()
        self.text_features_high = self.text_features_low

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
        desc_feats, class_to_desc = [], {}
        start_idx = 0
        for class_id, (class_name, desc_list) in enumerate(self.text_prompts.items()):
            features = self.encode_text(desc_list)
            end_idx = start_idx + features.size(0)
            desc_feats.append(features)
            class_to_desc[class_id] = (start_idx, end_idx)
            start_idx = end_idx
        return torch.cat(desc_feats, dim=0), class_to_desc

    def aggregate_class_features(self):
        C, D = self.num_classes, self.desc_text_features.size(1)
        class_feats = torch.zeros(C, D, device=self.device)
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_feats[class_id] = self.desc_text_features[start:end].max(dim=0)[0]
        return F.normalize(class_feats, dim=-1)

    # def compute_patch_scores(self, patch_feats, desc_feats):
    #     return patch_feats @ desc_feats.T
    #per-patch description similarity scores 
    def compute_patch_scores(self, patch_feats, desc_feats):
        """
        patch_feats: [B, N, D]
        desc_feats: [T, D]
        Returns:
            patch_scores: [B, N, T]
            slide_scores: [B, 1, T] (mean pooled)
        """
        patch_feats = F.normalize(patch_feats, dim=-1)
        desc_feats = F.normalize(desc_feats, dim=-1)  # [T, D]
        sim = torch.matmul(patch_feats, desc_feats.T)  # [B, N, T]
        #per-slide description similarity scores  
        # Aggregate patch scores to slide-level by mean
        # slide_scores = sim.mean(dim=1, keepdim=True)  # [B, 1, T]
        # sim: [B, N, T] = patch-to-description similarities
        weights = F.softmax(sim, dim=1)  # softmax over patches → [B, N, T]
        slide_scores = torch.sum(weights * sim, dim=1, keepdim=True)  # [B, 1, T]
        
        return sim, slide_scores
    

    # def get_class_scores_from_descriptions(self, logits_desc): # B=batch size, N=patches, T=total number of descriptions 
    #     B, N, T = logits_desc.shape
    #     C = self.num_classes
    #     class_scores = torch.zeros(B, N, C, device=logits_desc.device)
    #     for class_id, (start, end) in self.class_to_desc_idx.items():
    #         class_scores[:, :, class_id] = logits_desc[:, :, start:end].mean(dim=2)
    #     return class_scores # shape: [B, N, C] 

    def get_class_scores_from_slide_scores(self, slide_score):  # slide_score: [B, 1, T]
        B, _, T = slide_score.shape
        C = self.num_classes
        class_scores = torch.zeros(B, C, device=slide_score.device)
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_scores[:, class_id] = slide_score[:, 0, start:end].mean(dim=1)  # [B]
        return class_scores  # [B, C]
 
    def forward(self, x_s, coord_s, x_l, coord_l, label=None):
        if x_s.ndim == 2:
            x_s = x_s.unsqueeze(0)
            x_l = x_l.unsqueeze(0)
            coord_s = coord_s.unsqueeze(0)
            coord_l = coord_l.unsqueeze(0)

        B, N, D = x_s.shape

        x_s_proj = F.normalize(self.visual_proj(F.normalize(x_s, dim=-1)), dim=-1)
        # x_l_proj = F.normalize(self.visual_proj(F.normalize(x_l, dim=-1)), dim=-1)

        _, slide_score = self.compute_patch_scores(x_s_proj, self.desc_text_features)  # [B, N, T]
        
        logits = self.get_class_scores_from_slide_scores(slide_score)  # [B, C]
        loss = self.loss_ce(logits, label) if label is not None else None
 
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)

        return Y_prob, Y_hat, loss, slide_score

     