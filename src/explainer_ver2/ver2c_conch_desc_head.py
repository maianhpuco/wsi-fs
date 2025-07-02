import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer


class Ver2c(nn.Module):
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
        # Use actual number of classes from class_to_desc_idx instead of self.num_classes
        actual_num_classes = len(self.class_to_desc_idx)
        D = self.desc_text_features.size(1)
        class_feats = torch.zeros(actual_num_classes, D, device=self.device)
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_feats[class_id] = self.desc_text_features[start:end].max(dim=0)[0]
        return F.normalize(class_feats, dim=-1)

    # def compute_patch_scores(self, patch_feats, desc_feats):
    #     return patch_feats @ desc_feats.T
    #per-patch description similarity scores 

    def compute_patch_scores(self, patch_feats, desc_feats):
        D = patch_feats.shape[-1]
        scale = D ** -0.5
        desc_feats = desc_feats.T  # [D, T]
        
        patch_feats = F.normalize(patch_feats, dim=-1)
        desc_feats = F.normalize(desc_feats, dim=0)

        sim = torch.matmul(patch_feats, desc_feats) * scale  # [B, N, T]
        attn = F.softmax(sim, dim=-1)  # [B, N, T]
        return attn 

    def get_class_scores_from_descriptions(self, logits_desc): # B=batch size, N=patches, T=total number of descriptions 
        B, N, T = logits_desc.shape
        C = self.num_classes
        class_scores = torch.zeros(B, N, C, device=logits_desc.device)
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_scores[:, :, class_id] = logits_desc[:, :, start:end].mean(dim=2)
        return class_scores # shape: [B, N, C] 


    def forward(self, x_s, coord_s, x_l, coord_l, label=None):
        if x_s.ndim == 2:
            x_s = x_s.unsqueeze(0)
            x_l = x_l.unsqueeze(0)
            coord_s = coord_s.unsqueeze(0)
            coord_l = coord_l.unsqueeze(0)

        B, N, D = x_s.shape

        x_s_proj = F.normalize(self.visual_proj(F.normalize(x_s, dim=-1)), dim=-1)
        # x_l_proj = F.normalize(self.visual_proj(F.normalize(x_l, dim=-1)), dim=-1)

        # Step 1: Identify "No Tumor Detected" class and main tumor classes
        all_class_ids = list(self.class_to_desc_idx.keys())
        total_classes = len(all_class_ids)
        no_tumor_class_id = total_classes - 1  # Assume "No Tumor Detected" is the last class
        main_class_ids = [cid for cid in all_class_ids if cid != no_tumor_class_id]

        # Step 2: Filter patches using "No Tumor Detected" descriptions
        if no_tumor_class_id in self.class_to_desc_idx:
            no_tumor_start, no_tumor_end = self.class_to_desc_idx[no_tumor_class_id]
            no_tumor_desc_feats = self.desc_text_features[no_tumor_start:no_tumor_end]
            
            # Compute scores against no tumor descriptions
            no_tumor_scores = self.compute_patch_scores(x_s_proj, no_tumor_desc_feats)  # [B, N, num_no_tumor_desc]
            no_tumor_class_scores = no_tumor_scores.mean(dim=-1)  # [B, N] - average across no tumor descriptions
            
            # Create mask for patches with low no tumor scores (likely tumor patches)
            no_tumor_threshold = 0.5  # Threshold for filtering
            tumor_patch_mask = no_tumor_class_scores < no_tumor_threshold  # [B, N]
            
            # Apply mask to features - zero out likely non-tumor patches
            x_s_proj_filtered = x_s_proj * tumor_patch_mask.unsqueeze(-1)  # [B, N, D]
        else:
            # If no "No Tumor Detected" class, use all patches
            x_s_proj_filtered = x_s_proj
            tumor_patch_mask = torch.ones(B, N, device=x_s_proj.device, dtype=torch.bool)

        # Step 3: Extract descriptions for main tumor classes
        main_class_desc_feats = []
        main_class_to_desc = {}
        current_start = 0
        
        for class_id in main_class_ids:
            if class_id in self.class_to_desc_idx:
                start, end = self.class_to_desc_idx[class_id]
                class_desc = self.desc_text_features[start:end]
                main_class_desc_feats.append(class_desc)
                main_class_to_desc[class_id] = (current_start, current_start + class_desc.size(0))
                current_start += class_desc.size(0)
        
        main_desc_feats = torch.cat(main_class_desc_feats, dim=0)  # [total_main_desc, D]

        # Step 4: Compute patch scores for filtered patches against main classes
        logits_desc_s = self.compute_patch_scores(x_s_proj_filtered, main_desc_feats)  # [B, N, total_main_desc]

        # Step 5: Convert description scores to class scores for main classes
        num_main_classes = len(main_class_ids)
        class_scores_s = torch.zeros(B, N, num_main_classes, device=logits_desc_s.device)
        for idx, class_id in enumerate(main_class_ids):
            if class_id in main_class_to_desc:
                start, end = main_class_to_desc[class_id]
                class_scores_s[:, :, idx] = logits_desc_s[:, :, start:end].mean(dim=2)

        # Only consider patches that passed the tumor filter
        class_scores_s = class_scores_s * tumor_patch_mask.unsqueeze(-1)
        
        class_scores = class_scores_s  # [B, N, num_main_classes]
 
        # Apply attention over patches for each class
        attn_weights = F.softmax(class_scores, dim=1)  # [B, N, num_main_classes]
        logits = torch.sum(attn_weights * class_scores, dim=1)  # [B, num_main_classes]

        # Compute loss if label is provided
        loss = self.loss_ce(logits, label) if label is not None else None

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)

        return Y_prob, Y_hat, loss

     