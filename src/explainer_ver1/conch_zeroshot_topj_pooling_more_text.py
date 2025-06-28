import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH")
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

class CONCH_ZeroShot_Model_TopjMaxDesc(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.device = config.device
        self.num_classes = num_classes
        self.text_prompts = config.text_prompts  # dict: {class_name: [desc1, desc2, ...]}
        self.topj = getattr(config, "topj", 10)

        assert isinstance(self.text_prompts, dict)
        for prompts in self.text_prompts.values():
            assert isinstance(prompts, list)

        # Load CONCH model
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

        # Encode all text descriptions and associate with class
        self.desc_text_features, self.class_to_desc_idx = self.init_text_features()

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        text_features = self.model.encode_text(tokenized["input_ids"])
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

        # Use topj pooling from both resolutions
        patch_scores_s = x_s_proj.norm(p=2, dim=-1)  # optional: relevance
        patch_scores_l = x_l_proj.norm(p=2, dim=-1)

        topj = min(topj, N)
        idx_s = torch.topk(patch_scores_s, topj, dim=1)[1]
        idx_l = torch.topk(patch_scores_l, topj, dim=1)[1]

        def gather(feats, idx):
            return torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)))

        top_feat_s = gather(x_s_proj, idx_s)  # [B, j, D]
        top_feat_l = gather(x_l_proj, idx_l)  # [B, j, D]

        top_feat = F.normalize(torch.cat([top_feat_s, top_feat_l], dim=1), dim=-1)  # [B, 2j, D]

        # Compute similarity to each description
        logits = torch.einsum("bnd,td->bnt", top_feat, self.desc_text_features.T)  # [B, 2j, T]
        logits_max = logits.max(dim=1)[0]  # max over top patches -> [B, T]

        # For each class, aggregate max of its prompts
        class_logits = torch.zeros(B, self.num_classes).to(self.device)
        for class_id, (start, end) in self.class_to_desc_idx.items():
            class_logits[:, class_id] = logits_max[:, start:end].max(dim=1)[0]  # max over that class's descs

        Y_prob = F.softmax(class_logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)

        loss = self.loss_ce(class_logits, label) if label is not None else None
        return Y_prob, Y_hat, loss
