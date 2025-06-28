import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH") 
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer


class CONCH_ZeroShot_Model_TopjPooling(nn.Module):
    def __init__(self, config, num_classes=None):
        super(CONCH_ZeroShot_Model_TopjPooling, self).__init__()
        self.device = config.device
        self.num_classes = num_classes
        self.text_prompt = config.text_prompt
        self.topj = getattr(config, "topj", (1, 5, 10))  # Default topj values

        assert isinstance(self.text_prompt, list) and len(self.text_prompt) == 2 * self.num_classes, \
            f"Expected 2 * num_classes text prompts, got {len(self.text_prompt)}"

        # Load CONCH model
        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )

        self.tokenizer = get_tokenizer()
        self.visual = self.model.visual
        self.logit_scale = self.model.logit_scale
        self.loss_ce = nn.CrossEntropyLoss()

        # Precompute text features
        self.text_features_low, self.text_features_high = self.init_text_features()

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        text_features = self.model.encode_text(tokenized["input_ids"])
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

    # def forward(self, x_s, coord_s, x_l, coord_l, label):
    #     """
    #     Args:
    #         x_s: [B, N, D] low-res patch features
    #         x_l: [B, N, D] high-res patch features
    #         label: [B] ground-truth labels
    #     Returns:
    #         Y_probs_dict: {j: [B, C]} softmax outputs for each top-j
    #         Y_hats_dict: {j: [B]} predicted class indices
    #         loss: cross-entropy loss computed on top-1 pooled logits
    #     """
    #     B, N_s, D_s = x_s.shape
    #     B, N_l, D_l = x_l.shape

    #     x_s_proj = self.forward_project(x_s.view(-1, D_s)).view(B, N_s, -1)
    #     x_l_proj = self.forward_project(x_l.view(-1, D_l)).view(B, N_l, -1)

    #     text_features_low = self.text_features_low.to(self.device)
    #     text_features_high = self.text_features_high.to(self.device)

    #     logits_low_all = torch.matmul(x_s_proj, text_features_low.T)  # [B, N_s, C]
    #     logits_high_all = torch.matmul(x_l_proj, text_features_high.T)  # [B, N_l, C]
    #     logits_all = logits_low_all + logits_high_all  # [B, N, C]

    #     Y_probs_dict = {}
    #     Y_hats_dict = {}
    #     pooled_logits_dict = {}

    #     logit_scale = self.logit_scale.exp()

    #     for b in range(B):
    #         patch_logits = logits_all[b]  # [N, C]
    #         preds, pooled_logits = self.topj_pooling(patch_logits, self.topj, logit_scale=logit_scale)

    #         for j in self.topj:
    #             if j not in Y_hats_dict:
    #                 Y_hats_dict[j] = []
    #                 Y_probs_dict[j] = []
    #                 pooled_logits_dict[j] = []

    #             Y_hats_dict[j].append(preds[j])
    #             Y_probs_dict[j].append(F.softmax(pooled_logits[j], dim=1))
    #             pooled_logits_dict[j].append(pooled_logits[j])

    #     # Stack results per topj
    #     for j in self.topj:
    #         Y_hats_dict[j] = torch.stack(Y_hats_dict[j], dim=0)  # [B]
    #         Y_probs_dict[j] = torch.cat(Y_probs_dict[j], dim=0)  # [B, C]
    #         pooled_logits_dict[j] = torch.cat(pooled_logits_dict[j], dim=0)  # [B, C]

    #     loss = self.loss_ce(pooled_logits_dict[self.topj[0]], label)
    #     return Y_probs_dict, Y_hats_dict, loss

    @staticmethod
    def topj_pooling(logits, topj, logit_scale=None):
        """
        Args:
            logits: [N, C] per-patch logits
            topj: tuple of top-k patch counts
            logit_scale: optional scaling (exp() from model.logit_scale)
        Returns:
            preds: {j: Tensor[1]} class index
            pooled_logits: {j: Tensor[1, C]} logits before softmax
        """
        maxj = min(max(topj), logits.size(0))
        values, _ = logits.topk(maxj, dim=0, largest=True, sorted=True)  # [maxj, C]

        preds = {}
        pooled_logits = {}

        for j in topj:
            mean_logits = values[:min(j, maxj)].mean(dim=0, keepdim=True)  # [1, C]
            if logit_scale is not None:
                mean_logits = mean_logits * logit_scale
            pooled_logits[j] = mean_logits
            preds[j] = mean_logits.argmax(dim=1)

        return preds, pooled_logits
    
    def forward(self, x_s, coord_s, x_l, coord_l, label, topj=100):
        """
        Args:
            x_s: low-res patch features [B, N, D]
            x_l: high-res patch features [B, N, D]
            label: ground-truth label [B]
            topj: number of top patches to average for mean pooling
        Returns:
            Y_prob: softmax probabilities [B, C]
            Y_hat: predicted label [B]
            loss: cross-entropy loss
        """
        B, N, D = x_s.shape
        x_s_proj = self.forward_project(x_s.view(-1, D)).view(B, N, -1)  # [B, N, D']

        B, N, D = x_l.shape
        x_l_proj = self.forward_project(x_l.view(-1, D)).view(B, N, -1)  # [B, N, D']

        # Compute logits for each patch
        logits_s = torch.matmul(x_s_proj, self.text_features_low.T.cuda())  # [B, N, C]
        logits_l = torch.matmul(x_l_proj, self.text_features_high.T.cuda())  # [B, N, C]

        # Patch-level max-class logits to score patch importance
        patch_scores_s = logits_s.max(dim=2)[0]  # [B, N]
        patch_scores_l = logits_l.max(dim=2)[0]  # [B, N]

        # Get top-j indices per sample
        topj = min(topj, N)
        top_idx_s = torch.topk(patch_scores_s, topj, dim=1)[1]  # [B, j]
        top_idx_l = torch.topk(patch_scores_l, topj, dim=1)[1]  # [B, j]

        # Gather top-j patch embeddings
        def gather_top_patches(feats, idx):
            B, N, D = feats.shape
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, j, D]
            return torch.gather(feats, dim=1, index=idx_exp)  # [B, j, D]

        top_feat_s = gather_top_patches(x_s_proj, top_idx_s)  # [B, j, D]
        top_feat_l = gather_top_patches(x_l_proj, top_idx_l)  # [B, j, D]

        image_features_low = F.normalize(top_feat_s.mean(dim=1), dim=-1)  # [B, D]
        image_features_high = F.normalize(top_feat_l.mean(dim=1), dim=-1)  # [B, D]

        # Compute logits
        logits_low = image_features_low @ self.text_features_low.T.cuda()     # [B, C]
        logits_high = image_features_high @ self.text_features_high.T.cuda()  # [B, C]
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        return Y_prob, Y_hat, loss
