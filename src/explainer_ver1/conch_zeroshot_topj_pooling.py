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
        self.topj = getattr(config, "topj", (1, 5, 10))  # default fallback

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

    def encode_image(self, image):
        image_features = self.model.encode_image(image)
        return F.normalize(image_features, dim=-1)

    def forward_project(self, patch_features):
        projected = self.visual.forward_project(patch_features)
        return F.normalize(projected, dim=-1)

    def init_text_features(self):
        low_text = self.text_prompt[:self.num_classes]
        high_text = self.text_prompt[self.num_classes:]
        text_features_low = self.encode_text(low_text)   # [C, D]
        text_features_high = self.encode_text(high_text) # [C, D]
        return text_features_low, text_features_high

    def forward(self, x_s, coord_s, x_l, coord_l, label):
        """
        Args:
            x_s: low-res patch features [B, N, D]
            x_l: high-res patch features [B, N, D]
            label: ground-truth label [B]
        Returns:
            Y_probs_dict: softmax probabilities for topj values {j: [B, C]}
            Y_hats_dict: predictions for topj values {j: [B]}
            loss: cross-entropy loss (computed on top-1 pooled logits)
        """
        B, N_s, D_s = x_s.shape
        B, N_l, D_l = x_l.shape

        x_s_proj = self.forward_project(x_s.view(-1, D_s)).view(B, N_s, -1)  # [B, N_s, D']
        x_l_proj = self.forward_project(x_l.view(-1, D_l)).view(B, N_l, -1)  # [B, N_l, D']

        logits_low_all = torch.matmul(x_s_proj, self.text_features_low.T.cuda())  # [B, N_s, C]
        logits_high_all = torch.matmul(x_l_proj, self.text_features_high.T.cuda())  # [B, N_l, C]

        logits_all = logits_low_all + logits_high_all  # [B, N, C]

        Y_probs_dict = {}
        Y_hats_dict = {}
        pooled_logits_dict = {}

        for b in range(B):
            patch_logits = logits_all[b]  # [N, C]
            
            preds, pooled_logits =self.topj_pooling(patch_logits, self.topj)
            for j in self.topj:
                if j not in Y_hats_dict:
                    Y_hats_dict[j] = []
                    Y_probs_dict[j] = []
                    pooled_logits_dict[j] = []
                Y_hats_dict[j].append(preds[j])
                Y_probs_dict[j].append(F.softmax(pooled_logits[j], dim=1))
                pooled_logits_dict[j].append(pooled_logits[j])

        # Stack results
        for j in self.topj:
            Y_hats_dict[j] = torch.stack(Y_hats_dict[j])  # [B]
            Y_probs_dict[j] = torch.cat(Y_probs_dict[j], dim=0)  # [B, C]
            pooled_logits_dict[j] = torch.cat(pooled_logits_dict[j], dim=0)  # [B, C]

        loss = self.loss_ce(pooled_logits_dict[self.topj[0]], label)
        return Y_probs_dict, Y_hats_dict, loss
    
    @staticmethod
    def topj_pooling(logits, topj):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
        values, _ = logits.topk(maxj, 0, True, True) # maxj x C
        preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
        pooled_logits = {key: val for key,val in preds.items()}    
        preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
        return preds, pooled_logits
