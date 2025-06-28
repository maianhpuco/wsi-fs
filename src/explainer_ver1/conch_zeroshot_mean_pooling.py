import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("src/externals/CONCH") 
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

class CONCH_ZeroShot_Model_MeanPooling(nn.Module):
    def __init__(self, config, num_classes=None):
        super(CONCH_ZeroShot_Model_MeanPooling, self).__init__()
        self.device = config.device
        self.num_classes = num_classes
        self.text_prompt = config.text_prompt  # <- list of 2 * num_classes

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
        print("Encoding text prompts...")
        print(prompts)

        # Return PyTorch tensors, pad to the longest sequence
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # Use only input_ids, as required by open_clip-style model
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
            Y_prob: softmax probabilities [B, C]
            Y_hat: predicted label [B]
            loss: cross-entropy loss
        """
        B, N, D = x_s.shape
        x_s_proj = F.normalize(x_s, dim=-1)
        # x_s_proj = self.forward_project(x_s.view(-1, D)).view(B, N, -1)  # [B, N, D']

        B, N, D = x_l.shape
        x_l_proj = F.normalize(x_l, dim=-1)
        # x_l_proj = self.forward_project(x_l.view(-1, D)).view(B, N, -1)  # [B, N, D']
 
        # B = x_s.size(0)
        # Mean pooling over patches
        image_features_low = F.normalize(x_s_proj.mean(dim=1), dim=-1)    # [B, D]
        image_features_high = F.normalize(x_l_proj.mean(dim=1), dim=-1)   # [B, D]

        # Compute logit
        logits_low = image_features_low @ self.text_features_low.T.cuda()     # [B, C]
        logits_high = image_features_high @ self.text_features_high.T.cuda()  # [B, C]
        logits = logits_low + logits_high

        # loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        return Y_prob, Y_hat, None 