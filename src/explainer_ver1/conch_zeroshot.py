import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pandas as pd

# Setup path to CONCH
current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
_path = os.path.abspath(os.path.join(current_dir, "../..", 'src/externals/CONCH'))
sys.path.append(_path)

from conch.open_clip_custom import create_model_from_pretrained

class CONCH_ZeroShot_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(CONCH_ZeroShot_Model, self).__init__()
        self.device = config.device
        self.num_classes = num_classes

        # Load CoCa-based CONCH model
        self.conch_model, self.preprocess = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )
        self.logit_scale = self.conch_model.logit_scale
        self.tokenizer = self.conch_model.tokenizer  # use built-in tokenizer

        # Load text prompts
        if isinstance(config.text_prompt, str) and config.text_prompt.endswith(".csv"):
            self.text_prompt = pd.read_csv(config.text_prompt, header=None)[0].tolist()
        else:
            self.text_prompt = config.text_prompt

        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1

        # Patch-level attention
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh()).to(self.device)
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid()).to(self.device)
        self.attention_weights = nn.Linear(self.D, self.K).to(self.device)

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts).to(self.device)
        text_features = self.conch_model.encode_text(tokenized)  # shape [num_prompts, D]
        return F.normalize(text_features, dim=-1)

    def encode_features(self, x):
        x = x.float()  # [B, N, L]
        A_V = self.attention_V(x)         # [B, N, D]
        A_U = self.attention_U(x)         # [B, N, D]
        A = self.attention_weights(A_V * A_U)  # [B, N, K]
        A = A.transpose(1, 2)             # [B, K, N]
        A = F.softmax(A, dim=-1)          # attention over patches
        feat = torch.bmm(A, x)            # [B, K, L]
        feat = F.normalize(feat, dim=-1)  # [B, K, L]
        return feat.squeeze(1) if feat.size(1) == 1 else feat  # [B, L] if K=1

    def forward(self, x_s, coord_s, x_l, coords_l, label=None):
        # Encode visual features
        feat_low = self.encode_features(x_s)    # [B, L]
        feat_high = self.encode_features(x_l)   # [B, L]

        # Encode textual prompts
        text_features = self.encode_text(self.text_prompt)  # [2*num_classes, D]
        text_low = text_features[:self.num_classes]          # [num_classes, D]
        text_high = text_features[self.num_classes:]         # [num_classes, D]

        # Compute similarity logits
        logits_low = feat_low @ text_low.T * self.logit_scale.exp()    # [B, C]
        logits_high = feat_high @ text_high.T * self.logit_scale.exp()# [B, C]
        logits = logits_low + logits_high                              # [B, C]

        # Classification output
        Y_prob = F.softmax(logits, dim=1)           # [B, C]
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)  # [B]

        # Optional loss
        loss = None
        if label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, label)

        return Y_prob, Y_hat, loss
