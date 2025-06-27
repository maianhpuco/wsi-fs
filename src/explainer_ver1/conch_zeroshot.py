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
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, embeddings, tokenized_prompts):
        x = embeddings + self.positional_embedding.type(self.dtype)  # [B, 77, D]
        x = x.permute(1, 0, 2)  # [77, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 77, D]
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class CONCH_ZeroShot_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(CONCH_ZeroShot_Model, self).__init__()
        self.device = config.device
        self.num_classes = num_classes

        self.conch_model, self.preprocess = create_model_from_pretrained(
            model_cfg="conch_ViT-B-16",
            checkpoint_path=config.weight_path,
            device=self.device,
            hf_auth_token=os.environ.get("HF_TOKEN")
        )
        self.logit_scale = self.conch_model.logit_scale
        self.tokenizer = _Tokenizer()
        self.text_encoder = TextEncoder(self.conch_model).to(self.device)

        # Load text prompts
        if isinstance(config.text_prompt, str) and config.text_prompt.endswith(".csv"):
            self.text_prompt = pd.read_csv(config.text_prompt, header=None)[0].tolist()
        else:
            self.text_prompt = config.text_prompt

        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh()).to(self.device)
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid()).to(self.device)
        self.attention_weights = nn.Linear(self.D, self.K).to(self.device)

    def encode_text(self, prompts):
        tokenized = [self.tokenizer.encode(p) for p in prompts]
        max_len = min(max(len(t) for t in tokenized), 77)

        # Pad and truncate
        token_tensor = torch.zeros(len(prompts), 77, dtype=torch.long)
        for i, tokens in enumerate(tokenized):
            tokens = tokens[:77]
            token_tensor[i, :len(tokens)] = torch.tensor(tokens)

        token_tensor = token_tensor.to(self.device)
        embeddings = self.conch_model.token_embedding(token_tensor).type(self.conch_model.dtype)  # [B, 77, D]
        return F.normalize(self.text_encoder(embeddings, token_tensor), dim=-1)

    def encode_features(self, x):
        x = x.float()
        A_V = self.attention_V(x)         # [B, N, D]
        A_U = self.attention_U(x)         # [B, N, D]
        A = self.attention_weights(A_V * A_U)  # [B, N, K]
        A = A.transpose(1, 2)             # [B, K, N]
        A = F.softmax(A, dim=-1)
        feat = torch.bmm(A, x)            # [B, K, L]
        feat = F.normalize(feat, dim=-1)
        return feat.squeeze(1) if feat.size(1) == 1 else feat

    def forward(self, x_s, coord_s, x_l, coords_l, label=None):
        feat_low = self.encode_features(x_s)
        feat_high = self.encode_features(x_l)

        text_features = self.encode_text(self.text_prompt)
        text_low = text_features[:self.num_classes]
        text_high = text_features[self.num_classes:]

        logits_low = feat_low @ text_low.T * self.logit_scale.exp()
        logits_high = feat_high @ text_high.T * self.logit_scale.exp()
        logits = logits_low + logits_high

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        loss = None
        if label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, label)

        return Y_prob, Y_hat, loss
