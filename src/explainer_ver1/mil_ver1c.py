# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from .model_utils import *  # Contains MultiheadAttention
from explainer_ver1 import PromptLearner, TextEncoder

logger = logging.getLogger(__name__)

# Truncated normal initialization
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(ViLa_MIL_Model, self).__init__()
        self.device = config.device
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        self.prototype_number = config.prototype_number

        # Attention modules
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        # Load CLIP model
        clip_model, _ = clip.load("RN50", device='cpu')
        clip_model.float()
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float()).to(self.device)
        self.text_encoder = TextEncoder(clip_model.float()).to(self.device)

        # MLP for text-conditioned prototype generation: p_i = MLP(T_i)
        self.prototype_mlp = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.L)
        ).to(self.device)

        # Cross-attention layers
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1).to(self.device)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1).to(self.device)
        self.norm = nn.LayerNorm(self.L).to(self.device)

        # Move attention modules
        self.attention_V.to(self.device)
        self.attention_U.to(self.device)
        self.attention_weights.to(self.device)

    def forward(self, x_s, coord_s, x_l, coords_l, label):
        device = x_s.device

        # === TEXT FEATURE EXTRACTION ===
        prompts = self.prompt_learner().to(device)
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
        text_features = self.text_encoder(prompts, tokenized_prompts).to(device)  # shape: [C + extra, L]

        # === TEXT-CONDITIONED PROTOTYPES ===
        assert self.prototype_number <= text_features.shape[0], \
            "Prototype number exceeds available text features"
        prototype_inputs = text_features[:self.prototype_number]  # shape: [P, L]
        prototypes = self.prototype_mlp(prototype_inputs).unsqueeze(1)  # [P, 1, L]

        # === LOW-SCALE BRANCH ===
        M = x_s.float()
        compents, _ = self.cross_attention_1(prototypes, M, M)
        compents = self.norm(compents + prototypes)

        H = compents.squeeze().float()  # shape: [P, L]
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)  # [1, P]
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, H)  # [1, L]

        # === HIGH-SCALE BRANCH ===
        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(prototypes, M_high, M_high)
        compents_high = self.norm(compents_high + prototypes)

        H_high = compents_high.squeeze().float()  # shape: [P, L]
        A_V_high = self.attention_V(H_high)
        A_U_high = self.attention_U(H_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)  # [1, P]
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, H_high)  # [1, L]

        # === TEXT CONTEXTUALIZATION (CROSS-ATTENTION) ===
        text_features_low = text_features[:self.num_classes]  # [C, L]
        image_context = torch.cat((compents.squeeze(), M), dim=0)
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1), image_context, image_context)
        text_features_low = text_context_features.squeeze() + text_features_low  # residual

        text_features_high = text_features[self.num_classes:]  # [extra, L]
        image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high, image_context_high)
        text_features_high = text_context_features_high.squeeze() + text_features_high

        # === CLASSIFICATION LOGITS ===
        logits_low = image_features_low @ text_features_low.T.to(device)  # [1, C]
        logits_high = image_features_high @ text_features_high.T.to(device)  # [1, extra]
        logits = logits_low + logits_high[:, :self.num_classes]  # sum only over shared classes

        # === LOSS & PREDICTIONS ===
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        return Y_prob, Y_hat, loss
