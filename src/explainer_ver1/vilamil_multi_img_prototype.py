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
import clip
from os.path import join as pjoin
from .model_utils import *

logger = logging.getLogger(__name__)
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
_path = os.path.abspath(os.path.join(current_dir, "../..", 'src/explainer'))
sys.path.append(_path)

from explainer_ver1 import PromptLearner, TextEncoder

# Utility function for truncated normal initialization
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
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


class ViLa_MIL_Model_MultiProto(nn.Module):
    def __init__(self, config, num_classes=3):
        super(ViLa_MIL_Model_MultiProto, self).__init__()
        self.device = config.device
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1

        # Attention modules for pooling
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        # Load CLIP model and initialize prompt learner and text encoder
        clip_model, _ = clip.load("RN50", device='cpu')
        clip_model.float()
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float()).to(self.device)
        self.text_encoder = TextEncoder(clip_model.float()).to(self.device)

        # Cross-attention and normalization layers
        self.norm = nn.LayerNorm(config.input_size).to(self.device)
        self.cross_attention_1 = nn.MultiheadAttention(embed_dim=config.input_size, num_heads=1, batch_first=True).to(self.device)
        self.cross_attention_2 = nn.MultiheadAttention(embed_dim=config.input_size, num_heads=1, batch_first=True).to(self.device)

        # Class-specific prototypes (learnable)
        self.class_prototypes = nn.ParameterList([
            nn.Parameter(torch.empty(1, config.input_size, device=self.device))
            for _ in range(num_classes)
        ])
        for p in self.class_prototypes:
            trunc_normal_(p, std=0.02)

        # Move modules to device
        self.attention_V.to(self.device)
        self.attention_U.to(self.device)
        self.attention_weights.to(self.device)

    def forward(self, x_s, coord_s, x_l, coords_l, label):
        device = x_s.device

        # Text encoding with prompts
        prompts = self.prompt_learner().to(device)
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
        text_features = self.text_encoder(prompts, tokenized_prompts).to(device)

        M = x_s.float()       # Low resolution patch features
        M_high = x_l.float()  # High resolution patch features

        # Cross-attention using class-specific prototypes
        compents_list = []
        compents_high_list = []

        for c in range(self.num_classes):
            proto = self.class_prototypes[c].unsqueeze(0)  # (1, 1, D)
            comp, _ = self.cross_attention_1(proto, M, M)
            comp = self.norm(comp + proto)
            compents_list.append(comp.squeeze(0))

            comp_high, _ = self.cross_attention_1(proto, M_high, M_high)
            comp_high = self.norm(comp_high + proto)
            compents_high_list.append(comp_high.squeeze(0))

        compents = torch.stack(compents_list, dim=0)        # (C, D)
        compents_high = torch.stack(compents_high_list, dim=0)  # (C, D)

        # Attention-pooling to get image-level representation
        image_features_low, image_features_high = [], []

        for c in range(self.num_classes):
            H_c = compents[c].unsqueeze(0)
            A = self.attention_weights(self.attention_V(H_c) * self.attention_U(H_c))
            A = F.softmax(A, dim=1)
            feat = torch.matmul(A, H_c)
            image_features_low.append(feat.squeeze(0))

            H_h = compents_high[c].unsqueeze(0)
            A_h = self.attention_weights(self.attention_V(H_h) * self.attention_U(H_h))
            A_h = F.softmax(A_h, dim=1)
            feat_h = torch.matmul(A_h, H_h)
            image_features_high.append(feat_h.squeeze(0))

        image_features_low = torch.stack(image_features_low, dim=0)
        image_features_high = torch.stack(image_features_high, dim=0)

        # Text features after context alignment (low + high)
        text_features_low = text_features[:self.num_classes]
        text_features_high = text_features[self.num_classes:]

        image_context = torch.cat((compents, M), dim=0)
        text_context_features, _ = self.cross_attention_2(text_features_low.unsqueeze(1), image_context, image_context)
        text_features_low = text_context_features.squeeze() + text_features_low

        image_context_high = torch.cat((compents_high, M_high), dim=0)
        text_context_features_high, _ = self.cross_attention_2(text_features_high.unsqueeze(1), image_context_high, image_context_high)
        text_features_high = text_context_features_high.squeeze() + text_features_high

        # Classification logits
        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high

        # Compute loss and predictions
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        return Y_prob, Y_hat, loss