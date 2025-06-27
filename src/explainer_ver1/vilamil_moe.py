# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import clip

from .model_utils import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from explainer_ver1 import PromptLearner, TextEncoder

logger = logging.getLogger(__name__)
_tokenizer = _Tokenizer()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
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

        clip_model, _ = clip.load("RN50", device='cpu')
        clip_model.float()
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float()).to(self.device)
        self.text_encoder = TextEncoder(clip_model.float()).to(self.device)

        self.norm = nn.LayerNorm(config.input_size).to(self.device)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1).to(self.device)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1).to(self.device)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.input_size),
                nn.Linear(config.input_size, config.input_size),
                nn.ReLU(),
                nn.Linear(config.input_size, config.input_size)
            ) for _ in range(self.num_classes)
        ]).to(self.device)

    def forward(self, x_s, coord_s, x_l, coords_l, label):
        device = x_s.device

        prompts = self.prompt_learner().to(device)
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
        text_features = self.text_encoder(prompts, tokenized_prompts).to(device)

        M = x_s.float()
        compents, _ = self.cross_attention_1(M, M, M)
        compents = self.norm(compents + M)

        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(M_high, M_high, M_high)
        compents_high = self.norm(compents_high + M_high)

        # Expert outputs (dense): each expert sees all features and aggregates
        expert_outputs_low = []
        for expert in self.experts:
            feat = expert(compents)
            expert_outputs_low.append(feat.mean(dim=0, keepdim=True))
        dense_expert_features_low = torch.cat(expert_outputs_low, dim=0)

        expert_outputs_high = []
        for expert in self.experts:
            feat_high = expert(compents_high)
            expert_outputs_high.append(feat_high.mean(dim=0, keepdim=True))
        dense_expert_features_high = torch.cat(expert_outputs_high, dim=0)

        # Align with corresponding text features
        text_features_low = text_features[:self.num_classes]
        text_features_high = text_features[self.num_classes:]

        image_context = torch.cat((compents, M), dim=0)
        text_context_low, _ = self.cross_attention_2(text_features_low.unsqueeze(1), image_context, image_context)
        text_features_low = text_context_low.squeeze() + text_features_low

        image_context_high = torch.cat((compents_high, M_high), dim=0)
        text_context_high, _ = self.cross_attention_2(text_features_high.unsqueeze(1), image_context_high, image_context_high)
        text_features_high = text_context_high.squeeze() + text_features_high

        logits_low = dense_expert_features_low @ text_features_low.T.cuda()
        logits_high = dense_expert_features_high @ text_features_high.T.cuda()
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        return Y_prob, Y_hat, loss