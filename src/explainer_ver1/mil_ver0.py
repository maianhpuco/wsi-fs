# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
# Set up path
import os 
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
_path = os.path.abspath(os.path.join(current_dir, "../..", 'src/explainer'))
sys.path.append(_path) 

from explainer_ver1 import PromptLearner 
from explainer_ver1 import TextEncoder

 
 
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class Explainer_Ver1(nn.Module):
    def __init__(self, config, num_classes=3):
        super(Explainer_Ver1, self).__init__()
        
        # Define cross-entropy loss for classification
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size  # Feature dimension of each image patch

        # Load CLIP model backbone (ResNet-50 in this case)
        clip_model, _ = clip.load("RN50", device="cpu")

        # Initialize learnable text prompts (e.g., class-specific descriptions)
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float())

        # Text encoder from CLIP for processing tokenized prompts
        self.text_encoder = TextEncoder(clip_model.float())

        # Cross-attention module for aligning text features with image features
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1)

        # Layer normalization applied after attention
        self.norm = nn.LayerNorm(self.L)
        
        # Gated attention for patch-level feature aggregation
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.L), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.L, 1)
        
    def forward(self, x_s, coord_s, x_l, coords_l, label):
        # -------- TEXT ENCODING --------
        # Generate prompt embeddings (learned prompt tokens)
        prompts = self.prompt_learner()
        # Get tokenized input for CLIP's transformer
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        # Encode the prompts into dense vectors (text features), shape: [C+extra, D]
        text_features = self.text_encoder(prompts, tokenized_prompts)


        # -------- IMAGE FEATURES --------
        # Convert low-res and high-res patch features to float
        M = x_s.float()       # Low-resolution features [N_low, D]
        M_high = x_l.float()  # High-resolution features [N_high, D]

        # -------- CROSS-ATTENTION (LOW) --------
        # Extract only the class-related text features for low-res
        text_features_low = text_features[:self.num_classes]

        # Use cross-attention to refine text features using image features as context
        image_context = M
        text_context_features, _ = self.cross_attention_2(
            text_features_low.unsqueeze(1),  # [C, 1, D]
            image_context, image_context     # [N, D], [N, D]
        )
        # Residual update of text features with attended context
        text_features_low = text_context_features.squeeze() + text_features_low


        # -------- CROSS-ATTENTION (HIGH) --------
        # Same as above but for high-res patches and remaining prompts
        text_features_high = text_features[self.num_classes:]
        image_context_high = M_high
        text_context_features_high, _ = self.cross_attention_2(
            text_features_high.unsqueeze(1),
            image_context_high, image_context_high
        )
        text_features_high = text_context_features_high.squeeze() + text_features_high

        # -------- GATED ATTENTION POOLING (LOW) --------
        A_V = self.attention_V(M)          # [N_low, D]
        A_U = self.attention_U(M)          # [N_low, D]
        A = self.attention_weights(A_V * A_U)  # [N_low, 1]
        A = torch.transpose(A, 1, 0)       # [1, N_low]
        A = F.softmax(A, dim=1)            # Attention weights
        image_features_low = torch.mm(A, M)  # [1, D]

        # -------- GATED ATTENTION POOLING (HIGH) --------
        A_V_high = self.attention_V(M_high)
        A_U_high = self.attention_U(M_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, M_high)
 

        # -------- IMAGE FEATURE POOLING --------
        # Compute global image feature by averaging all patch features
        # image_features_low = torch.mean(M, dim=0, keepdim=True)        # [1, D]
        # image_features_high = torch.mean(M_high, dim=0, keepdim=True)  # [1, D]


        # -------- CLASSIFICATION LOGITS --------
        # Compute similarity between image and refined text features
        logits_low = image_features_low @ text_features_low.T.cuda()     # [1, C]
        logits_high = image_features_high @ text_features_high.T.cuda()  # [1, C]
        logits = logits_low + logits_high  # Combine both resolutions


        # -------- PREDICTION --------
        # Compute classification loss
        loss = self.loss_ce(logits, label)

        # Softmax for class probabilities
        Y_prob = F.softmax(logits, dim=1)

        # Get predicted class with highest probability
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return Y_prob, Y_hat, loss
