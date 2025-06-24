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
        self.device = config.device  # Store the device for later use
        # Add class weighting for imbalanced dataset (TCGA Renal: KIRP, KIRC, KICH)
        # Based on typical TCGA renal distribution, KICH is usually underrepresented
        class_weights = torch.tensor([1.0, 1.0, 2.0], device=config.device)  # Give more weight to KICH (class 2)
        self.loss_ce = nn.CrossEntropyLoss(weight=class_weights)
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        clip_model, _ = clip.load("RN50", device='cpu')
        clip_model.float() 
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float()).to(self.device)
        self.text_encoder = TextEncoder(clip_model.float()).to(self.device)

        self.norm = nn.LayerNorm(config.input_size).to(self.device)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1).to(self.device)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1).to(self.device)

        self.learnable_image_center = nn.Parameter(
            torch.empty(config.prototype_number, 1, config.input_size, device=self.device)
        )
        trunc_normal_(self.learnable_image_center, std=.02)

        # TOP-style instance-level prompt learning for text prototypes
        # Define renal cell carcinoma-specific text prototypes
        instance_prompt_names = [
            "papillary renal cell carcinoma with papillary architecture",
            "clear cell renal carcinoma with clear cytoplasm", 
            "chromophobe renal cell carcinoma with eosinophilic cytoplasm"
        ]
        
        # Instance-level prompt learner for text prototypes (following TOP approach)
        self.instance_prompt_learner = PromptLearner(instance_prompt_names, clip_model.float()).to(self.device)
        
        # Text prototype names for reference
        self.prototype_names = instance_prompt_names
        
        # Text projection for instance-prototype similarity
        self.text_proto_projection = nn.Linear(config.input_size, config.input_size).to(self.device)

        # Also move attention modules to device
        self.attention_V.to(self.device)
        self.attention_U.to(self.device)
        self.attention_weights.to(self.device)
 
 
    def forward(self, x_s, coord_s, x_l, coords_l, label):
        device = x_s.device 
        self.learnable_image_center = self.learnable_image_center
        
        # Also move prompt and tokenized prompts if needed
        prompts = self.prompt_learner().to(device)
        tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device)
        text_features = self.text_encoder(prompts, tokenized_prompts).to(device) 

        M = x_s.float()
        M_high = x_l.float()
        
        # TOP-style text prototype integration
        # Encode instance-level text prototypes through text encoder
        instance_prompts = self.instance_prompt_learner().to(device)
        instance_tokenized_prompts = self.instance_prompt_learner.tokenized_prompts.to(device)
        instance_text_features = self.text_encoder(instance_prompts, instance_tokenized_prompts).to(device)
        instance_text_features = instance_text_features / instance_text_features.norm(dim=-1, keepdim=True)
        
        # Compute text-image prototype similarity for low resolution features
        M_proj = self.text_proto_projection(M)
        text_proto_similarity_low = M_proj @ instance_text_features.t()  # [N_patches, N_prototypes]
        text_proto_weights_low = F.softmax(text_proto_similarity_low, dim=-1)
        text_enhanced_features_low = text_proto_weights_low @ instance_text_features  # [N_patches, embed_dim]
        
        # Compute text-image prototype similarity for high resolution features  
        M_high_proj = self.text_proto_projection(M_high)
        text_proto_similarity_high = M_high_proj @ instance_text_features.t()  # [N_patches, N_prototypes]
        text_proto_weights_high = F.softmax(text_proto_similarity_high, dim=-1)
        text_enhanced_features_high = text_proto_weights_high @ instance_text_features  # [N_patches, embed_dim]
        
        # Image prototype processing with text enhancement
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M) 
        compents = self.norm(compents + self.learnable_image_center)
        # Enhance with properly encoded text prototypes
        text_enhancement_low = text_enhanced_features_low.mean(0, keepdim=True).unsqueeze(0)
        compents = compents + text_enhancement_low

        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)
        # Enhance with properly encoded text prototypes
        text_enhancement_high = text_enhanced_features_high.mean(0, keepdim=True).unsqueeze(0)
        compents_high = compents_high + text_enhancement_high

        H = compents.squeeze().float()
        A_V = self.attention_V(H)  
        A_U = self.attention_U(H)  
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)  
        A = F.softmax(A, dim=1)  
        image_features_low = torch.mm(A, H)  

        H_high = compents_high.squeeze().float() # shape: [P, D]
        A_V_high = self.attention_V(H_high)  
        A_U_high = self.attention_U(H_high)  
        A_high = self.attention_weights(A_V_high * A_U_high) 
        A_high = torch.transpose(A_high, 1, 0)  
        A_high = F.softmax(A_high, dim=1)  
        image_features_high = torch.mm(A_high, H_high)  

        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(), M), dim=0)
        text_context_features, _ = self.cross_attention_2(text_features_low.unsqueeze(1), image_context, image_context)
        text_features_low = text_context_features.squeeze() + text_features_low

        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
        text_context_features_high, _ = self.cross_attention_2(text_features_high.unsqueeze(1), image_context_high, image_context_high)
        text_features_high = text_context_features_high.squeeze() + text_features_high


        # Use instance text prototypes for final classification (TOP approach)
        # Combine both the original text features and instance text prototypes
        combined_text_features_low = torch.cat([text_features_low, instance_text_features], dim=0)
        combined_text_features_high = torch.cat([text_features_high, instance_text_features], dim=0)
        
        logits_low = image_features_low @ combined_text_features_low.T
        logits_high = image_features_high @ combined_text_features_high.T
        
        # Average the logits from both text feature sets
        logits_original_low = image_features_low @ text_features_low.T
        logits_original_high = image_features_high @ text_features_high.T
        logits_instance_low = image_features_low @ instance_text_features.T
        logits_instance_high = image_features_high @ instance_text_features.T
        
        # Combine logits: use instance text prototypes as primary classifier
        logits = (logits_instance_low + logits_instance_high) + 0.5 * (logits_original_low + logits_original_high)

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim = 1)
        # Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)
  
        return Y_prob, Y_hat, loss

