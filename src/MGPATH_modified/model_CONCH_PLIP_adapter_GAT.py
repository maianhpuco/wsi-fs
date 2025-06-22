import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from transformers import CLIPModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# === Import local modules ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))
from MGPATH_modified import PLIPTextEncoder, PromptLearner, Adapter


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_().mul_(std * math.sqrt(2.)).add_(mean).clamp_(min=a, max=b)
        return tensor

class PLIPTextOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(">> Using PLIP for Text Only")
        self.text_model = CLIPModel.from_pretrained(config['text_encoder_ckpt_dir'])
        self.TextMLP = Adapter(image=False, hidden=512)
        self.temperature = nn.Parameter(torch.tensor([np.log(1 / 0.02)]), requires_grad=True)

    def to(self, device):
        super().to(device)
        self.text_model = self.text_model.to(device)
        self.TextMLP = self.TextMLP.to(device)
        self.temperature = self.temperature.to(device)
        return self

class CONCH_PLIP_adapter_GAT(nn.Module):
    def __init__(self, config, num_classes=3):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config['input_size']
        self.D = config['hidden_size']
        self.device = config['device']
 
        self.K = 1

        # === Text encoder from PLIP ===
        self.clip_model = PLIPTextOnly(config)
        self.text_encoder = PLIPTextEncoder(self.clip_model).to(self.device)
        self.prompt_learner = PromptLearner(
            config['text_prompt'], 
            self.clip_model, 
            config['text_encoder_ckpt_dir']).to(self.device)

        # === Adapter to project CONCH features ===
        self.adapter = Adapter(image=True, hidden=512).to(self.device)

        # === Freeze most of text encoder ===
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.proj.parameters():
            param.requires_grad = True

        # === Learnable Prototypes ===
        self.learnable_image_center = nn.Parameter(
            torch.empty(config.prototype_number, 1, config.input_size, device=self.device)
        )
        trunc_normal_(self.learnable_image_center, std=.02)

        # === Attention Modules ===
        self.norm = nn.LayerNorm(self.L).to(self.device)
        self.cross_attention_1 = MultiheadAttention(embed_dim=self.L, num_heads=1, batch_first=True).to(self.device)
        self.cross_attention_2 = MultiheadAttention(embed_dim=self.L, num_heads=1, batch_first=True).to(self.device)

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)
        
        # Also move attention modules to device
        self.attention_V.to(self.device)
        self.attention_U.to(self.device)
        self.attention_weights.to(self.device)
  
    def forward(self, x_s, coord_s, x_l, coord_l, label):
        device = x_s.device
        print("========> device", device)
        # === Text prompt encoding ===
        prompts = self.prompt_learner().to(device)
        tokenized = {
            "input_ids": self.prompt_learner.tokenized_prompts["input_ids"].to(device),
            "attention_mask": self.prompt_learner.tokenized_prompts["attention_mask"].to(device),
        }
        

        # tokenized = self.prompt_learner.tokenized_prompts.to(device)
        text_features, _ = self.text_encoder(prompts, tokenized['attention_mask'], tokenized['input_ids'])
        text_features = text_features.contiguous().view(1, -1, self.L).squeeze(0)

        # === Image feature projection (CONCH -> Adapter) ===
        M = self.adapter(x_s.squeeze(0).float())      # Low-res features
        M_high = self.adapter(x_l.squeeze(0).float()) # High-res features
        if M.dim() == 2:
            M = M.unsqueeze(0)  # [1, num_patches, L] 
            
        if M_high.dim() == 2:
            M_high = M_high.unsqueeze(0)  # [1, num_patches, L]
        # learnable_center = self.learnable_image_center.to(x_s.device)

        # comp_low, _ = self.cross_attention_1(learnable_center, M, M)
        # comp_low = self.norm(comp_low + learnable_center)

        # comp_high, _ = self.cross_attention_1(learnable_center, M_high, M_high)
        # comp_high = self.norm(comp_high + learnable_center)  
        # === Cross attention: Image → Prototypes ===
        query_low = self.learnable_image_center.permute(1, 0, 2)  # [1, P, L]
        comp_low, _ = self.cross_attention_1(query_low, M, M)
        comp_low = self.norm(comp_low + query_low)

        query_high = self.learnable_image_center.permute(1, 0, 2)
        comp_high, _ = self.cross_attention_1(query_high, M_high, M_high)
        comp_high = self.norm(comp_high + query_high)

        # === Prototype attention pooling ===
        A = F.softmax(self.attention_weights(
            self.attention_V(comp_low.squeeze()) * self.attention_U(comp_low.squeeze())), dim=0).T
        image_features_low = A @ comp_low.squeeze()

        A_high = F.softmax(self.attention_weights(
            self.attention_V(comp_high.squeeze()) * self.attention_U(comp_high.squeeze())), dim=0).T
        image_features_high = A_high @ comp_high.squeeze()

        # === Cross-attend prototypes with text ===
        text_low = text_features[:self.num_classes]
        context_low = torch.cat([comp_low, M], dim=1)
        text_low = text_low.unsqueeze(0)  # [1, C, L]
        refined_text_low, _ = self.cross_attention_2(text_low, context_low, context_low)
        text_low = refined_text_low + text_low 
        # context_low = torch.cat([comp_low.squeeze(), M], dim=0)
        # refined_text_low, _ = self.cross_attention_2(text_low.unsqueeze(1), context_low, context_low)
        # text_low = refined_text_low.squeeze() + text_low

        text_high = text_features[self.num_classes:]
        context_high = torch.cat([comp_high, M], dim=1)
        text_high = text_high.unsqueeze(0)
        refined_text_high, _ = self.cross_attention_2(text_high, context_high, context_high)
        text_high = refined_text_high + text_high
        
        # context_high = torch.cat([comp_high.squeeze(), M_high], dim=0)
        # refined_text_high, _ = self.cross_attention_2(text_high.unsqueeze(1), context_high, context_high)
        # text_high = refined_text_high.squeeze() + text_high
        text_low = text_low.squeeze(0)
        text_high = text_high.squeeze(0)
 
        # === Compute logits ===
        logits_low = image_features_low @ text_low.T
        logits_high = image_features_high @ text_high.T
        logits = logits_low + logits_high

        # === Loss and prediction ===
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = Y_prob.argmax(dim=1)
        return Y_prob, Y_hat, loss
