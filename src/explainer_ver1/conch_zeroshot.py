import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # research/
_path = os.path.abspath(os.path.join(current_dir, "../..", 'src/externals/CONCH'))
sys.path.append(_path) 
 
# Import CONCH model and tokenizer from the cloned repository
from conch.model import CONCHVisionLanguageModel
from conch.tokenizer import CONCHTokenizer

class CONCH_ZeroShot_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(CONCH_ZeroShot_Model, self).__init__()
        self.device = config.device
        self.num_classes = num_classes
        
        # Load CONCH model from pretrained weights
        self.conch_model = CONCHVisionLanguageModel.from_pretrained(
            config.weight_path, device_map=self.device
        )
        self.tokenizer = CONCHTokenizer()
        self.text_prompt = config.text_prompt
        
        # Attention mechanism for aggregating patch features
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh()
        ).to(self.device)
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Sigmoid()
        ).to(self.device)
        self.attention_weights = nn.Linear(self.D, self.K).to(self.device)

    def encode_text(self, prompts):
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.conch_model.encode_text(tokenized['input_ids'], tokenized['attention_mask'])
        return text_features

    def forward(self, x_s, coord_s, x_l, coords_l, label=None):
        # Process low magnification patches
        M = x_s.float()  # Shape: [batch, num_patches, input_size]
        A_V = self.attention_V(M)
        A_U = self.attention_U(M)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        image_features_low = torch.mm(A, M)  # Aggregated low-mag features

        # Process high magnification patches
        M_high = x_l.float()
        A_V_high = self.attention_V(M_high)
        A_U_high = self.attention_U(M_high)
        A_high = self.attention_weights(A_V_high * A_U_high)
        A_high = torch.transpose(A_high, 1, 0)
        A_high = F.softmax(A_high, dim=1)
        image_features_high = torch.mm(A_high, M_high)  # Aggregated high-mag features

        # Encode text prompts
        text_features = self.encode_text(self.text_prompt)
        
        # Zero-shot prediction: compute logits for low and high magnification
        logits_low = image_features_low @ text_features[:self.num_classes].T
        logits_high = image_features_high @ text_features[self.num_classes:].T
        logits = logits_low + logits_high

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)

        loss = None
        if label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, label)

        return Y_prob, Y_hat, loss