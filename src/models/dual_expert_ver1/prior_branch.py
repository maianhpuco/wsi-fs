from torch import nn 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class PriorBranch(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.text_adapter = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
    
    def forward(self, class_prompts, slide_features, clip_text_encoder):
        text_emb = clip_text_encoder(class_prompts)  # [C, 512]
        adapted_emb = self.text_adapter(text_emb)   # [C, 512]
        prior_logits = 100.0 * slide_features @ adapted_emb.t()  # [B, C]
        return prior_logits, adapted_emb
    


class KnowledgeEnrichedPriorBranch(nn.Module):
    def __init__(self, feature_dim=512, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        # Load biomedical text encoder (PubMedBERT)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Learnable adapter
        self.adapter = nn.Sequential(
            nn.Linear(768, 128),  # PubMedBERT output is 768-dim
            nn.ReLU(),
            nn.Linear(128, feature_dim)  # Align with CLIP feature space
        )
        
        # Hyperparameters
        self.tau = 0.1  # Softmax temperature

    def encode_text(self, descriptions):
        """Encode class descriptions into embeddings."""
        # Tokenize descriptions
        inputs = self.tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_emb = outputs.last_hidden_state.mean(dim=1)  # [C, 768]
        
        # Apply adapter
        adapted_emb = self.adapter(text_emb)  # [C, 512]
        return adapted_emb

    def forward(self, slide_features, class_descriptions):
        """Compute prior logits for slide-level classification."""
        # Encode class descriptions
        class_emb = self.encode_text(class_descriptions)  # [C, 512]
        
        # Normalize features and embeddings
        slide_features = F.normalize(slide_features, dim=-1)  # [B, 512]
        class_emb = F.normalize(class_emb, dim=-1)  # [C, 512]
        
        # Compute cosine similarity
        similarities = slide_features @ class_emb.t()  # [B, C]
        
        # Softmax for prior logits
        prior_logits = F.softmax(similarities / self.tau, dim=-1)  # [B, C]
        
        return prior_logits, class_emb

    def train_step(self, slide_features, slide_labels, class_descriptions, optimizer):
        """Training step for fine-tuning the adapter."""
        prior_logits, _ = self.forward(slide_features, class_descriptions)
        
        # Cross-entropy loss
        loss = F.cross_entropy(prior_logits, slide_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()