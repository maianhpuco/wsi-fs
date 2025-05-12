import torch
import torch.nn as nn
import torch.nn.functional as F

class VQGANCacheModule(nn.Module):
    def __init__(self, prototypes_per_class=10, n_classes=10, feature_dim=512, vqgan_codebook=None, codebook_size=1000):
        super().__init__()
        self.prototypes_per_class = prototypes_per_class
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.vqgan_codebook = vqgan_codebook  # [1000, 512]
        self.codebook_size = codebook_size

        # Memory bank
        self.memory_bank = {
            'visual_words': torch.empty(prototypes_per_class * n_classes, feature_dim),
            'class_embeddings': torch.empty(prototypes_per_class * n_classes, feature_dim),
            'token_prompts': nn.Parameter(torch.randn(prototypes_per_class * n_classes, feature_dim)),  # Learnable prompts
            'importance_scores': torch.empty(prototypes_per_class * n_classes)
        }

        # Hyperparameters
        self.gamma = 0.7  # Visual-semantic balance
        self.tau = 0.1    # Attention temperature
        self.alpha = 10.0 # Cache logit scaling

    def build_memory_bank(self, k_shot_features, k_shot_labels, class_descriptions, text_encoder, vqgan_encoder):
        """Build memory bank from K-shot patches."""
        # Tokenize K-shot patches
        with torch.no_grad():
            word_indices = vqgan_encoder(k_shot_features)  # [K * C] -> Nearest codebook indices

        for c in range(self.n_classes):
            class_features = k_shot_features[k_shot_labels == c]  # [K, 512]
            class_word_indices = word_indices[k_shot_labels == c]  # [K]

            # Count frequency of visual words
            word_counts = torch.bincount(class_word_indices, minlength=self.codebook_size)
            
            # Compute importance scores (distance-based)
            distances = torch.cdist(class_features, self.vqgan_codebook).min(dim=1)[0]
            importance = distances / class_features.shape[0]
            
            # Select top visual words
            top_indices = word_counts.argsort(descending=True)[:self.prototypes_per_class]
            prototypes = self.vqgan_codebook[top_indices]
            
            # Store in memory bank
            start_idx = c * self.prototypes_per_class
            self.memory_bank['visual_words'][start_idx:start_idx + self.prototypes_per_class] = prototypes
            self.memory_bank['class_embeddings'][start_idx:start_idx + self.prototypes_per_class] = text_encoder([class_descriptions[c]])
            self.memory_bank['importance_scores'][start_idx:start_idx + self.prototypes_per_class] = importance[top_indices]

    def augment_features(self, features, word_indices):
        """Augment features with learnable token context prompts."""
        token_prompts = self.memory_bank['token_prompts']  # [P * C, 512]
        selected_prompts = token_prompts[word_indices]  # [B, 512]
        return features + selected_prompts  # [B, 512]

    def retrieve(self, test_features, predicted_class_emb, vqgan_encoder):
        """Retrieve cache logits for test patches."""
        # Tokenize test features
        with torch.no_grad():
            word_indices = vqgan_encoder(test_features)  # [B] -> Nearest codebook indices
            test_words = self.vqgan_codebook[word_indices]  # [B, 512]

        # Augment test features
        augmented_features = self.augment_features(test_features, word_indices)  # [B, 512]

        # Compute similarities
        visual_sim = F.cosine_similarity(test_words.unsqueeze(1), self.memory_bank['visual_words'], dim=-1)  # [B, P*C]
        semantic_sim = F.cosine_similarity(predicted_class_emb.unsqueeze(1), self.memory_bank['class_embeddings'], dim=-1)  # [B, P*C]
        combined_sim = self.gamma * visual_sim + (1 - self.gamma) * semantic_sim
        
        # Attention
        attention = F.softmax(combined_sim / self.tau, dim=-1)  # [B, P*C]
        
        # Cache logits
        cache_logits = attention @ self.memory_bank['class_embeddings']  # [B, 512]
        return cache_logits, augmented_features