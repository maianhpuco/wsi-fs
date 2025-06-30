import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionMIL, self).__init__()
        # Attention network: MLP to compute attention weights
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, patch_features):
        # patch_features: (num_patches, feature_dim)
        # Compute attention weights
        attention_scores = self.attention(patch_features)  # (num_patches, 1)
        attention_weights = F.softmax(attention_scores, dim=0)  # (num_patches, 1)
        
        # Compute slide-level representation
        slide_representation = torch.sum(attention_weights * patch_features, dim=0)  # (feature_dim,)
        
        return slide_representation, attention_weights

def get_top_concepts(slide_representation, concept_names, top_k=5):
    # slide_representation: (feature_dim,) tensor of aggregated concept scores
    # concept_names: list of concept names corresponding to feature_dim
    # Sort concepts by their contribution (absolute value for robustness)
    concept_scores = slide_representation.abs()
    top_k_indices = torch.argsort(concept_scores, descending=True)[:top_k]
    top_concepts = [(concept_names[i], concept_scores[i].item()) for i in top_k_indices]
    return top_concepts

# Example usage
if __name__ == "__main__":
    # Simulated data
    num_patches = 1111
    num_concepts = 10
    num_classes = 4  # KICH, KIRP, KIRC, non-tumor
    feature_dim = num_concepts * num_classes  # Concatenated concept scores
    patch_features = torch.randn(num_patches, feature_dim)  # Simulated patch-level concept scores
    
    # Concept names (e.g., 10 concepts per class + 1 non-tumor + shared tumor concepts)
    concept_names = [f"concept_{c}_class_{k}" for k in range(num_classes) for c in range(num_concepts)]
    
    # Initialize model
    model = AttentionMIL(input_dim=feature_dim)
    
    # Forward pass
    slide_representation, attention_weights = model(patch_features)
    
    # Get top-K concepts
    top_k = 5
    top_concepts = get_top_concepts(slide_representation, concept_names, top_k)
    
    # Print results
    print("Slide-level representation shape:", slide_representation.shape)
    print(f"Top {top_k} representative concepts:")
    for concept, score in top_concepts:
        print(f"Concept: {concept}, Score: {score:.4f}")
    
    # Optional: Identify top patches for top concepts
    top_patch_indices = torch.argsort(attention_weights.squeeze(), descending=True)[:10]
    print("Top 10 patches by attention weight:", top_patch_indices.tolist())