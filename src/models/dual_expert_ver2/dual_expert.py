import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from models.vqgan_cache import VQGANCacheModule
from models.prior_branch import KnowledgeEnrichedPriorBranch
from models.fusion_router import FusionRouter
from utils.contrastive_loss import supervised_contrastive_loss

class LoRAAdapter(nn.Module):
    def __init__(self, vision_encoder, rank=8, feature_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lora_A = nn.Parameter(torch.randn(feature_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, feature_dim))
    
    def forward(self, x):
        features = self.vision_encoder(x)
        lora_update = features @ self.lora_A @ self.lora_B
        return features + lora_update

class DualExpert(nn.Module):
    def __init__(self, clip_model_name="ViT-B-32", vqgan_codebook=None, prototypes_per_class=10, 
                 n_classes=10, feature_dim=512, lora_rank=8, class_descriptions=None):
        super().__init__()
        # Initialize CLIP
        self.clip, _, _ = open_clip.create_model_and_transforms(clip_model_name)
        self.clip.eval()
        
        # VQ-GAN (placeholder, assumes pre-trained codebook)
        self.vqgan_codebook = vqgan_codebook  # [1000, 512]
        
        # Modules
        self.cache = VQGANCacheModule(prototypes_per_class, n_classes, feature_dim, vqgan_codebook)
        self.prior = KnowledgeEnrichedPriorBranch(feature_dim)
        self.fusion = FusionRouter()
        self.lora = LoRAAdapter(self.clip.visual, rank=lora_rank)
        
        # Class descriptions
        self.class_descriptions = class_descriptions or [
            "A dense cluster of atypical epithelial cells with nuclear pleomorphism, mitotic figures, and irregular gland formation, often accompanied by lymphocytic infiltration.",
            "Healthy glandular structures with uniform nuclei, clear cytoplasmic margins, and absence of inflammation or mitosis."
            # Add more for other classes
        ]

    def vqgan_encode(self, features):
        """Placeholder for VQ-GAN encoding (returns codebook indices)."""
        distances = torch.cdist(features, self.vqgan_codebook)
        return distances.argmin(dim=1)  # [B]

    def forward(self, patches, slide_labels):
        # Extract features with LoRA
        with torch.no_grad():
            raw_features = self.clip.visual(patches)  # [B, 512]
        features = self.lora(raw_features)  # [B, 512]
        
        # Cache branch
        cache_logits, augmented_features = self.cache.retrieve(features, 
                                                              self.prior.encode_text(self.class_descriptions)[slide_labels], 
                                                              self.vqgan_encode)
        
        # CLIP logits
        class_emb = self.prior.encode_text(self.class_descriptions)  # [C, 512]
        clip_logits = 100.0 * augmented_features @ class_emb.t()  # [B, C]
        
        # Prior branch
        slide_features = augmented_features.mean(dim=0, keepdim=True)  # [1, 512]
        prior_logits, _ = self.prior(slide_features, self.class_descriptions)
        
        # Fusion
        patch_logits, slide_logits = self.fusion(clip_logits, prior_logits, clip_logits, cache_logits)
        
        return patch_logits, slide_logits, augmented_features
    
    def build_cache(self, k_shot_patches, k_shot_labels):
        """Build cache memory bank."""
        with torch.no_grad():
            k_shot_features = self.lora(self.clip.visual(k_shot_patches))  # [K*C, 512]
        self.cache.build_memory_bank(k_shot_features, k_shot_labels, self.class_descriptions, 
                                     self.prior.encode_text, self.vqgan_encode)