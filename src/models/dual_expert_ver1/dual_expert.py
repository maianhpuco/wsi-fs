
class DualExpert(nn.Module):
    def __init__(self, clip_model, vqgan_model, prototypes_per_class=10, n_classes=10):
        super().__init__()
        self.clip = clip_model
        self.vqgan = vqgan_model
        self.cache = VQGANCacheModule(prototypes_per_class, n_classes, vqgan_codebook=vqgan_model.codebook)
        self.prior = PriorBranch()
        self.fusion = FusionRouter()
        self.lora = LoRAAdapter(self.clip.vision_encoder)  # Placeholder for LoRA implementation
    
    def forward(self, patches, slide_labels, class_prompts):
        # Extract features with LoRA
        features = self.lora(self.clip.encode_image(patches))  # [B, 512]
        
        # Cache branch
        cache_logits, augmented_features = self.cache(features, slide_labels, class_prompts, 
                                                    self.clip.encode_text, self.vqgan.encode)
        
        # CLIP logits
        text_emb = self.clip.encode_text(class_prompts)  # [C, 512]
        clip_logits = 100.0 * augmented_features @ text_emb.t()  # [B, C]
        
        # Prior branch
        slide_features = augmented_features.mean(dim=0, keepdim=True)  # [1, 512]
        prior_logits, adapted_text_emb = self.prior(class_prompts, slide_features, self.clip.encode_text)
        
        # Fusion
        patch_logits, slide_logits = self.fusion(clip_logits, prior_logits, clip_logits, cache_logits)
        
        return patch_logits, slide_logits, augmented_features
    
    def train_step(self, patches, slide_labels, class_prompts, optimizer):
        patch_logits, slide_logits, _ = self.forward(patches, slide_labels, class_prompts)
        
        # Losses
        loss_ce = F.cross_entropy(slide_logits, slide_labels)
        loss_supcon = supervised_contrastive_loss(augmented_features, slide_labels)
        loss = loss_ce + 0.5 * loss_supcon
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()