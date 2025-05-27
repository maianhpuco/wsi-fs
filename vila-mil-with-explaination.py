import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn import MultiheadAttention


class SimplePatchEncoder(nn.Module):
    def __init__(self, num_proposals=100):
        super().__init__()
        self.encoder = resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove classification head
        self.num_proposals = num_proposals
        self.feature_dim = 512

    def forward(self, patches):
        B, N, C, H, W = patches.shape
        patches_flat = patches.view(-1, C, H, W)  # [B*N, C, H, W]
        with torch.no_grad():
            features_flat = self.encoder(patches_flat)  # [B*N, 512]
        features = features_flat.view(B, N, -1)  # [B, N, 512]

        # Select top-k using L2 norm as importance score
        scores = torch.norm(features, dim=-1)  # [B, N]
        topk_idx = torch.topk(scores, self.num_proposals, dim=1).indices  # [B, K]
        selected_features = torch.gather(
            features, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        )  # [B, K, 512]

        return selected_features


class VisionLanguageTransformer(nn.Module):
    def __init__(self, vision_dim=512, text_model_name="t5-small", num_heads=8):
        super().__init__()
        self.vision_projection = nn.Linear(vision_dim, 512)
        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(text_model_name)
        self.cross_attention = MultiheadAttention(embed_dim=512, num_heads=num_heads)
        self.cls_head = nn.Linear(512, 2)

        self.prompt = "Describe the histopathological features of this slide:"
        self.prompt_ids = self.tokenizer.encode(self.prompt, return_tensors="pt")

    def forward(self, patch_features, generate_explanation=True):
        B, M, _ = patch_features.shape
        patch_features = self.vision_projection(patch_features)

        attn_output, _ = self.cross_attention(
            patch_features, patch_features, patch_features
        )
        pooled = torch.sum(attn_output * patch_features, dim=1)
        logits = self.cls_head(pooled)

        explanations = []
        attention_maps = []
        if generate_explanation:
            prompt_ids = self.prompt_ids.to(patch_features.device).expand(B, -1)
            outputs = self.text_model(
                input_ids=prompt_ids,
                encoder_hidden_states=patch_features,
                output_attentions=True,
                return_dict=True
            )
            generated_ids = self.text_model.generate(
                input_ids=prompt_ids,
                encoder_hidden_states=patch_features,
                max_length=100
            )
            for ids in generated_ids:
                explanation = self.tokenizer.decode(ids, skip_special_tokens=True)
                explanations.append(explanation)

            attention_maps = outputs.cross_attentions[-1]  # [B, heads, text_len, patches]

        return logits, explanations, attention_maps


class VLM_MIL_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SimplePatchEncoder(config.num_proposals)
        self.vlm = VisionLanguageTransformer(vision_dim=512)

        self.loss_ce = nn.CrossEntropyLoss()
        self.lambda_class = config.lambda_class
        self.lambda_ground = config.lambda_ground

    def forward(self, patches, labels=None, generate_explanation=True):
        patch_features = self.encoder(patches)  # [B, num_proposals, 512]
        logits, explanations, attention = self.vlm(patch_features, generate_explanation)

        loss = 0
        if labels is not None:
            loss_class = self.loss_ce(logits, labels)
            loss += self.lambda_class * loss_class

            if generate_explanation and attention is not None:
                image_to_text = attention.transpose(-2, -1)
                loss_ground = F.kl_div(
                    F.log_softmax(attention, dim=-1),
                    F.softmax(image_to_text, dim=-1),
                    reduction='batchmean'
                )
                loss += self.lambda_ground * loss_ground

        return logits, explanations, loss


class Config:
    def __init__(self):
        self.num_proposals = 100
        self.lambda_class = 1.0
        self.lambda_ground = 0.1


def main():
    config = Config()
    model = VLM_MIL_Model(config).cuda()
    patches = torch.randn(1, 1000, 3, 256, 256).cuda()
    labels = torch.tensor([1], dtype=torch.long).cuda()
    logits, explanations, loss = model(patches, labels)
    print("Logits:", logits)
    print("Explanations:", explanations)
    print("Loss:", loss)


if __name__ == "__main__":
    main()
