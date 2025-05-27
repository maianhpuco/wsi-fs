import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.nn import MultiheadAttention


class RegionProposalNetwork(nn.Module):
    def __init__(self, num_proposals=100):
        super().__init__()
        self.rpn = fasterrcnn_resnet50_fpn(pretrained=True)
        self.num_proposals = num_proposals
        for param in self.rpn.backbone.parameters():
            param.requires_grad = False
        self.feature_dim = 256  # Assumed FPN output feature dim

    def forward(self, patches):
        B, N, C, H, W = patches.shape
        patches_flat = patches.view(-1, C, H, W)
        # outputs = self.rpn(patches_flat)
        self.rpn.eval()  # Set RPN to eval mode to suppress training requirement
        with torch.no_grad():  # Disable gradients for inference
            outputs = self.rpn(patches_flat)
 
        selected_features = []
        for i in range(B):
            start, end = i * N, (i + 1) * N
            boxes = outputs['boxes'][start:end]
            scores = outputs['scores'][start:end]
            top_k = torch.topk(scores, min(self.num_proposals, scores.shape[0])).indices
            features = outputs['features'][start:end][top_k]
            selected_features.append(features)

        return torch.stack(selected_features)  # [B, num_proposals, feature_dim]


class VisionLanguageTransformer(nn.Module):
    def __init__(self, vision_dim=256, text_model_name="t5-small", num_heads=8):
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
        self.rpn = RegionProposalNetwork(config.num_proposals)
        self.vlm = VisionLanguageTransformer(vision_dim=256)

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_lm = nn.CrossEntropyLoss(ignore_index=0)
        self.lambda_class = config.lambda_class
        self.lambda_lm = config.lambda_lm
        self.lambda_ground = config.lambda_ground

    def forward(self, patches, labels=None, generate_explanation=True):
        patch_features = self.rpn(patches)  # [B, num_proposals, vision_dim]
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
        self.lambda_lm = 0.5
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
