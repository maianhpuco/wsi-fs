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

# Use TOP's PromptLearner and TextEncoder
sys.path.append(os.path.abspath(os.path.join(current_dir, "../..", 'src/externals/TOP/models')))
from learnable_prompt import PromptLearner, TextEncoder

 
 
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
        
        # Only use TextEncoder for instance-level text prototypes
        self.text_encoder = TextEncoder(clip_model.float()).to(self.device)

        self.norm = nn.LayerNorm(config.input_size).to(self.device)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1).to(self.device)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1).to(self.device)

        self.learnable_image_center = nn.Parameter(
            torch.empty(config.prototype_number, 1, config.input_size, device=self.device)
        )
        trunc_normal_(self.learnable_image_center, std=.02)

        # TOP-style instance-level prompt learning for text prototypes
        # Define comprehensive TCGA renal cell carcinoma-specific text prototypes - 20 per class
        # Based on TOP lung methodology adapted for renal classification
        instance_prompt_names = [
            # Papillary RCC (KIRP) - 20 extensive detailed descriptions
            "A WSI of KIRP with papillary growth pattern and fibrovascular cores",
            "A WSI of papillary renal carcinoma with complex architectural features and nuclei arranged in layers", 
            "A WSI of PRCC with multiple papillary formations and pinkish coloration",
            "A WSI of KIRP with papillary architecture and varied architectural patterns",
            "A WSI of papillary carcinoma with tubular and papillary growth patterns",
            "A WSI of PRCC with foamy macrophages and inflammatory infiltrates",
            "A WSI of papillary RCC with nuclear pseudoinclusions and psammoma bodies",
            "A WSI of KIRP with heterogeneous cytoplasm and basement membrane thickening",
            "A WSI of papillary carcinoma with nested growth pattern and clear cytoplasm",
            "A WSI of PRCC with oncocytic features and eosinophilic cytoplasm",
            "A WSI of papillary RCC type 1 with small uniform nuclei and scanty cytoplasm",
            "A WSI of KIRP type 2 with large nuclei and abundant eosinophilic cytoplasm",
            "A WSI of papillary carcinoma with microcystic and solid growth patterns",
            "A WSI of PRCC with cholesterol clefts and hemosiderin deposits",
            "A WSI of papillary RCC with collecting duct-like features and high nuclear grade",
            "A WSI of KIRP with extensive fibrosis and desmoplastic stroma",
            "A WSI of papillary carcinoma with signet ring cell morphology",
            "A WSI of PRCC with rhabdoid features and aggressive behavior",
            "A WSI of papillary RCC with warty dysplasia and multifocal growth",
            "A WSI of KIRP with adenomatous hyperplasia and microscopic foci",
            
            # Clear Cell RCC (KIRC) - 20 extensive detailed descriptions  
            "A WSI of KIRC with clear cytoplasm and round nuclei",
            "A WSI of clear cell renal carcinoma with prominent nucleoli and rich vascularity",
            "A WSI of CCRCC with pale yellowish color and homogeneous texture",
            "A WSI of KIRC with irregular blood vessels and intratumoral septa",
            "A WSI of clear cell RCC with glycogen-rich cytoplasm and delicate capillary network",
            "A WSI of CCRCC with solid and alveolar growth patterns",
            "A WSI of clear cell carcinoma with hemorrhage and necrosis",
            "A WSI of KIRC with nuclear pleomorphism and high-grade features",
            "A WSI of clear cell RCC with cystic degeneration and calcification",
            "A WSI of CCRCC with sarcomatoid differentiation and spindle cells",
            "A WSI of clear cell carcinoma with VHL gene mutation and hypoxia markers",
            "A WSI of KIRC with extensive clear cell morphology and lipid accumulation",
            "A WSI of clear cell RCC with tubular growth pattern and basement membrane",
            "A WSI of CCRCC with multilocular cystic features and benign behavior",
            "A WSI of clear cell carcinoma with rhabdoid differentiation and poor prognosis",
            "A WSI of KIRC with granular cell change and eosinophilic transformation",
            "A WSI of clear cell RCC with papillary features and mixed morphology",
            "A WSI of CCRCC with collecting duct admixture and transitional zones",
            "A WSI of clear cell carcinoma with oncocytoma-like features and hybrid tumors",
            "A WSI of KIRC with extensive fibrosis and treatment-related changes",
            
            # Chromophobe RCC (KICH) - 20 extensive detailed descriptions
            "A WSI of KICH with eosinophilic cytoplasm and perinuclear halos",
            "A WSI of chromophobe renal carcinoma with perinuclear vacuolization and multiple nucleoli", 
            "A WSI of CRCC with pale tan coloration and distinct cell borders",
            "A WSI of KICH with binucleation and finely granular chromatin",
            "A WSI of chromophobe RCC with plant-like cell borders and prominent nucleoli",
            "A WSI of CRCC with solid and nested growth patterns",
            "A WSI of chromophobe carcinoma with oncocytic features and abundant mitochondria",
            "A WSI of KICH with minimal nuclear pleomorphism and uniform cells",
            "A WSI of chromophobe RCC with hyalinized stroma and fibrosis",
            "A WSI of CRCC with typical and eosinophilic variant morphology",
            "A WSI of chromophobe carcinoma with BHD syndrome association and multifocal growth",
            "A WSI of KICH with renal oncocytosis and bilateral involvement",
            "A WSI of chromophobe RCC with hybrid oncocytic features and intermediate morphology",
            "A WSI of CRCC with papillary growth pattern and architectural overlap",
            "A WSI of chromophobe carcinoma with sarcomatoid transformation and dedifferentiation",
            "A WSI of KICH with extensive calcification and ossification",
            "A WSI of chromophobe RCC with cystic degeneration and multicystic appearance",
            "A WSI of CRCC with inflammatory infiltrate and reactive changes",
            "A WSI of chromophobe carcinoma with clear cell change and morphological variation",
            "A WSI of KICH with neuroendocrine differentiation and synaptophysin expression"
        ]
        
        # Instance-level prompt learner for text prototypes (following TOP approach)
        # Use simple initialization to avoid tensor size mismatch
        self.instance_prompt_learner = PromptLearner(
            n_ctx=16, 
            ctx_init="",  # Empty string for random initialization
            all_ctx_trainable=True, 
            csc=True,  # Use class-specific context for instance-level
            classnames=["KIRP", "KIRC", "KICH"],  # 3 RCC subtypes - uniform length
            clip_model='RN50', 
            p_drop_out=0.1
        ).to(self.device)
        
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

        M = x_s.float()
        M_high = x_l.float()
        
        # TOP-style text prototype integration
        # Encode instance-level text prototypes through text encoder
        instance_prompts = self.instance_prompt_learner().to(device)
        instance_tokenized_prompts = self.instance_prompt_learner.tokenized_prompts.to(device)
        instance_text_features = self.text_encoder(instance_prompts, instance_tokenized_prompts).to(device)
        instance_text_features = instance_text_features / instance_text_features.norm(dim=-1, keepdim=True)
        
        # Compute text-image prototype similarity for low resolution features
        # TOP's PromptLearner outputs 3 class-specific text prototypes
        M_proj = self.text_proto_projection(M)
        text_proto_similarity_low = M_proj @ instance_text_features.t()  # [N_patches, 3_classes]
        text_proto_weights_low = F.softmax(text_proto_similarity_low, dim=-1)
        text_enhanced_features_low = text_proto_weights_low @ instance_text_features  # [N_patches, embed_dim]
        
        # Compute text-image prototype similarity for high resolution features  
        M_high_proj = self.text_proto_projection(M_high)
        text_proto_similarity_high = M_high_proj @ instance_text_features.t()  # [N_patches, 3_classes]
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

        # Use only instance text prototypes for final classification
        # TOP's PromptLearner with csc=True outputs one text feature per class (3 total)
        class_text_features = instance_text_features  # [3 classes, feature_dim] already aggregated by TOP
        
        # Compute logits using only class-specific text prototypes
        logits_low = image_features_low @ class_text_features.T
        logits_high = image_features_high @ class_text_features.T
        
        # Combine logits from both scales
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim = 1)
        # Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1].squeeze(1)
  
        return Y_prob, Y_hat, loss

