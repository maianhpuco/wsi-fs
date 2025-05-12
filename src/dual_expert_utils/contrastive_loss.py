import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, dim=-1)
    sim_matrix = features @ features.t()
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    logits = sim_matrix / temperature
    exp_logits = torch.exp(logits) * (1 - torch.eye(len(labels), device=logits.device))
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    return -mean_log_prob_pos.mean()