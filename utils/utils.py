import torch
import numpy as np
import torch.nn.functional as F


def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)

def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def entropy_loss(score, reduction='mean'):
    p = F.softmax(score, dim=1)
    NH = torch.sum(p * torch.log(p + 1e-7), dim=1)
    if reduction == 'none':
        return NH
    elif reduction == 'mean':
        NH = torch.mean(NH)
    return NH