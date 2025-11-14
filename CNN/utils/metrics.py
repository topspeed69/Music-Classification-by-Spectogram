"""
Utility functions for training and evaluation
"""

import torch
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors or matrices.
    
    Args:
        a: Tensor [N, D] or [D]
        b: Tensor [M, D] or [D]
    Returns:
        Similarity scores [N, M] or scalar
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    
    return torch.mm(a_norm, b_norm.t())


def compute_embeddings_similarity(embeddings, query_embedding, top_k=10):
    """
    Find most similar embeddings to query.
    
    Args:
        embeddings: Database embeddings [N, D]
        query_embedding: Query embedding [D] or [1, D]
        top_k: Number of top results to return
    Returns:
        (indices, similarities) - top-k indices and similarity scores
    """
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
    
    similarities = cosine_similarity(query_embedding, embeddings).squeeze()
    
    top_k = min(top_k, len(similarities))
    top_similarities, top_indices = torch.topk(similarities, top_k)
    
    return top_indices.cpu().numpy(), top_similarities.cpu().numpy()
