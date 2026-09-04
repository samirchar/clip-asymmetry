import torch
import tqdm
from src.distributed import environment_setup
from datetime import timedelta


def rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Compute the effective rank of the embeddings using the RankMe method.
    
    Args:
        embeddings (torch.Tensor): The input embeddings of shape (N, K).
        eps (float): A small constant to prevent division by zero.
        
    Returns:
        float: The effective rank of the embeddings.
    """
    # Compute the singular values of the embeddings
    sigma = torch.linalg.svdvals(embeddings)
    
    # Normalize the singular values to get a probability distribution
    p = sigma / sigma.sum() + eps
    
    # Compute the entropy of the distribution
    entropy = - (p * torch.log(p)).sum()
    
    # Return the effective rank
    return torch.exp(entropy).item()

def alignment(image_emb: torch.Tensor, text_emb: torch.Tensor, alpha: float = 2.0) -> float:
    """Alignment: mean ‖x_i − y_i‖^alpha over positive pairs (lower is better)."""
    return (image_emb - text_emb).norm(p=2, dim=1).pow(alpha).mean().item()


def uniformity(embeddings: torch.Tensor, t: float = 2.0) -> float:
    """Uniformity: log of average pairwise Gaussian potential (lower is better)."""
    return torch.pdist(embeddings, p=2).pow(2).mul(-t).exp().mean().log().item()
