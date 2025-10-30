# src/utils/losses.py
import torch.nn.functional as F
import torch

def info_nce(z2d, z3d, temp=0.2):
    """
    Stable InfoNCE for cross-modal alignment.
    z2d, z3d: [B, D] (unnormalized or normalized). Function normalizes internally.
    """
    z2 = F.normalize(z2d, dim=1)
    z3 = F.normalize(z3d, dim=1)
    B = z2.size(0)
    z = torch.cat([z2, z3], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t()) / temp
    mask = torch.eye(2*B, device=sim.device).bool()
    sim.masked_fill_(mask, -1e9)
    labels = torch.arange(2*B, device=sim.device)
    pos_idx = (labels + B) % (2*B)
    loss = F.cross_entropy(sim, pos_idx)
    return loss
