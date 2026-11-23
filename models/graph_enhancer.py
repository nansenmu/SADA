"""Standard HGNN symmetric propagation, NOT the MRePath sheaf operator.
Reuses spatial/latent hyperedge groups; no N x N dense matrix or new loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


def propagate(x, incidence):
    node, edge = incidence
    n = x.shape[0]
    e = int(edge.max()) + 1
    dv = torch.bincount(node, minlength=n).to(x.dtype).clamp_min(1)
    de = torch.bincount(edge, minlength=e).to(x.dtype).clamp_min(1)
    scale = dv.rsqrt()
    z = x * scale[:, None]
    groups = x.new_zeros(e, x.shape[1]).index_add(0, edge, z[node]) / de[:, None]
    return x.new_zeros(x.shape).index_add(0, node, groups[edge]) * scale[:, None]


class GraphEnhancer(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=False)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, incidence):
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError('Expected one patient [1,N,D]')
        if incidence is None:
            raise ValueError('Graph incidence missing; install data/core adapters')
        incidence = incidence.to(device=x.device, dtype=torch.long)
        if incidence.ndim != 2 or incidence.shape[0] != 2 or not incidence.numel():
            raise ValueError('Expected nonempty incidence [2,M]')
        if incidence.min() < 0 or incidence[0].max() >= x.shape[1]:
            raise ValueError('Incidence node indices do not match WSI tokens')
        update = propagate(self.linear(self.norm(x[0])), incidence)
        update = F.dropout(F.gelu(update), p=0.1, training=self.training)
        return x + self.strength.tanh() * update.unsqueeze(0)
