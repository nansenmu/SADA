"""Capacity-constrained sqrt-size quotas and random sampling without replacement."""
import numpy as np
import torch

def sqrt_sample(ids, limit, generator=None):
    if limit < 1: raise ValueError('Positive budget required')
    if len(ids)<=limit: return torch.arange(len(ids))
    _, mapped=torch.unique(ids,sorted=True,return_inverse=True)
    counts=torch.bincount(mapped).numpy()
    weights=np.sqrt(counts.astype(float))
    quota=np.zeros(len(counts),dtype=np.int64)
    remaining=int(limit)
    active=np.ones(len(counts),dtype=bool)
    while remaining:
        ix=np.flatnonzero(active)
        target=remaining*weights[ix]/weights[ix].sum()
        saturated=target>=counts[ix]
        if saturated.any():
            full=ix[saturated];quota[full]=counts[full]
            remaining-=int(counts[full].sum());active[full]=False
            continue
        base=np.floor(target).astype(np.int64)
        quota[ix]=base
        extra=remaining-int(base.sum())
        # Random tie-breaking avoids systematic preference for cluster IDs.
        tie=torch.randperm(len(ix),generator=generator).numpy()
        order=tie[np.argsort(-(target-base)[tie],kind='stable')]
        quota[ix[order[:extra]]]+=1
        break
    order=torch.randperm(len(ids),generator=generator)
    order=order[torch.argsort(mapped[order],stable=True)]
    buckets=torch.split(order,counts.tolist())
    selected=torch.cat([bucket[:int(q)] for bucket,q in zip(buckets,quota)])
    assert len(selected)==limit
    return selected
