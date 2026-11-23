"""Opt-in per-slide cluster coverage sampling. Original features, full validation."""
import hashlib, os
from pathlib import Path
import h5py
import numpy as np
import torch
from utils.region_io import coverage_sample

def load_cluster_patient(dataset,slide_ids):
    xs=[]; groups=[]; offset=0
    for slide in slide_ids:
        stem=str(slide).removesuffix('.svs')
        with h5py.File(Path(dataset.data_dir)/(stem+'.h5'),'r') as f: x=f['features'][:]
        xs.append(torch.from_numpy(x))
        if dataset.sample:
            with np.load(Path(os.environ['SADA_CLUSTER_DIR'])/(stem+'.npz'),allow_pickle=False) as c:
                ids=c['labels'].astype(np.int64)
                if str(c['sha256'])!=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest():
                    raise ValueError(f'Feature fingerprint mismatch: {stem}')
                if len(ids)!=len(x) or ids.min()<0: raise ValueError(f'Invalid assignments: {stem}')
                groups.append(torch.from_numpy(ids)+offset); offset+=int(ids.max())+1
    x=torch.cat(xs)
    if not dataset.sample: return x,torch.ones(1)
    # No duplicate patches, fresh random representatives each epoch, original order.
    policy=os.environ.get('SADA_CLUSTER_POLICY','balanced')
    if policy == 'sqrt':
        from cluster_sqrt import sqrt_sample
        selected=sqrt_sample(torch.cat(groups),int(dataset.num_patches)).sort().values
    elif policy == 'balanced':
        selected=coverage_sample(torch.cat(groups),int(dataset.num_patches)).sort().values
    else:
        raise ValueError('Unknown cluster policy: '+policy)
    result=x[selected]; pad=dataset.num_patches-len(result)
    mask=torch.cat([torch.zeros(len(result)),torch.ones(pad)])
    if pad: result=torch.cat([result,result.new_zeros(pad,result.shape[1])])
    return result,mask
