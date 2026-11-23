"""Per-slide clustering for SADA sampling; not a PANTHER reproduction."""
import argparse, hashlib, json, os, time
from pathlib import Path
import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

def fingerprint(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/root/autodl-tmp/blca_features_uni_v1')
    p.add_argument('--output',default='/root/autodl-tmp/blca_cluster_cache_k32_v1')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--clusters',type=int,default=32)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    paths=sorted(Path(a.source).glob('*.h5'))
    if a.limit: paths=paths[:a.limit]
    if not paths: raise ValueError('No H5 files found')
    reports=[]
    with threadpool_limits(limits=4):
        for i,path in enumerate(paths):
            start=time.time()
            with h5py.File(path,'r') as f: x=f['features'][:]
            if x.ndim!=2 or x.shape[1]!=1024 or not len(x) or not np.isfinite(x).all():
                raise ValueError(f'Invalid features: {path}')
            sha=fingerprint(x); dest=out/(path.stem+'.npz')
            if dest.exists():
                with np.load(dest,allow_pickle=False) as c:
                    if str(c['sha256'])!=sha or int(c['requested_k'])!=a.clusters:
                        raise ValueError(f'Stale cache: {dest}')
                print(f'[{i+1}/{len(paths)}] cached {path.stem}',flush=True)
                continue
            # Clustering uses normalized features only. Model inputs remain original.
            z=x.astype(np.float32); z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            k=min(a.clusters,len(z))
            km=MiniBatchKMeans(n_clusters=k,random_state=1,n_init=3,batch_size=2048,max_iter=100,reassignment_ratio=0)
            raw=km.fit_predict(z)
            _,labels=np.unique(raw,return_inverse=True); labels=labels.astype(np.int32)
            count=np.bincount(labels); centers=[]; representatives=[]; dispersion=[]
            for g in range(len(count)):
                ix=np.flatnonzero(labels==g); center=z[ix].mean(0)
                d=((z[ix]-center)**2).sum(1)
                centers.append(x[ix].mean(0)); representatives.append(ix[np.argmin(d)])
                dispersion.append(float(d.mean()))
            with dest.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,labels=labels,counts=count,centers=np.array(centers),
                    representative_indices=np.array(representatives),dispersion=np.array(dispersion),
                    sha256=np.array(sha),requested_k=np.array(a.clusters),seed=np.array(1))
            os.replace(dest.with_suffix('.tmp'),dest)
            r=dict(slide=path.stem,n=len(x),occupied_k=len(count),smallest=int(count.min()),largest=int(count.max()),seconds=round(time.time()-start,2))
            reports.append(r); print(f'[{i+1}/{len(paths)}] '+json.dumps(r),flush=True)
    (out/'last_run.json').write_text(json.dumps(dict(status='completed',input_files=len(paths),new_files=reports),indent=2))
    print('COMPLETED',flush=True)

if __name__=='__main__': main()
