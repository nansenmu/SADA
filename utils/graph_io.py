"""Trusted local MRePath PyG graphs -> sampled features and incidence indices."""
import hashlib
import os
from pathlib import Path
import torch


def load_trusted(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def remap_edges(edges, selected, n):
    mapping = torch.full((n,), -1, dtype=torch.long)
    mapping[selected] = torch.arange(len(selected))
    mapped = mapping[edges]
    return mapped[:, (mapped >= 0).all(dim=0)]


def incidence_from_edges(edges_list, n):
    # Each original source and its retained neighbours form one hyperedge.
    # Singleton groups are retained for nodes whose neighbours were not sampled.
    pairs = []
    for group, edges in enumerate(edges_list):
        centers = torch.arange(n)
        pairs.append(torch.stack((centers, centers + group * n)))
        pairs.append(torch.stack((edges[1], edges[0] + group * n)))
    return torch.unique(torch.cat(pairs, dim=1), dim=1)


def load_patient_graph(dataset, slide_ids):
    if os.environ.get('SADA_WSI_MODE') == 'cluster_sample':
        if os.environ.get('SADA_GRAPH_ENABLED') != '0' or os.environ.get('SADA_MODEL_VARIANT', 'single') != 'single':
            raise ValueError('Cluster sampling requires original SADA: graph off, variant single')
        from cluster_io import load_cluster_patient
        return load_cluster_patient(dataset, slide_ids)
    if os.environ.get('SADA_MODEL_VARIANT') == 'joint_hypergraph':
        from utils.joint_io import load_joint
        return load_joint(dataset, slide_ids)
    if os.environ.get('SADA_WSI_MODE') == 'mrepath_sheaf':
        from utils.sheaf_io import load_patient_sheaf
        return load_patient_sheaf(dataset, slide_ids)
    # REGION_CONTEXT_V1
    if os.environ.get('SADA_WSI_MODE', 'hypergraph') == 'region_context':
        from utils.region_io import load_patient_regions
        return load_patient_regions(dataset, slide_ids)
    root = Path(os.environ['SADA_GRAPH_DIR'])
    xs, edge_sets = [], [[], []]
    offset = 0
    for slide in slide_ids:
        name = str(slide)
        if name.endswith('.svs'):
            name = name[:-4]
        path = root / (name + '.pt')
        if not path.is_file():
            raise FileNotFoundError(f'Missing graph; patient is NOT silently skipped: {path}')
        graph = load_trusted(path)
        x = graph.x.detach().cpu().float()
        if x.ndim != 2 or x.shape[1] != 1024 or not torch.isfinite(x).all():
            raise ValueError(f'Expected finite [N,1024] graph.x: {path}')
        for bucket, key in zip(edge_sets, ['edge_index', 'edge_latent']):
            edges = getattr(graph, key).cpu().long()
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f'Invalid {key}: {path}')
            if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
                raise ValueError(f'Out-of-range {key}: {path}')
            bucket.append(edges + offset)
        xs.append(x)
        offset += len(x)
    x = torch.cat(xs)
    if not len(x):
        raise ValueError('Empty patient graph')
    limit = int(dataset.num_patches)
    if limit < 1:
        raise ValueError('num_patches must be positive')
    full_edges = [torch.cat(bucket, dim=1) for bucket in edge_sets]
    generator = None
    if not dataset.sample:
        digest = hashlib.sha256('|'.join(map(str, slide_ids)).encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
    mode = os.environ.get('GRAPH_SAMPLE_MODE', 'neighborhood')
    selected = select_nodes(full_edges, len(x), limit, mode, generator)
    edges = [remap_edges(edge, selected, len(x)) for edge in full_edges]
    if os.environ.get('GRAPH_STATS', '0') == '1':
        report_graph_stats(full_edges, selected, len(x), mode)
    incidence = incidence_from_edges(edges, len(selected))
    return x[selected], incidence


def collate_graph_survpath(batch):
    if len(batch) != 1:
        raise ValueError('Graph experiment requires batch_size=1; SADA still buffers patients')
    features, omics, label, time, censor, clinical, incidence = batch[0]
    return [features.unsqueeze(0), [omics], torch.as_tensor(label).long().reshape(-1),
            torch.as_tensor(time).float().reshape(-1), torch.as_tensor(censor).float().reshape(-1),
            [clinical], incidence]


def select_nodes(edges_list, n, limit, mode='neighborhood', generator=None):
    """Random anchors across the patient, retaining their one-hop neighbours.
    Uses both edge types; no labels. Output has unique nodes and fixed budget.
    Changes sample distribution; compare both graph-on and graph-off with this sampler.
    """
    if mode not in ('random', 'neighborhood'):
        raise ValueError('GRAPH_SAMPLE_MODE must be random/neighborhood')
    if n <= limit:
        return torch.arange(n)
    order = torch.randperm(n, generator=generator)
    if mode == 'random':
        return order[:limit]
    edges = torch.cat(edges_list, dim=1)
    permutation = torch.argsort(edges[0])
    targets = edges[1, permutation]
    counts = torch.bincount(edges[0], minlength=n)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    seen = torch.zeros(n, dtype=torch.bool)
    result = []
    for anchor in order.tolist():
        if seen[anchor]:
            continue
        group = torch.cat([torch.tensor([anchor]), targets[offsets[anchor]:offsets[anchor+1]]]).unique()
        group = group[~seen[group]]
        # Shuffle within the group so truncation does not prefer small node IDs.
        group = group[torch.randperm(len(group), generator=generator)]
        group = group[:limit-len(result)]
        seen[group] = True
        result.extend(group.tolist())
        if len(result) == limit:
            break
    return torch.tensor(result, dtype=torch.long)


def report_graph_stats(full_edges, selected, n, mode):
    print(f'[GRAPH sample={mode}] nodes={n}->{len(selected)}', flush=True)
    for name, original in zip(['spatial', 'feature'], full_edges):
        mapped = remap_edges(original, selected, n)
        degree = torch.bincount(mapped[0], minlength=len(selected))
        selected_mask = torch.zeros(n, dtype=torch.bool)
        selected_mask[selected] = True
        source_edges = int(selected_mask[original[0]].sum())
        retained = mapped.shape[1] / max(source_edges, 1)
        print(f'  {name}: retained-neighbours={retained:.1%}; '
              f'mean-outdegree={degree.float().mean():.2f}; '
              f'no-outgoing-neighbour={(degree == 0).float().mean():.1%}', flush=True)

