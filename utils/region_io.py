"""WSI-wide spatial summaries and region-balanced patch coverage, without labels."""
import hashlib
import os
from pathlib import Path
import torch


def spatial_regions(coords, grid=8):
    if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
        raise ValueError('Expected nonempty [N,2] patch coordinates')
    if not torch.isfinite(coords).all() or grid < 1:
        raise ValueError('Invalid coordinates or grid size')
    coords = coords.double()
    lo = coords.min(0).values
    extent = (coords.max(0).values - lo).clamp_min(1)
    cells = ((coords - lo) / extent * grid).floor().long().clamp(0, grid - 1)
    _, ids = torch.unique(cells[:, 0] * grid + cells[:, 1], sorted=True, return_inverse=True)
    return ids


def region_means(x, ids):
    count = int(ids.max()) + 1
    sums = x.new_zeros(count, x.shape[1]).index_add_(0, ids, x)
    return sums / torch.bincount(ids, minlength=count).to(x.dtype)[:, None]


def coverage_sample(ids, limit, generator=None):
    """Round-robin across occupied regions, random order within each region.

    Retains all nodes when the budget permits. Each region is represented when
    budget >= region count; if not, select a random subset of regions.
    """
    if limit < 1:
        raise ValueError('Patch budget must be positive')
    n = len(ids)
    if n <= limit:
        return torch.arange(n)
    random_order = torch.randperm(n, generator=generator)
    sorted_order = random_order[torch.argsort(ids[random_order], stable=True)]
    counts = torch.bincount(ids)
    buckets = list(torch.split(sorted_order, counts.tolist()))
    region_order = torch.randperm(len(buckets), generator=generator).tolist()
    selected, level = [], 0
    while len(selected) < limit:
        for region in region_order:
            if level < len(buckets[region]):
                selected.append(int(buckets[region][level]))
                if len(selected) == limit:
                    break
        level += 1
    return torch.tensor(selected, dtype=torch.long)


def load_patient_regions(dataset, slide_ids):
    default_protocol = 'sada' if os.environ.get('SADA_MODEL_VARIANT', 'single') == 'single' else 'legacy'
    protocol = os.environ.get('REGION_INPUT_PROTOCOL', default_protocol)
    if protocol == 'sada':
        if os.environ.get('SADA_MODEL_VARIANT', 'single') != 'single':
            raise ValueError('SADA-aligned input is implemented for V1 only')
        return load_patient_regions_sada(dataset, slide_ids)
    if protocol not in ('legacy', 'legacy_train_full_val'):
        raise ValueError('Unknown REGION_INPUT_PROTOCOL')
    if os.environ.get('SADA_GRAPH_ENABLED') != '1':
        raise ValueError('region_context requires its enhancer to remove summary tokens')
    from utils.graph_io import load_trusted
    root = Path(os.environ['SADA_GRAPH_DIR'])
    xs, ids_list, summaries, slide_groups = [], [], [], []
    offset = 0
    grid = int(os.environ.get('REGION_GRID', '8'))
    for slide_index, slide in enumerate(slide_ids):
        path = root / (str(slide).removesuffix('.svs') + '.pt')
        graph = load_trusted(path)
        x = graph.x.detach().cpu().float()
        coords = getattr(graph, 'centroid', None)
        if coords is None:
            raise ValueError(f'Missing centroid: {path}')
        coords = coords.detach().cpu()
        if x.ndim != 2 or x.shape[1] != 1024 or len(coords) != len(x) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid aligned features/coordinates: {path}')
        local_ids = spatial_regions(coords, grid)
        means = region_means(x, local_ids)
        xs.append(x)
        ids_list.append(local_ids + offset)
        summaries.append(means)
        slide_groups.append(torch.full((len(means),), slide_index, dtype=torch.long))
        offset += len(means)
    if not xs:
        raise ValueError('Patient has no slides')
    if offset > 2048:
        raise ValueError('Too many region tokens; review patient slides/grid before training')
    x, region_ids = torch.cat(xs), torch.cat(ids_list)
    generator = None
    if not dataset.sample:
        digest = hashlib.sha256('|'.join(map(str, slide_ids)).encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
    if protocol == 'legacy_train_full_val' and not dataset.sample:
        selected = torch.arange(len(x))
    else:
        selected = coverage_sample(region_ids, int(dataset.num_patches), generator)
    # Existing batch/core adapters carry a tensor. Layout: region ID, token type,
    # slide ID. Summaries follow patches and are removed inside the enhancer.
    regions = torch.cat(summaries)
    region_slides = torch.cat(slide_groups)
    selected_ids = region_ids[selected]
    layout = torch.stack([
        torch.cat([selected_ids, torch.arange(offset)]),
        torch.cat([torch.zeros(len(selected), dtype=torch.long), torch.ones(offset, dtype=torch.long)]),
        torch.cat([region_slides[selected_ids], region_slides]),
    ])
    # ATTENTION_V3: include every unselected patch only for summary pooling.
    if os.environ.get('SADA_MODEL_VARIANT') == 'attention_v3':
        unselected_mask = torch.ones(len(x), dtype=torch.bool)
        unselected_mask[selected] = False
        extra = torch.arange(len(x))[unselected_mask]
        extra_ids = region_ids[extra]
        extra_layout = torch.stack([extra_ids, torch.full_like(extra_ids, 2), region_slides[extra_ids]])
        layout = torch.cat([layout[:, :len(selected)], extra_layout, layout[:, len(selected):]], dim=1)
        return torch.cat([x[selected], x[extra], regions]), layout
    return torch.cat([x[selected], regions]), layout


def load_patient_regions_sada(dataset, slide_ids):
    """Historical SADA patch order, NumPy sampling and zero padding, plus V1 summaries."""
    import h5py
    import numpy as np
    if os.environ.get('SADA_GRAPH_ENABLED') != '1':
        raise ValueError('V1 requires its region adapter')
    xs, ids_list, summaries, groups = [], [], [], []
    offset = 0
    for slide_index, slide in enumerate(slide_ids):
        # Match the historical H5 path convention exactly.
        path = Path(dataset.data_dir) / ('{}.h5'.format(str(slide).rstrip('.svs')))
        with h5py.File(path, 'r') as f:
            x = torch.from_numpy(f['features'][:])
            coords = torch.from_numpy(f['coords'][:])
        if len(x) != len(coords) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid aligned H5 data: {path}')
        ids = spatial_regions(coords, int(os.environ.get('REGION_GRID', '8')))
        means = region_means(x, ids)
        xs.append(x)
        ids_list.append(ids + offset)
        summaries.append(means)
        groups.append(torch.full((len(means),), slide_index, dtype=torch.long))
        offset += len(means)
    x, ids = torch.cat(xs), torch.cat(ids_list)
    region_slides = torch.cat(groups)
    if dataset.sample:
        n = min(len(x), dataset.num_patches)
        selected = np.sort(np.random.choice(len(x), n, replace=False))
        patches, patch_ids = x[selected], ids[selected]
        padding = dataset.num_patches - n
        if padding:
            patches = torch.cat([patches, torch.zeros(padding, x.shape[1])])
            # -1 identifies historical zero padding, which receives no context update.
            patch_ids = torch.cat([patch_ids, torch.full((padding,), -1, dtype=torch.long)])
    else:
        patches, patch_ids = x, ids
    patch_slides = region_slides[patch_ids.clamp_min(0)]
    layout = torch.stack([
        torch.cat([patch_ids, torch.arange(offset)]),
        torch.cat([torch.zeros(len(patches), dtype=torch.long), torch.ones(offset, dtype=torch.long)]),
        torch.cat([patch_slides, region_slides]),
    ])
    return torch.cat([patches, torch.cat(summaries)]), layout
