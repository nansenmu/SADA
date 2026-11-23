# SADA

Code for **SADA: Survival-aware disentangled alignment for multimodal cancer survival prediction**.

SADA combines histology patch features and transcriptomic pathway features through bidirectional channel gating, multimodal attention, and survival-aware shared/specific representation learning.

## Contents

- `models/model_SurvPath.py`: SADA model, retaining its historical module name.
- `utils/core_utils.py`, `utils/sada_loss.py`: training, validation, and loss functions.
- `cluster_sqrt.py`, `cluster_io.py`, `prepare_clusters.py`: square-root cluster sampling and cache preparation.
- `splits/5foldcv/`: the 25 original input split files for the five cohorts.
- `configs/`: fixed training settings, source/input checksums, effective patient lists, and reference scores.
- `scripts/run.py`: training and checkpoint evaluation entry point.

The model, training loop, data loader, optimizer, and sampling source files are copied unchanged from the archived five-cohort run. Several small imported compatibility modules are retained. The launcher selects the SADA configuration; graph extensions and other experimental variants are disabled.

## Environment

Linux and a CUDA-capable GPU are required for the reference training procedure. The current server uses Python 3.12.3 and PyTorch 2.12.0+cu130. Its package versions are recorded in `configs/environment_snapshot.json`; these are a current environment snapshot, not a recovered lockfile from the original training run. A fresh installation has not been validated.

Install a PyTorch build compatible with your GPU and driver, then install the Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-legacy-dgl.txt
```

The historical data loader imports DGL even though the selected SADA configuration does not use graphs. The recorded server has DGL 2.4.0+cu121; its wheel availability and binary compatibility must be checked for a new environment. Do not substitute a different optimizer implementation: the archived `custom_optims/radam.py` is used.

## Data preparation

This repository contains code and split/configuration files. WSI features, RNA matrices, clinical tables, cluster caches, and checkpoints are not included in this release.

The original data are from TCGA. See the [GDC portal](https://portal.gdc.cancer.gov/) and the [SurvPath data preparation instructions](https://github.com/mahmoodlab/SurvPath#transcriptomics-and-pathway-compositions) for data sources. A fresh upstream download may differ from the processed inputs used here; `configs/input_sha256.json` records the required file versions.

Place or link the processed inputs in the following layout relative to the repository:

```text
datasets_csv/
  metadata/
    combine_signatures.csv
    tcga_brca.csv
    tcga_blca.csv
    tcga_hnsc.csv
    tcga_stad.csv
  pathway_compositions/combine_comps.csv
  clinical_data/tcga_<cohort>_clinical.csv
  raw_rna_data/combine/<cohort>/rna_clean.csv
cohort/
  metadata.csv                    # COADREAD metadata from the archived run
```

For example:

```bash
ln -s /path/to/processed/datasets_csv datasets_csv
mkdir -p cohort
ln -s /path/to/archived/cohort/metadata.csv cohort/metadata.csv
```

UNI features are stored as one H5 file per slide, with a `features` dataset of shape `(patches, 1024)`. Preserve slide identifiers, patch ordering, feature values, RNA column order, and the complete metadata tables. The input hashes are checked before a run. The data loader fits the RNA scaler on training data; its historical zero handling and survival-bin construction are retained.

The raw split lists can contain cases removed when the data loader matches available modalities. `configs/effective_splits/` records the resulting patient lists and order. Their cohort totals are BRCA 866, BLCA 359, COADREAD 295, HNSC 392, and STAD 318. Do not regenerate the folds or replace the original input split files with the filtered lists: RNA preprocessing also depends on the original lists.

For COADREAD, the archived command supplied `cohort/splits`, but `_prepare_for_experiment` actually read `splits/5foldcv/tcga_coadread`. This release uses the effective directory and retains `cohort/metadata.csv` as the label source.

Use the recorded K32 cluster caches when available. To prepare caches from the identical UNI features:

```bash
python prepare_clusters.py --source /path/to/blca_features_uni_v1 \
  --output /path/to/blca_cluster_cache_k32_v1 --clusters 32
```

Clustering uses L2-normalized features, `MiniBatchKMeans`, seed 1, `n_init=3`, `batch_size=2048`, `max_iter=100`, and `reassignment_ratio=0`. Model inputs remain the original UNI features. Regenerating caches under a different scikit-learn version may change assignments.

## Fixed training configuration

| Setting | Value |
|---|---|
| Endpoint / time intervals | DSS / 4 |
| Folds | 0, 1, 2, 3, 4, run sequentially |
| Random seed | 1, initialized once before the five-fold run |
| Epochs | 30 |
| Optimizer | Archived RAdam implementation |
| Learning rate / weight decay | 1e-4 / 1e-5 |
| Learning-rate scheduler | Cosine; scheduler warm-up parameter 1 |
| Patient batch size | 1 |
| Shared-loss buffer size | 8 patients |
| SADA warm-up | First 2 epochs: main survival loss only |
| Auxiliary / orthogonality / shared / specific weights | 0.3 / 0.1 / 0.5 / 0.1 |
| Contrastive temperature / bin distance | 0.1 / 1 |
| Training sampling | Square-root cluster quotas, K=32, 4096 patches |
| Validation sampling | All available patches |
| CUDA data-loader workers | 8 |
| NLL censoring parameter | `alpha_surv=0.0` |
| Prediction | 0.7 main logits + 0.3 auxiliary logits |

The learning-rate scheduler and its step counting are retained as implemented in the archive. The scheduler warm-up parameter and the two-epoch SADA warm-up are separate settings.

## Training

Run from the repository root. First inspect the resolved configuration:

```bash
python scripts/run.py train --cohort blca \
  --features /path/to/blca_features_uni_v1 \
  --clusters /path/to/blca_cluster_cache_k32_v1 --dry-run
```

Remove `--dry-run` to train. Supported cohorts are `brca`, `blca`, `coadread`, `hnsc`, and `stad`. Each invocation runs all five folds in the original order. Starting each fold in a separate process resets the random-number sequence and does not match the reference procedure.

Use the launcher rather than calling `main.py` with its defaults. It clears inherited experimental environment variables and explicitly fixes the SADA settings. It checks source/split hashes, input hashes, and the K32 cache metadata before training. The data loader additionally checks feature fingerprints against the caches.
