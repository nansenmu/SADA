"""Evaluate saved checkpoints with the archived data loader and risk calculation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    separator = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(5))
    opts = parser.parse_args(sys.argv[1:separator])
    sys.argv = ['main.py', *sys.argv[separator + 1:]]
    import numpy as np
    import torch
    from datasets.dataset_survival import SurvivalDatasetFactory
    from utils.process_args import _process_args
    from utils.general_utils import _prepare_for_experiment
    from utils.core_utils import (_init_model, _init_loaders, _init_loss_function,
                                  _extract_survival_metadata, _summary)
    from utils.file_utils import _save_pkl
    args = _prepare_for_experiment(_process_args())
    factory = SurvivalDatasetFactory(
        study=args.study, label_file=args.label_file, omics_dir=args.omics_dir,
        seed=args.seed, print_info=True, n_bins=args.n_classes, label_col=args.label_col,
        eps=1e-6, num_patches=args.num_patches, is_mcat=False,
        is_survpath=True, type_of_pathway=args.type_of_path)
    args.dataset_factory = factory
    expected = json.loads((ROOT / f'configs/effective_splits/{opts.cohort}.json').read_text())
    results = []
    for fold in ([opts.fold] if opts.fold is not None else range(5)):
        train, val = factory.return_splits(args, f'{args.split_dir}/splits_{fold}.csv', fold)
        for key, dataset in [('train', train), ('val', val)]:
            if list(dataset.metadata.case_id.astype(str)) != expected[fold][key]:
                raise RuntimeError(f'{opts.cohort} fold {fold}: {key} patient order differs from the archive')
        model = _init_model(args)
        checkpoint = opts.checkpoints / f's_{fold}_checkpoint.pt'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        train_loader, val_loader = _init_loaders(args, train, val)
        output = _summary(factory, model, args.modality, val_loader, _init_loss_function(args),
                          _extract_survival_metadata(train_loader, val_loader))
        _save_pkl(str(Path(args.results_dir) / f'predictions_{fold}.pkl'), output[0])
        results.append({'fold': fold, 'cindex': float(output[1])})
        del model, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {'cohort': opts.cohort, 'folds': results,
              'mean': float(np.mean([row['cindex'] for row in results])),
              'std_population': float(np.std([row['cindex'] for row in results], ddof=0))}
    (Path(args.results_dir) / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
