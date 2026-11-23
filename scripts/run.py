"""Run the archived SADA configuration for one cohort."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_sources():
    expected = json.loads((ROOT / 'configs/source_sha256.json').read_text())
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Archived source or split has changed: {relative}')


def verify_inputs(cohort):
    expected = json.loads((ROOT / 'configs/input_sha256.json').read_text())[cohort]
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {relative}. See README.md.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Input differs from the recorded data: {relative}')


def configure(cohort, features, clusters, output):
    config = json.loads((ROOT / f'configs/{cohort}.json').read_text())
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('SADA_', 'GRAPH_', 'REGION_', 'JOINT_')):
            del env[key]
    env.update(config['environment'])
    env['SADA_CLUSTER_DIR'] = str(Path(clusters).resolve())
    args = dict(config['args'])
    args['data_root_dir'] = str(Path(features).resolve())
    args['results_dir'] = output
    argv = [item for key, value in args.items() for item in (f'--{key}', str(value))]
    return config, env, argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'evaluate'])
    parser.add_argument('--cohort', choices=['brca', 'blca', 'coadread', 'hnsc', 'stad'], required=True)
    parser.add_argument('--features', required=True, help='Directory of UNI feature H5 files')
    parser.add_argument('--clusters', required=True, help='Directory of the recorded K32 cache files')
    parser.add_argument('--checkpoints', help='Experiment directory containing s_0_checkpoint.pt, ...')
    parser.add_argument('--fold', type=int, choices=range(5), help='Evaluate one fold; training always runs 0-4 sequentially')
    parser.add_argument('--dry-run', action='store_true', help='Verify source files and print the command without loading research data')
    opts = parser.parse_args()
    if opts.mode == 'train' and opts.fold is not None:
        parser.error('The reference training run uses all five folds in one process.')
    if opts.mode == 'evaluate' and not opts.checkpoints:
        parser.error('--checkpoints is required for evaluation')
    os.chdir(ROOT)
    verify_sources()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    config, env, argv = configure(opts.cohort, opts.features, opts.clusters, f'sada_{opts.cohort}_{opts.mode}_{stamp}')
    if opts.mode == 'train':
        command = [sys.executable, '-u', str(ROOT / 'main.py'), *argv]
    else:
        command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate.py'),
                   '--cohort', opts.cohort, '--checkpoints', str(Path(opts.checkpoints).resolve())]
        if opts.fold is not None:
            command.extend(['--fold', str(opts.fold)])
        command.extend(['--', *argv])
    print(json.dumps({'environment': config['environment'], 'cluster_directory': env['SADA_CLUSTER_DIR']}, indent=2))
    print(shlex.join(command), flush=True)
    if opts.dry_run:
        return
    verify_inputs(opts.cohort)
    if not Path(opts.features).is_dir():
        raise FileNotFoundError(opts.features)
    if opts.mode == 'train' and not Path(opts.clusters).is_dir():
        raise FileNotFoundError(opts.clusters)
    if opts.mode == 'train':
        import numpy as np
        features = sorted(Path(opts.features).glob('*.h5'))
        if not features:
            raise RuntimeError('No H5 features found')
        for feature in features:
            with np.load(Path(opts.clusters) / f'{feature.stem}.npz', allow_pickle=False) as cache:
                if int(cache['requested_k']) != 32 or int(cache['seed']) != 1:
                    raise RuntimeError(f'Expected the K32, seed-1 cache for {feature.stem}')
        probe = subprocess.run([sys.executable, '-c',
            'import torch; assert torch.cuda.is_available(), "CUDA GPU required for reference training"'], env=env)
        if probe.returncode:
            raise SystemExit(probe.returncode)
    if sys.platform != 'win32':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))
    raise SystemExit(subprocess.run(command, env=env, cwd=ROOT).returncode)


if __name__ == '__main__':
    main()
