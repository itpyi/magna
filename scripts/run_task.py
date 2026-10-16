"""Run a portable task directory using only the public simulator interface."""
import argparse
import json
from pathlib import Path
import sys


def load(directory):
    """Each invocation loads one explicitly selected task and its local decoder."""
    from magna import load_controllers
    directory = Path(directory).resolve()
    config = json.loads((directory / 'run.json').read_text())
    allowed = {'circuit', 'task', 'controllers', 'shots', 'seed', 'options', 'sampling', 'workers'}
    if set(config) - allowed:
        raise ValueError(f'Unknown run settings: {sorted(set(config) - allowed)}')
    def local(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Missing or nonlocal task file: {name}')
        return path
    # Decoder code is explicitly selected by controllers.json, as with installed bindings.
    sys.path.insert(0, str(directory))
    controllers = load_controllers(local(config['controllers'])) if config.get('controllers') else None
    return local(config['circuit']), dict(task=local(config['task']), controllers=controllers,
        options=config.get('options', {}), sampling=config.get('sampling', {'method': 'physical'})), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--shots', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, help='native sampling threads (default 1)')
    parser.add_argument('--history-sampler', choices=('affine_support', 'exact_marginal'))
    parser.add_argument('--progress-every', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from magna import run
    circuit, settings, config = load(args.directory)
    if args.history_sampler:
        settings['options']['history_sampler'] = args.history_sampler
    result = run(circuit, shots=args.shots if args.shots is not None else config.get('shots', 1024),
                 seed=args.seed if args.seed is not None else config.get('seed', 0),
                 workers=args.workers if args.workers is not None else config.get('workers', 1),
                 progress_every=args.progress_every, **settings)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
