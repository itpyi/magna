#!/usr/bin/env python3
"""Run fixed seeded tasks on two built checkouts and report every field that differs.

CI compares `main` with each pull request that changes the package. An optimization or refactor
should leave seeded results unchanged, so this lists what moved. It reports and never fails.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SHOTS = 16384
SEED = 20261008
SHOWN = 3
# Enough draws that each task has a nonzero error estimate, so a changed draw cannot hide behind zeros.
CASES = (
    ('dk', 'examples/dk', ()),
    ('dk (exact marginal)', 'examples/dk', ('--history-sampler', 'exact_marginal')),
    ('bh20-S', 'examples/bh20-S', ()),
    ('bh40-T', 'examples/bh40-T', ()),
    ('color3', 'examples/color3', ()),
    ('rp2-d3', 'examples/rp2-d3', ()),
    ('importance sampling', 'tests/fixtures/importance-sampling', ()),
)
CATEGORIES = ('Results', 'Counters', 'Fingerprints')


def prepare(source, target):
    """Copy a task folder. The importance-sampling fixture has no run.json, so write one."""
    shutil.copytree(source, target)
    if not (target / 'run.json').exists():
        sampling = json.loads((target / 'sampling.json').read_text())
        (target / 'run.json').write_text(json.dumps(
            {'circuit': 'physical.magna', 'task': 'task.json', 'seed': SEED, 'sampling': sampling}))


def run_case(tree, folder, options, shots, output):
    """Run one task with one checkout's package; return its result, or the reason it has none."""
    command = (sys.executable, tree / 'scripts' / 'run_task.py', folder,
               '--shots', str(shots), '--output', output, *options)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tree / 'core'), str(tree))))
    done = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True)
    if output.exists():
        return json.loads(output.read_text()), None
    return None, (done.stderr.strip().splitlines() or ['no output'])[-1]


def differences(old, new, path=''):
    """Yield the path and both values of every leaf that differs. Timings always differ, so skip them."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key != 'timings' and key != 'software_version' and not key.endswith('_seconds'):
                yield from differences(old.get(key), new.get(key), f'{path}.{key}' if path else key)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, pair in enumerate(zip(old, new)):
            yield from differences(*pair, f'{path}[{index}]')
    elif old != new:
        yield path, old, new


def category(path):
    """Counters and fingerprints can change while the results stay exactly the same."""
    if path.startswith('resources.'):
        return 'Counters'
    if 'fingerprint' in path:
        return 'Fingerprints'
    return 'Results'


def compare(old, new):
    """Sort the differences between two results into CATEGORIES."""
    found = {name: [] for name in CATEGORIES}
    for path, before, after in differences(old, new):
        found[category(path)].append((path, before, after))
    return found


def says_unchanged(body):
    return re.search(r'seeded[^:\n]*:\s*unchanged', body or '', re.I) is not None


def short(value):
    """Keep a full-precision number but not a whole fingerprint."""
    text = str(value)
    return text if len(text) <= 24 else text[:24] + '…'


def describe(name, found):
    """List the first changes of one task, estimates first because the other results follow from them."""
    changes = [change for key in CATEGORIES for change in found[key]]
    changes.sort(key=lambda change: not change[0].startswith('estimates.'))
    lines = [f'{name}: `{path}` {short(before)} → {short(after)}' for path, before, after in changes[:SHOWN]]
    return lines + ([f'{name}: {len(changes) - SHOWN} more'] if len(changes) > SHOWN else [])


def render(rows, details, problems, shots):
    lines = ['## Seeded results', '',
             f'`main` and this pull request ran {len(rows)} tasks with the same seeds, {shots} draws each.', '',
             '| Task | ' + ' | '.join(CATEGORIES) + ' |', '|---|' + '---|' * len(CATEGORIES)]
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    if details:
        lines += ['', '### Differences', '']
        lines += [f'- {detail}' for detail in details]
    lines += ['', '### Open items', '']
    lines += [f'- {problem}' for problem in problems] or ['Nothing open.']
    return '\n'.join(lines)


def review(base, head, body, shots):
    """Run every case on both checkouts, always from the same copy of the base task files."""
    rows, details, problems, moved = [], [], [], []
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        for index, (name, folder, options) in enumerate(CASES):
            task = temporary / f'task-{index}'
            prepare(base / folder, task)
            old, old_error = run_case(base, task, options, shots, temporary / f'main-{index}.json')
            new, new_error = run_case(head, task, options, shots, temporary / f'head-{index}.json')
            if old_error or new_error:
                rows.append((name, *['did not run'] * len(CATEGORIES)))
                problems.append(f'{name} did not run on {"`main`" if old_error else "this pull request"}: '
                                f'{old_error or new_error}')
                continue
            found = compare(old, new)
            rows.append((name, *[f'{len(found[c])} changed' if found[c] else 'same' for c in CATEGORIES]))
            moved += [name] if found['Results'] else []
            details += describe(name, found)
    if moved and says_unchanged(body):
        problems.append('The description says seeded results are unchanged, but they differ for: ' + ', '.join(moved))
    return rows, details, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='a built checkout of main')
    parser.add_argument('--head', type=Path, default=ROOT, help='a built checkout of the pull request')
    parser.add_argument('--body-file', type=Path, help='the pull request description; CI reads it from the event')
    parser.add_argument('--shots', type=int, default=SHOTS)
    args = parser.parse_args()
    event = os.environ.get('GITHUB_EVENT_PATH')
    body = ''
    if args.body_file:
        body = args.body_file.read_text()
    elif event:
        body = json.loads(Path(event).read_text())['pull_request']['body']

    rows, details, problems = review(args.base.resolve(), args.head.resolve(), body, args.shots)
    text = render(rows, details, problems, args.shots)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(text + '\n')
    if os.environ.get('GITHUB_ACTIONS'):
        for problem in problems:
            print('::warning title=Seeded results::' + problem.replace('%', '%25'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
