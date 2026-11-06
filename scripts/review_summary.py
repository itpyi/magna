#!/usr/bin/env python3
"""Compare a pull request with its description and summarize both for reviewers.

CI runs this on every pull request. It reports and never fails, so a description
it cannot read costs a reminder rather than a red check.
"""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / '.github' / 'pull_request_template.md'
SECTIONS = ('Purpose', 'Scope', 'Interface, format and results', 'Tests')
BOX = re.compile(r'\s*- \[([ xX])\]\s*(.*)')


def backticked(text):
    return re.findall(r'`([^`]+)`', text)


def clean(body):
    """GitHub stores descriptions with CRLF line ends and keeps the template's comments."""
    return re.sub(r'<!--.*?-->', '', (body or '').replace('\r\n', '\n'), flags=re.S)


def section(body, title):
    """The text under one `## title` heading, or None if the description has no such heading."""
    match = re.search(rf'^## {re.escape(title)}[ \t]*$(.*?)(?=^## |\Z)', body, re.M | re.S)
    return match[1] if match else None


def read_parts(template):
    """The template's scope boxes are the one list of repository parts."""
    boxes = filter(None, map(BOX.fullmatch, template.splitlines()))
    return [(box[2].split('`')[0].strip(), backticked(box[2])) for box in boxes]


def matches(path, pattern):
    """A trailing slash means a folder, a star stays inside one folder, anything else is one file."""
    if pattern.endswith('/'):
        return path.startswith(pattern)
    if '*' in pattern:
        return path.count('/') == pattern.count('/') and fnmatch.fnmatchcase(path, pattern)
    return path == pattern


def read_scope(scope):
    """Return the paths a Scope section names, its ticked boxes that name none, and how many it ticks."""
    paths, vague, ticked = [], [], 0
    for line in scope.splitlines():
        box = BOX.fullmatch(line)
        if box and box[1] == ' ':
            continue
        paths += backticked(line)
        if box:
            ticked += 1
            if not backticked(line):
                vague.append(box[2])
    return paths, vague, ticked


def review(files, body, parts):
    """Return one table row per changed part, and what the description leaves open."""
    body = clean(body)
    items = [f'Add the "{name}" section from the pull request template.'
             for name in SECTIONS if section(body, name) is None]
    scope = section(body, 'Scope')
    paths, vague, ticked = read_scope(scope or '')
    readable = ticked > 0 and not vague
    if scope is not None and not ticked:
        items.append('Tick the parts this change touches under "Scope".')
    items += [f'Name the path for this box under "Scope": {text}' for text in vague]

    rows = []
    for label, group in by_part(files, parts):
        missing = [f for f in group if not any(matches(f, p) for p in paths)]
        if not readable:
            declared = '?'
        elif missing:
            declared = 'no' if len(missing) == len(group) else f'{len(group) - len(missing)} of {len(group)}'
            items.append(f'Not declared under "Scope": {label}, {count(len(missing), "file")}. '
                         'Tick the part, or name each file in backticks with a reason.')
        else:
            declared = 'yes'
        rows.append((label, len(group), declared))

    interface = section(body, 'Interface, format and results') or ''
    if re.search(r'unchanged / changed|updated / not needed', interface):
        items.append('Choose one answer on each line under "Interface, format and results".')
    if (re.search(r'public interface[^:\n]*:\s*changed', interface, re.I)
            and not any(f.startswith('docs/software/') for f in files)):
        items.append('The description says the public interface changes, so update the manual in `docs/software/`.')
    if 'CHANGELOG.md' not in files and not re.search(r'CHANGELOG[^\n]*not needed', body):
        items.append('Add a line under "Unreleased" in `CHANGELOG.md`, or answer "not needed" in the description.')
    if any(f.startswith('core/') for f in files) and not any(f.startswith('tests/') for f in files):
        items.append('The package changes but `tests/` does not. New behaviour and bug fixes need a test.')
    return rows, items


def by_part(files, parts):
    """Group files under the first part that covers them, with any others in a last group."""
    groups, rest = [], list(files)
    for label, patterns in parts:
        group = [f for f in rest if any(matches(f, p) for p in patterns)]
        rest = [f for f in rest if f not in group]
        if group:
            groups.append((label, group))
    return groups + ([('Outside every part', rest)] if rest else [])


def count(number, noun):
    return f'{number} {noun}' + ('' if number == 1 else 's')


def maintainer_notes(files, behind, overlap):
    """What a maintainer would otherwise work out by hand before reviewing."""
    notes = []
    if any(f.startswith('core/magna/native/') and not f.startswith('core/magna/native/cuda/') for f in files):
        notes.append('This changes the C++ kernel, so it needs the benchmark suite before merging.')
    if behind:
        note = f'`main` has {count(behind, "newer commit")}.'
        if overlap:
            note += f' {count(len(overlap), "file")} also changed there, so ask for an updated branch: '
            note += ', '.join(f'`{f}`' for f in overlap)
        notes.append(note)
    return notes


def git(*args, cwd=ROOT):
    return subprocess.run(('git', *args), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def compare(base, head, cwd=ROOT):
    """Return the files a branch changes, its size, and how far `base` has moved since the branch started."""
    fork = git('merge-base', base, head, cwd=cwd)
    def changed(start, end):
        return git('diff', '--name-only', '--no-renames', start, end, cwd=cwd).splitlines()
    files = changed(fork, head)
    behind = int(git('rev-list', '--count', f'{fork}..{base}', cwd=cwd))
    # Nearly every pull request adds to the changelog, and GitHub reports any real conflict there.
    overlap = sorted((set(files) & set(changed(fork, base))) - {'CHANGELOG.md'})
    return files, git('diff', '--shortstat', fork, head, cwd=cwd), behind, overlap


def render(size, rows, items, notes):
    lines = ['## Review summary', '', size or 'No files changed.', '']
    if rows:
        lines += ['| Part | Files | Declared |', '|---|---|---|']
        lines += [f'| {label} | {number} | {declared} |' for label, number, declared in rows]
        lines.append('')
    lines += ['### Open items', '']
    lines += [f'- {item}' for item in items] or ['Nothing open.']
    if notes:
        lines += ['', '### For the maintainers', '']
        lines += [f'- {note}' for note in notes]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='origin/main', help='the branch the pull request merges into')
    parser.add_argument('--head', default='HEAD', help='the pull request branch')
    parser.add_argument('--body-file', type=Path, help='the pull request description; CI reads it from the event')
    args = parser.parse_args()
    event = os.environ.get('GITHUB_EVENT_PATH')
    if args.body_file:
        body = args.body_file.read_text()
    elif event:
        body = json.loads(Path(event).read_text())['pull_request']['body']
    else:
        parser.error('give --body-file with the pull request description')

    files, size, behind, overlap = compare(args.base, args.head)
    rows, items = review(files, body, read_parts(TEMPLATE.read_text()))
    text = render(size, rows, items, maintainer_notes(files, behind, overlap))
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(text + '\n')
    if os.environ.get('GITHUB_ACTIONS'):
        for item in items:
            print('::warning title=Review summary::' + item.replace('%', '%25'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
