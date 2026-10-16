"""Measure packaged tasks through the existing public task-runner CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

try:
    import resource
except ImportError:  # Unavailable on Windows.
    resource = None


ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def load_average():
    return list(os.getloadavg()) if hasattr(os, 'getloadavg') else None


def child_resources(before):
    if before is None:
        return None
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(user_seconds=after.ru_utime - before.ru_utime,
                system_seconds=after.ru_stime - before.ru_stime,
                voluntary_switches=after.ru_nvcsw - before.ru_nvcsw,
                involuntary_switches=after.ru_nivcsw - before.ru_nivcsw,
                minor_faults=after.ru_minflt - before.ru_minflt)


def source_digest():
    digest = hashlib.sha256()
    sources = [ROOT / 'scripts/run_task.py']
    for suffix in ('*.py', '*.cpp', '*.hpp', '*.json'):
        sources.extend((ROOT / 'core/magna').rglob(suffix))
    for path in sorted(sources):
        if path.name.startswith('native-') and path.name.endswith('build.json'):
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def run_command(command, timeout, measure_rss):
    if not measure_rss:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    # GNU time is an observer process. Keep its child in the same new process
    # group so timing out the benchmark also stops the actual task runner.
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--workers', type=positive, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--shots', type=positive)
    parser.add_argument('--repeats', type=positive, default=3)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--timeout', type=positive, default=600,
                        help='maximum seconds for each task-runner process')
    parser.add_argument('--label', default='working tree')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workers = list(dict.fromkeys(args.workers))
    baseline_workers = min(workers)
    runner = str(ROOT / 'scripts/run_task.py')
    runner_help = subprocess.check_output([sys.executable, runner, '--help'], text=True)
    runner_supports_workers = '--workers' in runner_help
    if not runner_supports_workers and workers != [1]:
        parser.error('this revision of the task runner only supports the serial baseline')
    report = dict(label=args.label, source_sha256=source_digest(),
                  environment=dict(python=sys.version, platform=platform.platform(),
                                   machine=platform.machine(), processor=platform.processor(),
                                   logical_cpus=os.cpu_count(),
                                   affinity_cpus=len(os.sched_getaffinity(0))
                                   if hasattr(os, 'sched_getaffinity') else None),
                  workers=workers, baseline_workers=baseline_workers,
                  repeats=args.repeats, runs=[], summary=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    with tempfile.TemporaryDirectory(prefix='magna-benchmark-') as temporary:
        result_path = Path(temporary) / 'result.json'
        rss_path = Path(temporary) / 'peak-rss.txt'
        # GNU time reports this individual child's peak RSS; RUSAGE_CHILDREN's
        # ru_maxrss is a cumulative maximum and cannot be differenced per run.
        measure_rss = sys.platform.startswith('linux') and Path('/usr/bin/time').is_file()
        for directory in args.directories:
            directory = directory.resolve()
            config = json.loads((directory / 'run.json').read_text())
            shots = args.shots if args.shots is not None else config.get('shots', 1024)
            task_runs = []
            for repeat in range(args.repeats):
                # Rotate execution order to reduce a consistent first-run/load bias.
                order = workers[repeat % len(workers):] + workers[:repeat % len(workers)]
                for count in order:
                    command = [sys.executable, runner, str(directory),
                               '--shots', str(shots), '--seed', str(args.seed + repeat),
                               '--output', str(result_path)]
                    # The unchanged serial command also benchmarks older revisions.
                    if runner_supports_workers:
                        command += ['--workers', str(count)]
                    if measure_rss:
                        command = ['/usr/bin/time', '-f', '%M', '-o', str(rss_path), *command]
                    before = load_average()
                    usage = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
                    started = perf_counter()
                    completed = run_command(command, args.timeout, measure_rss)
                    wall_seconds = perf_counter() - started
                    resources = child_resources(usage)
                    if measure_rss and completed.returncode == 0:
                        resources['peak_rss_kib'] = int(rss_path.read_text().strip())
                    if completed.returncode:
                        save()
                        raise RuntimeError(f'Task runner failed ({directory.name}, workers={count}):\n'
                                           + completed.stderr + completed.stdout)
                    result = json.loads(result_path.read_text())
                    if result['status'] != 'complete' or result['completed'] != shots:
                        raise RuntimeError(f'Incomplete benchmark result: {result}')
                    entry = dict(task=directory.name, workers=count, repeat=repeat,
                                 shots=shots, seed=args.seed + repeat, wall_seconds=wall_seconds,
                                 load_before=before, load_after=load_average(), resources=resources,
                                 result=result)
                    report['runs'].append(entry)
                    task_runs.append(entry)
                    save()
                    print(f'{directory.name} workers={count} repeat={repeat + 1}: '
                          f'wall={wall_seconds:.4f}s '
                          f'sampling={result["timings"]["sampling_seconds"]:.4f}s', flush=True)
            medians = {}
            for count in workers:
                entries = [entry for entry in task_runs if entry['workers'] == count]
                medians[count] = dict(
                    wall_seconds=statistics.median(entry['wall_seconds'] for entry in entries),
                    total_seconds=statistics.median(entry['result']['timings']['total_seconds']
                                                    for entry in entries),
                    preparation_seconds=statistics.median(entry['result']['timings']['preparation_seconds']
                                                          for entry in entries),
                    sampling_seconds=statistics.median(entry['result']['timings']['sampling_seconds']
                                                       for entry in entries))
            for count, times in medians.items():
                report['summary'].append(dict(task=directory.name, shots=shots, workers=count,
                    **times, baseline_workers=baseline_workers,
                    wall_speedup=medians[baseline_workers]['wall_seconds'] / times['wall_seconds'],
                    sampling_speedup=medians[baseline_workers]['sampling_seconds'] / times['sampling_seconds'],
                    shots_per_sampling_second=shots / times['sampling_seconds']))
            save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
