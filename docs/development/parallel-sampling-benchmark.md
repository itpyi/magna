# Parallel sampling benchmark, 2026-10-16

The packaged `bh2-T`, `bh20-T`, `rp2-d3`, and `wg-T` tasks completed with
262,144 shots per run. With eight workers, median sampling speedups were
3.61x to 4.85x relative to the feature branch with one worker; complete CLI
invocations sped up by 3.30x to 4.48x. These are measurements on a shared
server, not a prediction for every circuit or machine.

## Public interface and method

The [benchmark driver](../../scripts/benchmark_parallel_sampling.py) launches
the existing `scripts/run_task.py` in a fresh Python process for every measurement.
That runner loads each task's unchanged `run.json` and calls public `magna.run`.
Only `shots`, `seed`, and `workers` are overridden. There is no benchmark-specific
sampling implementation, direct native invocation, or synthetic replacement circuit.

Each configuration has three repetitions, with seeds 20261016, 20261017, and
20261018. Worker order rotates between repetitions. Tables report medians;
the [measurement data](benchmark-results/parallel-sampling-20261016.json) retains
individual times, estimates, draw digests, load averages, and source/binary provenance.

`sampling_seconds` includes preparing additional worker runtimes, generating
their seeds, scheduling, sampling, and combining results. It excludes the initial
task preparation. CLI wall time includes Python startup, task loading, preparation,
sampling, and writing the result JSON. Every invocation starts with fresh caches.

All four tasks use their configured physical noise proposal and `affine_support`
history sampler. The measurements do not establish speedups for every history sampler,
importance proposal, or external decoder.

## Environment

- Host: `ghz`, two Intel Xeon Platinum 8358P processors at a nominal 2.60 GHz;
  64 physical cores, 128 logical CPUs, 503 GiB RAM.
- Python 3.10.12; GCC 11.4.0; Linux 5.15.0-187-generic, glibc 2.35.
- Native release build: C++17, `-O3 -DNDEBUG`; the candidate also links with `-pthread`.
- Main baseline: commit `56847478c840395735146ef1e1ccdc3d8aedc10f`.
  Its isolated source snapshot needed the missing standard headers `<optional>`
  and `<limits>` to compile with GCC 11. These same header fixes are in the feature
  branch. The local `main` checkout and reference were not modified.
- The throughput runs were unpinned on a shared machine. The one-minute host
  load average ranged from 82.45 to 90.29 during candidate runs. Other users' work
  continued, and some repository checks/builds used additional single CPU processes.
  Individual timings vary materially; the full observations are retained.

## Throughput at 262,144 shots

The comparison here uses the same feature build for both the one-worker and
eight-worker columns.

| Task | 1-worker sampling (s) | 8-worker sampling (s) | Sampling speedup | 1-worker CLI (s) | 8-worker CLI (s) | CLI speedup |
|---|---:|---:|---:|---:|---:|---:|
| `bh2-T` | 7.1521 | 1.4759 | 4.85x | 7.2410 | 1.6159 | 4.48x |
| `bh20-T` | 9.7614 | 2.7053 | 3.61x | 10.0103 | 3.0349 | 3.30x |
| `rp2-d3` | 4.6470 | 1.2811 | 3.63x | 4.7440 | 1.4246 | 3.33x |
| `wg-T` | 7.3407 | 1.5494 | 4.74x | 7.5462 | 1.6940 | 4.45x |

| Task | 2-worker sampling speedup | 4-worker sampling speedup | 8-worker sampling speedup |
|---|---:|---:|---:|
| `bh2-T` | 1.80x | 3.03x | 4.85x |
| `bh20-T` | 1.20x | 1.98x | 3.61x |
| `rp2-d3` | 1.46x | 2.30x | 3.63x |
| `wg-T` | 1.75x | 2.89x | 4.74x |

The earlier unpinned main baseline had median sampling times of 4.9493, 8.2986,
5.1374, and 3.7847 seconds, respectively. Comparing those medians with eight
workers gives 3.35x, 3.07x, 4.01x, and 2.44x. Those measurements occurred at
different times under substantial load variation; they are retained separately
from the feature-build scaling table.

## Serial compatibility and reproducibility

For the 12 large serial measurements, every reported draw/statistical field matched
main exactly: status and counts, estimates and their scientific representations,
standard errors, weights, efficiency, requested records, sampling settings, route,
and numerical metadata. The only differing fingerprint was `ideal_state`, whose
serialized circuit includes the absolute source filename in the separate checkouts.

A second comparison ran both revisions against the **same task paths**, pinned to
CPU 2, with three alternating pairs for `bh2-T` and `bh20-T`. CPU 2 and its SMT
sibling 66 had the lowest observed activity during an initial two-second probe.
All 18 fields listed in the evidence file, including all fingerprints, matched
exactly in all six pairs. These jobs ran after our throughput measurements and
expensive builds stopped; other users' workloads continued.

| Task | Main serial sampling median (s) | Feature serial sampling median (s) | Change |
|---|---:|---:|---:|
| `bh2-T` | 3.9057 | 4.0248 | +3.05% |
| `bh20-T` | 9.7438 | 9.3705 | -3.83% |

These paired medians do not show a consistent serial slowdown. They cannot exclude
small changes: interference persisted even with affinity, and `bh20-T` main timings
ranged from 6.69 to 17.37 seconds. Pinning was applied only to these comparison
processes and did not change other users' workloads.

Within the feature branch, all draw/statistical fields and fingerprints matched
exactly between two and four workers and between two and eight workers for every
large task and seed: 24 comparisons. Parallel and sequential streams intentionally
differ, so equality between those streams is not required.

## Short runs and Python decoders

The same driver also ran 1,024 shots with one and eight workers. `dk` uses its
packaged Python syndrome decoder. Neither short task gained end-to-end speed:

| Task | 1-worker sampling (s) | 8-worker sampling (s) | 1-worker CLI (s) | 8-worker CLI (s) | CLI speedup |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 0.01630 | 0.01448 | 0.09433 | 0.09857 | 0.96x |
| `dk` | 0.03250 | 0.03242 | 0.11802 | 0.12413 | 0.95x |

Additional worker preparation and scheduling must be amortized by enough work.
Python decoder callbacks acquire the interpreter lock, which can also limit scaling.

## Validation

The final release build passed all 369 tests in `scripts/project_checks.py --suite current`
on Python 3.10.12, with no skips. This included the installed-wheel interface tests,
the optional Stim differential checks, and standalone C++ checks. The wheel was built
with the normal package build and installed in an isolated environment with the
repository's example decoder fixture.

All 17 parallel-sampling tests also passed against a separately built UBSan extension
(`-fsanitize=undefined -fno-sanitize-recover=all`), with no sanitizer findings.
They cover deterministic partitioning, exact and stochastic oracles, callbacks,
failure recovery, shared query/time budgets, signal interruption and reentry.
Documentation language/link checks and `mkdocs build --strict` passed.

Validation on the supported minimum Python version exposed two pre-existing test
assumptions: underscore parsing by Python's `Fraction` and empty-tuple interning.
Both failures were reproduced on main before correcting only the test fixtures.
The standalone sanitizer compilation timeout was increased from 120 to 300 seconds
for GCC on the shared server; simulation assertions were retained.

## Reproduce

From the feature checkout with its native extension built:

```sh
python3 scripts/build_native_core.py --cxx g++
PYTHONPATH=core:. python3 scripts/benchmark_parallel_sampling.py \
  examples/bh2-T examples/bh20-T examples/rp2-d3 examples/wg-T \
  --shots 262144 --workers 1 2 4 8 --repeats 3 --seed 20261016 \
  --label feat-parallel-sampling --output parallel-results.json

PYTHONPATH=core:. python3 scripts/benchmark_parallel_sampling.py \
  examples/bh2-T examples/dk --shots 1024 --workers 1 8 --repeats 3 \
  --seed 20261016 --output parallel-small-results.json
```

With an installed package, omit `PYTHONPATH=core:.`. For a single ordinary run:

```sh
python scripts/run_task.py examples/bh20-T --shots 262144 --seed 20261016 \
  --workers 8 --output bh20-parallel.json
```

To reproduce the paired serial check, invoke each revision's existing
`scripts/run_task.py` against one shared absolute example directory, with identical
shots and seeds, prefixing both commands with `taskset -c CPU`. Alternate revision
order across repeats and retain each JSON result. The default worker count is one
in both revisions; no baseline sampling code needs modification.
