# Shared execution plan scaling, 2026-10-16

This follow-up measures first-use worker preparation, memory use, and larger
32/64-worker sampling jobs through the existing public task runner. The baseline
is commit `28247f4`. The workloads are the unchanged `bh2-T`, `bh20-T`, `rp2-d3`,
and `wg-T` examples with their physical noise proposals and `affine_support`
history sampler.

Shared preparation reduces the native compilation count from 64/128 to two and
cuts 64-worker peak RSS by 71.8% to 86.2%. At 4,194,304 shots, the optimized
64-worker runs are 1.32x to 1.49x faster than 32-worker runs. Relative to the
previous 64-worker implementation, observed sampling gains are 1.01x to 1.12x;
near-unity elapsed differences remain sensitive to the shared-server conditions.

## Method and environment

Every timed observation launches `scripts/run_task.py` in a fresh process using
the existing `scripts/benchmark_parallel_sampling.py` driver. Sampling counts
are 1,048,576 and 4,194,304. A separate 4,096-shot case covers exactly one
native sampling window, making worker first-use costs more visible. It includes
real sampling and ordered statistics, so it is not a pure compilation timer.
Three repetitions use seeds 20261016, 20261017, and 20261018. Adjacent
task/worker pairs alternate version order, and configuration order rotates
between seeds. A fixed configuration keeps its version order across these
three repeats; small elapsed differences should not be read as robust gains.

Both versions read the same absolute example paths, preserving path-dependent
evaluation metadata. No benchmark calls a private native sampling entry point.
The benchmark driver now permits worker subsets such as `--workers 32 64`;
its explicit `baseline_workers` field identifies the smallest requested count
used for within-sweep speedups. The default still includes one worker.

The server is `ghz`, with two Intel Xeon Platinum 8358P processors, 64 physical
cores, 128 logical CPUs, two NUMA nodes, and 503 GiB RAM. Builds use Python
3.10.12 and GCC 11.4.0 with C++17, `-O3 -DNDEBUG -pthread`. Measurements run
unpinned on this shared server; other users' workloads remain active. Our timed
paired workloads do not overlap another project benchmark, test suite, or native
build. A 64-worker request does not reserve 64 physical cores.

Sampling seconds include worker preparation, seed generation, dispatch, draws,
and ordered accumulation. CLI wall seconds also include Python startup, task
loading, initial preparation, result serialization, and teardown. Public
`resources.compilations` counts actual native boundary compilations. Public
`timings.compile_seconds` sums compiler time across workers and therefore is not
an elapsed wall-clock measure.

GNU `/usr/bin/time -f %M` observes the peak resident set of each individual CLI
process. The benchmark also records child user/system CPU time, context switches,
page faults, and load averages. CPU deltas include the lightweight GNU time
observer. Peak RSS is not computed by differencing cumulative
`RUSAGE_CHILDREN.ru_maxrss`. Benchmark timeouts terminate the observer's process
group, including the task child; this behavior was checked with a deliberately
one-second timeout.

The baseline snapshot is preserved under
`/tmp/magna-shared-plans-20261016-1640/baseline`, and the optimized snapshot is
under `candidate` in that same experiment directory. Build manifests and actual
Python/native import paths are checked against each snapshot. The accompanying
[compact evidence](benchmark-results/shared-plan-sampling-20261016.json) records
individual observations, production-source and native-binary SHA-256 hashes,
scientific payloads, exact-result checks, and intermediate ablations. The final
production-source digest is
`af020c92f512b9781a012f440f9d73a4ad55907996074530192c5e7d2c0dc0f7`;
the native-binary digest is
`1bb8a57d1487853177db4f1eea8031a159b5ba7dda10866b413a74f915aa98e9`.

## Implementation and preparation cost

Workers share completed read-only boundary plans, reference networks, and the
history sampling plan. Their limits, resource counters, random state, sampling
sessions, and mutable contractions remain independent. Reconfiguration replaces
published owners before workers reuse the new preparation. The existing seeded
parallel stream, 4,096-draw windows, and ordered arithmetic accumulation remain
unchanged.

Sharing also needs care in the draw loop. An initial implementation shared the
underlying `PhaseResource` ownership counters between workers. In one 4,194,304-
shot `bh2-T` comparison, its 64-worker CPU cost rose from 143.20 to 174.26
seconds despite removing duplicate preparation. That intermediate result is
retained separately in the evidence and is not the final implementation.

For fixed prepared reference certificates, the final design gives each runtime
local ownership counters that pin the same immutable resource tables and
contractions. Generic and manual fallback paths keep their existing ownership.
A controlled three-pair comparison against the intermediate design retained the
same memory sharing and scientific results:

| Workers | Intermediate CPU (s) | Final CPU (s) | CPU reduction | Intermediate sampling (s) | Final sampling (s) |
|---|---:|---:|---:|---:|---:|
| 32 | 149.067 | 135.130 | 9.3% | 10.668 | 10.902 |
| 64 | 157.109 | 143.072 | 8.9% | 7.173 | 6.585 |

Peak RSS remained approximately 33.6 MiB and 42.6 MiB at 32 and 64 workers.
This ablation supports reducing shared ownership bookkeeping in the draw loop.
The 32-worker elapsed result also illustrates that lower CPU cost does not
guarantee a faster individual shared-server measurement. Hardware performance
counters were unavailable, so specific cache-coherence or NUMA costs are not
claimed as measured quantities.

## First-use preparation and resident memory

All four tasks require two native boundary compilations in the optimized
version, independently of worker count. The baseline performs 64 compilations
at 32 workers and 128 at 64 workers. The following one-window medians include
worker preparation plus 4,096 ordinary draws.

| Task | Before 32 (s) | After 32 (s) | Before 64 (s) | After 64 (s) | 64-worker gain |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 0.0377 | 0.0141 | 0.0776 | 0.0219 | 3.55x |
| `bh20-T` | 0.1992 | 0.0568 | 0.3658 | 0.1002 | 3.65x |
| `rp2-d3` | 0.0402 | 0.0200 | 0.0762 | 0.0302 | 2.52x |
| `wg-T` | 0.0913 | 0.0392 | 0.1346 | 0.0710 | 1.90x |

Peak RSS below uses the 4,194,304-shot paired observations, in MiB.

| Task | Before 32 | After 32 | Before 64 | After 64 | 64-worker reduction |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 95.8 | 33.7 | 169.2 | 42.6 | 74.8% |
| `bh20-T` | 546.4 | 92.0 | 1071.3 | 147.9 | 86.2% |
| `rp2-d3` | 102.5 | 38.6 | 182.2 | 51.5 | 71.8% |
| `wg-T` | 259.0 | 66.0 | 494.7 | 102.5 | 79.3% |

Memory still grows with worker count because runtime metadata and mutable
draw state remain private. Shared plans remove the dominant duplicated
prepared data; they do not make every worker allocation shared.

## Larger-sample scaling

The following tables show optimized fresh-process medians. The scaling gain
is the 32-worker sampling time divided by the 64-worker sampling time.

### 1,048,576 shots

| Task | 32 sampling (s) | 64 sampling (s) | Scaling gain | 32 CLI (s) | 64 CLI (s) |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 2.427 | 1.776 | 1.37x | 2.560 | 1.895 |
| `bh20-T` | 4.353 | 2.324 | 1.87x | 4.551 | 2.525 |
| `rp2-d3` | 2.018 | 1.375 | 1.47x | 2.114 | 1.477 |
| `wg-T` | 2.362 | 1.849 | 1.28x | 2.512 | 2.077 |

### 4,194,304 shots

| Task | 32 sampling (s) | 64 sampling (s) | Scaling gain | 32 CLI (s) | 64 CLI (s) |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 10.002 | 7.573 | 1.32x | 10.098 | 7.686 |
| `bh20-T` | 13.993 | 9.924 | 1.41x | 14.325 | 10.219 |
| `rp2-d3` | 8.565 | 5.767 | 1.49x | 8.666 | 5.922 |
| `wg-T` | 10.019 | 7.221 | 1.39x | 10.166 | 7.397 |

### Direct baseline comparison

Ratios below divide baseline medians by optimized medians for the same
task, shot count, and worker count. Values below 1 indicate slower observed
elapsed time. They are not confidence intervals.

| Shots | Task | Sampling gain, 32 | Sampling gain, 64 | CLI gain, 32 | CLI gain, 64 |
|---:|---|---:|---:|---:|---:|
| 1,048,576 | `bh2-T` | 1.05x | 1.20x | 1.07x | 1.25x |
| 1,048,576 | `bh20-T` | 0.92x | 1.24x | 1.01x | 1.51x |
| 1,048,576 | `rp2-d3` | 1.13x | 1.26x | 1.15x | 1.33x |
| 1,048,576 | `wg-T` | 1.01x | 1.12x | 1.08x | 1.25x |
| 4,194,304 | `bh2-T` | 1.05x | 1.10x | 1.06x | 1.11x |
| 4,194,304 | `bh20-T` | 1.09x | 1.08x | 1.11x | 1.12x |
| 4,194,304 | `rp2-d3` | 1.03x | 1.12x | 1.04x | 1.12x |
| 4,194,304 | `wg-T` | 1.00x | 1.01x | 1.01x | 1.06x |

The one-minute load average during the final paired sweep ranged from 69.31
to 88.45. Near-unity ratios and individual regressions must be
interpreted with this shared-server variability and the version-order pattern
described above. Cold-start compilation and memory reductions are much larger
than the remaining throughput differences.


## Consistency and remaining limits

All 72 final before/after pairs match exactly across 23 scientific and metadata
fields, including estimates, standard errors, scientific representations,
weights, efficiency, fingerprints, and draw bookkeeping. Both worker counts
also match for every task/shot-count/seed combination in each version. Native
query counts match as well. These comparisons use complete parsed results;
canonical JSON digests retain the comparisons in the compact evidence.

Two additional correctness checks use ordinary public interfaces and reversed
version order. Four single-worker example pairs at 4,096 shots preserve the
existing serial random stream exactly. A public `run` call loads `bh2-T` and
enables record 0 through its task mapping, then draws 65,536 shots. The complete
requested-record array and all scientific fields agree across the old/new and
32/64-worker combinations. Large record arrays are represented by count/hash
descriptors in the compact evidence after full-array comparison. These extra
checks are excluded from the performance tables.

The large-shot jobs amortize initial preparation enough that 64 workers beat
32 workers for all four examples in this experiment. Scaling remains below
2x. The [previous phase measurements](parallel-sampling-scaling.md) identified
serial seed generation and ordered accumulation; those phases remain in the
same execution path, and preserving ordered arithmetic keeps the exact
reproducibility contract. This experiment does not remeasure their individual
time shares. Dispatch/wake-up overhead, private per-draw allocation, and shared
machine scheduling also remain; hardware-specific attribution would require
additional measurements. For 1,048,576-shot `bh20-T` at 32 workers, sampling
rises from 4.009 to 4.353 seconds (+8.6%, or 0.92x speed), while CPU time falls
from 70.68 to 66.35 seconds (-6.1%) and the complete CLI median is approximately
unchanged. These observations do not establish the cause of the elapsed
regression. The evidence supports the preparation and memory improvements
without asserting universal elapsed-time gains.

## Release validation

The release native build matches all 97 source hashes in its build manifest.
All 26 focused parallel tests pass. The full regression suite passes 379 tests
in 261.855 seconds with no skips or failures, including installed-interface
checks after building and installing the wheel and example decoder fixture.
The separately built UBSan extension passes all 32 parallel and weighted-noise
tests in 4.413 seconds, with no skips or sanitizer findings. All 97 native
source hashes match between the local checkout, release build, and UBSan build.
Logs are retained in the experiment directory as `wheel-build.log`,
`wheel-install.log`, `full-tests.log`, and `sanitized-tests.log`. The benchmark
observer's one-second timeout and Ctrl-C cleanup were also checked against
ordinary public task invocations; neither left a task child running.
Repository documentation checks and strict MkDocs also pass after synchronizing
the final README, report, evidence, and sampling guide. The remote check covers
32 maintained documents; the local count is 33 because it includes the existing
untracked `AGENTS.md`. Logs are `docs-check-final.log` and `docs-build-final.log`.

## Reproduction

From the built source checkout, the larger scaling experiment uses the existing
driver and normal public task CLI:

```bash
PYTHONPATH=core:. python3 scripts/benchmark_parallel_sampling.py \
  examples/bh2-T examples/bh20-T examples/rp2-d3 examples/wg-T \
  --workers 32 64 --shots 4194304 --repeats 3 \
  --seed 20261016 --label shared-plans --output scaling-4194304.json
```

Repeat with `--shots 1048576` and `--shots 4096`. For controlled old/new
comparisons, use the same absolute example directories in both checkouts and
alternate versions for each task, worker count, and seed. The paired experiment
uses the same driver with one task, one worker count, and `--repeats 1` per
observation; it does not introduce a sampling API.
