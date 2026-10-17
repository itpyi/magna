# Parallel sampling: merge preparation measurements

This report consolidates two measured versions on 2026-10-17. The seven
native-only task rows retain round A; the `dk` row uses the optimized candidate
from round B. Every table and the figure below use this selection consistently.
There are 168 selected observations: 147 from A and 21 from B. This is not a
single-version eight-task rerun, and the native rows are not claimed to have
been remeasured on the optimized commit.

## Test versions and rounds

| Round | Tested version | Repetitions and seeds | Use in this summary |
|---|---|---|---|
| A | `5cffaf8` | A1: 20261017; A2: 20261018; A3: 20261019 | Seven native tasks; 147 selected observations |
| B | Optimized snapshot committed as `46a4769`; paired baseline `5533543` (same production code as `5cffaf8`) | B1: 20261020; B2: 20261021; B3: 20261022 | Optimized `dk`; 21 selected observations |

Round A originally measured all eight tasks (168 observations). Round B measured
both versions of `dk` at all seven worker counts and a `bh20-T` control at 1, 32,
and 64 workers (60 observations). The extra single-core and fixed-hash controls
in the [decoder follow-up](decoder-callback-benchmark.md) are diagnostics; they
are not substituted into these tables. All 128 recorded production inputs in
the optimized snapshot match commit `46a4769` after newline normalization.

Scaling is normalized to the **same task, version, and round's one-worker
median**. Compare the paired before/after measurements within B to assess the
optimization; do not divide B timings by A timings. The original A `dk` results
(best 1.98x at 4 workers; 1.28x at 64) remain in the immutable
[round A evidence](benchmark-results/parallel-sampling-merge-20261017.json).

## Workloads and method

| Category | Task | Shots per observation |
|---|---|---:|
| Distillation | `bh2-T` | 1,048,576 |
| Distillation | `bh20-T` | 1,048,576 |
| Cultivation | `rp2-d3` | 1,048,576 |
| Cultivation | `rp2-d5` | 1,048,576 |
| Cultivation | `color3` | 1,048,576 |
| Cultivation | `color5` | 1,048,576 |
| Transfer | `wg-T` | 1,048,576 |
| Python decoder | `dk` | 262,144 |

Each task keeps its packaged physical-noise proposal, `affine_support` history
sampler, circuit, target, and resource settings. Only shots, seed, and workers
are overridden through `scripts/benchmark_parallel_sampling.py`, which invokes
`scripts/run_task.py` and public `magna.run`. No private sampling API is called.
Seeds and tested versions are listed above. Within each task, the three
worker orders are 1/2/4/8/16/32/64, 64/32/16/8/4/2/1, and 8/32/2/64/4/16/1.
Task order rotates between repetitions. Round B also alternates the adjacent
baseline/candidate execution order. Three repetitions do not constitute
a fully balanced order design or a confidence interval.

In round A, the original four-task sweep completed before the additional cultivation and
decoder tasks ran. The additional tasks first ran separate 4,096-shot pilots
at 1 and 64 workers. The slower pilot selected a fixed power-of-two shot count
up to 1,048,576, targeting at most 40 seconds of extrapolated sampling time.
Pilot observations are excluded from all result tables. Shot counts never
change between workers or repetitions of the same task. Round B keeps `dk`
at 262,144 shots throughout.

Sampling time includes first-use worker preparation, seeds, scheduling, exact
per-draw calculations, and ordered statistical accumulation. It excludes
initial task loading/preparation. CLI wall time also includes interpreter
startup, initial compilation, JSON output, and teardown. All tables show
medians of three observations; throughput is shots divided by median sampling
time, and speedup is the same task's 1-worker median divided by its N-worker
median. These are scaling ratios within each row's version and round, not ratios against main.

## Hardware and operating conditions

| Item | Recorded environment |
|---|---|
| Host | `ghz` (`GHZ`) |
| CPU | 2 x Intel Xeon Platinum 8358P, nominal 2.60 GHz |
| CPU topology | 64 physical cores, 128 logical CPUs, SMT 2, 2 NUMA nodes |
| Process affinity | CPUs 0–63; one logical CPU per physical core |
| Memory visible to Linux | 503.51 GiB |
| OS | Linux-5.15.0-187-generic-x86_64-with-glibc2.35 |
| Python | Python 3.10.12 |
| Compiler | GCC 11.4.0; C++17, `-O3 -DNDEBUG -pthread` |

Campaign-specific observations (Asia/Shanghai, all on 2026-10-17):

| Condition | Round A | Round B paired sweep |
|---|---|---|
| Measurement interval | 12:56:43–13:26:54 | 13:49:20–13:57:29 |
| Available memory | 177.73–202.22 GiB | 160.26–179.91 GiB |
| One-minute load average | 72.80–79.07 | 76.22–79.85 |
| Aggregate CPU busy, min / median / max | 55.5% / 58.6% / 75.0% | Not summarized by the B harness |
| Host swap in use | 6.27–6.27 GiB | See B host monitoring; no derived swap-activity claim |
| Host swap activity | 64 pages in; 0 pages out | Not summarized by the B harness |

This is a shared server, not an exclusive-core experiment. The same affinity
mask applies at every worker count; threads may migrate within that mask.
Other users may run on those CPUs or their SMT siblings. Affinity limits our
own SMT use but does not reserve cores. No other users' jobs were stopped or
reconfigured. Our benchmark processes ran sequentially, without another project
benchmark, native build, or regression suite running alongside them. Frequency
and NUMA memory policy were not changed by the harness.

Load average is distinct from CPU utilization. Available memory and swap
counters are host-wide observations, not this process's allocation. GNU time
measures each task process's individual peak RSS below. One-second host samples,
per-run loads, CPU time, context switches, and page faults are retained in the
evidence or source logs. No runs are selected or removed based on speed.

## Sampling throughput

Unit: **thousand shots per second**; higher is better.

| Task | Round | Shots | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers | 32 workers | 64 workers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bh2-T` | A | 1,048,576 | 33.69 | 54.85 | 109.36 | 187.66 | 292.73 | 397.97 | 393.86 |
| `bh20-T` | A | 1,048,576 | 27.02 | 32.28 | 59.29 | 110.86 | 187.38 | 288.53 | 321.22 |
| `rp2-d3` | A | 1,048,576 | 42.91 | 67.40 | 129.29 | 227.47 | 349.17 | 442.79 | 447.12 |
| `rp2-d5` | A | 1,048,576 | 40.35 | 54.11 | 110.89 | 204.06 | 323.03 | 448.47 | 427.54 |
| `color3` | A | 1,048,576 | 45.71 | 71.90 | 130.97 | 229.27 | 344.70 | 423.04 | 422.94 |
| `color5` | A | 1,048,576 | 30.38 | 49.04 | 96.35 | 175.13 | 294.63 | 419.03 | 405.69 |
| `wg-T` | A | 1,048,576 | 38.98 | 64.68 | 118.69 | 204.39 | 323.43 | 426.24 | 418.73 |
| `dk` | B | 262,144 | 21.23 | 30.69 | 54.44 | 58.17 | 48.08 | 43.75 | 43.69 |

## Sampling speedup

Unit: **times the same task's one-worker throughput**.

| Task | Round | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers | 32 workers | 64 workers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bh2-T` | A | 1.00x | 1.63x | 3.25x | 5.57x | 8.69x | 11.81x | 11.69x |
| `bh20-T` | A | 1.00x | 1.19x | 2.19x | 4.10x | 6.93x | 10.68x | 11.89x |
| `rp2-d3` | A | 1.00x | 1.57x | 3.01x | 5.30x | 8.14x | 10.32x | 10.42x |
| `rp2-d5` | A | 1.00x | 1.34x | 2.75x | 5.06x | 8.01x | 11.12x | 10.60x |
| `color3` | A | 1.00x | 1.57x | 2.87x | 5.02x | 7.54x | 9.26x | 9.25x |
| `color5` | A | 1.00x | 1.61x | 3.17x | 5.76x | 9.70x | 13.79x | 13.35x |
| `wg-T` | A | 1.00x | 1.66x | 3.04x | 5.24x | 8.30x | 10.93x | 10.74x |
| `dk` | B | 1.00x | 1.45x | 2.56x | 2.74x | 2.26x | 2.06x | 2.06x |

![Sampling speedup across the requested task families](benchmark-results/parallel-sampling-speedup-current-20261017.png)

| Task | Round | Best measured worker count | Best speedup | 64-worker speedup |
|---|---|---:|---:|---:|
| `bh2-T` | A | 32 | 11.81x | 11.69x |
| `bh20-T` | A | 64 | 11.89x | 11.89x |
| `rp2-d3` | A | 64 | 10.42x | 10.42x |
| `rp2-d5` | A | 32 | 11.12x | 10.60x |
| `color3` | A | 32 | 9.26x | 9.25x |
| `color5` | A | 32 | 13.79x | 13.35x |
| `wg-T` | A | 32 | 10.93x | 10.74x |
| `dk` | B | 8 | 2.74x | 2.06x |

For the seven native-only tasks, the best observed speedups range from
9.26x to 13.79x. Moving from 32 to 64 workers changes
throughput by a factor of 0.95–1.11 across those tasks,
so this session does not show a doubling of throughput at the largest count.
The optimized round B `dk` task peaks at 8 workers (2.74x), compared with
2.06x at 64 workers. Within B's paired before/after comparison, the adapter
optimization improves throughput by 1.84x at 8 workers, 1.78x at 32, and 1.79x
at 64; these gains use the baseline measured in B, not the historical A row.
Python callback scaling remains limited above eight workers in this experiment.

These worker choices describe this workload and shared-host session only. They
are not universal defaults. More workers can lose to scheduling, serial seed
generation, ordered accumulation, and the cost of keeping multiple runtimes.
Different tasks use different shot counts, so compare worker ratios within a
row rather than treating absolute throughput across circuits as equal work.

The `dk` task invokes its bundled pure-Python Steane CSS decoder during terminal
recovery. Python callbacks acquire the interpreter lock; invocation order is
unspecified, while this decoder returns a deterministic result for its inputs.
Its row is included even when adding threads gives little benefit or makes it
slower. This is evidence for that complete task, not a measurement isolating the
GIL's exact cost. Reported decoder seconds sum worker intervals and can include
waiting for the GIL; they must not be read as a wall-time percentage.

## Sampling elapsed time and variability

Unit: seconds; lower is better.

| Task | Round | Shots | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers | 32 workers | 64 workers |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bh2-T` | A | 1,048,576 | 31.122 | 19.116 | 9.588 | 5.587 | 3.582 | 2.635 | 2.662 |
| `bh20-T` | A | 1,048,576 | 38.807 | 32.485 | 17.684 | 9.459 | 5.596 | 3.634 | 3.264 |
| `rp2-d3` | A | 1,048,576 | 24.435 | 15.556 | 8.110 | 4.610 | 3.003 | 2.368 | 2.345 |
| `rp2-d5` | A | 1,048,576 | 25.989 | 19.377 | 9.456 | 5.139 | 3.246 | 2.338 | 2.453 |
| `color3` | A | 1,048,576 | 22.941 | 14.584 | 8.006 | 4.574 | 3.042 | 2.479 | 2.479 |
| `color5` | A | 1,048,576 | 34.516 | 21.382 | 10.883 | 5.987 | 3.559 | 2.502 | 2.585 |
| `wg-T` | A | 1,048,576 | 26.900 | 16.211 | 8.835 | 5.130 | 3.242 | 2.460 | 2.504 |
| `dk` | B | 262,144 | 12.349 | 8.541 | 4.815 | 4.506 | 5.452 | 5.991 | 5.999 |

Observed minimum–maximum sampling seconds for selected counts:

| Task | Round | 1 worker | 32 workers | 64 workers |
|---|---|---:|---:|---:|
| `bh2-T` | A | 16.834–31.140 | 2.606–2.992 | 2.580–2.903 |
| `bh20-T` | A | 34.756–61.733 | 3.546–3.741 | 3.210–3.416 |
| `rp2-d3` | A | 15.902–24.519 | 2.304–2.421 | 2.179–2.374 |
| `rp2-d5` | A | 21.046–30.967 | 2.274–2.389 | 2.339–2.537 |
| `color3` | A | 17.929–23.928 | 2.341–2.494 | 2.247–2.545 |
| `color5` | A | 28.137–35.996 | 2.407–2.513 | 2.547–2.747 |
| `wg-T` | A | 16.264–27.980 | 2.377–2.501 | 2.285–2.692 |
| `dk` | B | 11.804–12.623 | 5.750–6.045 | 5.973–6.001 |

Individual times for every count remain in the JSON. These ranges describe
three observations and are not statistical confidence intervals.

## Complete CLI wall time

Unit: seconds; includes startup, initial preparation, serialization, teardown.

| Task | Round | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers | 32 workers | 64 workers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bh2-T` | A | 31.258 | 19.254 | 9.730 | 5.740 | 3.717 | 2.779 | 2.827 |
| `bh20-T` | A | 39.111 | 32.741 | 17.886 | 9.773 | 5.875 | 3.922 | 3.575 |
| `rp2-d3` | A | 24.594 | 15.708 | 8.268 | 4.747 | 3.149 | 2.558 | 2.493 |
| `rp2-d5` | A | 26.219 | 19.601 | 9.671 | 5.376 | 3.488 | 2.587 | 2.744 |
| `color3` | A | 23.196 | 14.828 | 8.255 | 4.819 | 3.304 | 2.722 | 2.734 |
| `color5` | A | 34.750 | 21.530 | 11.095 | 6.206 | 3.797 | 2.712 | 2.836 |
| `wg-T` | A | 27.115 | 16.340 | 9.052 | 5.346 | 3.495 | 2.699 | 2.767 |
| `dk` | B | 12.504 | 8.681 | 4.930 | 4.660 | 5.623 | 6.150 | 6.169 |

## Peak resident memory

Unit: MiB; median of each process's peak RSS, observed by GNU time.

| Task | Round | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers | 32 workers | 64 workers |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bh2-T` | A | 24.6 | 25.5 | 26.0 | 27.1 | 29.3 | 34.0 | 43.1 |
| `bh20-T` | A | 43.3 | 44.0 | 44.2 | 50.6 | 64.1 | 92.2 | 147.7 |
| `rp2-d3` | A | 25.3 | 26.6 | 27.0 | 28.7 | 31.9 | 38.6 | 51.7 |
| `rp2-d5` | A | 33.3 | 34.0 | 34.7 | 38.8 | 47.4 | 64.3 | 97.9 |
| `color3` | A | 25.0 | 25.7 | 26.6 | 28.3 | 31.4 | 38.3 | 51.5 |
| `color5` | A | 32.9 | 33.9 | 34.9 | 39.2 | 48.2 | 65.8 | 101.0 |
| `wg-T` | A | 32.3 | 33.2 | 34.1 | 38.8 | 48.0 | 66.3 | 103.3 |
| `dk` | B | 25.9 | 26.7 | 27.5 | 29.3 | 32.9 | 40.3 | 54.7 |

## Consistency and provenance

All 168 selected observations completed their requested shots. Round A's
original eight-task suite has 120 exact comparisons between parallel worker
counts. The seven selected native tasks account for 105 of those checks. Round
B adds 15 such checks for the selected optimized `dk`, giving 120 checks for the
consolidated selection. Comparisons exclude only timings, diagnostic resources,
and the requested worker count; native query counts also agree. Serial and
parallel random streams intentionally differ and are not claimed to match.

The paired decoder experiment additionally verified 42 before/after pairs,
including its separate controls, and 36 cross-worker comparisons across both
versions. All statistical result fields and native query counts in each matched
before/after pair agree exactly. See the [decoder follow-up](decoder-callback-benchmark.md)
for every observation, control, and measurement limitation.

The round A source SHA-256 is
`af020c92f512b9781a012f440f9d73a4ad55907996074530192c5e7d2c0dc0f7`;
its native binary SHA-256 is
`1bb8a57d1487853177db4f1eea8031a159b5ba7dda10866b413a74f915aa98e9`.
The round B candidate source SHA-256 is
`765ed12141b9fc7ea39ec82512e5231dd8f4895682f3666c7e609d4c27a62542`;
its native binary SHA-256 is
`73641364462d4715df291e90c6ab86ff3a9f883ca38c228f6b6ad03c3eeac7e1`.
These hashes identify the Linux measurement snapshots. The normalized source
hashes bind the B snapshot to the subsequently recorded commit `46a4769`.

The [consolidated table data](benchmark-results/parallel-sampling-merge-current-20261017.json)
records each row's commit, round, seeds, and source evidence file hash.
The immutable [round A evidence](benchmark-results/parallel-sampling-merge-20261017.json)
and [round B evidence](benchmark-results/decoder-callback-20261017.json) preserve
all original observations. Raw logs remain under
`/tmp/magna-merge-benchmark-20261017` and `/tmp/magna-decoder-20261017` on ghz.
The [final merge review](parallel-sampling-final-review.md) records the checked
source version, packaging validation, and remaining limits. This integration
changes reporting only.

## Reproduction

To reproduce a specific row, check out its recorded version and build it
normally, then run the existing driver. For the native rows, use `5cffaf8` and
the A seeds. To exercise the final implementation, use `46a4769`; historical
native-row timings are not predictions for that checkout. For example:

```bash
PYTHONPATH=core:. taskset -c 0-63 python3 scripts/benchmark_parallel_sampling.py \
  examples/bh2-T examples/bh20-T examples/rp2-d3 examples/wg-T \
  --workers 1 2 4 8 16 32 64 --shots 1048576 --repeats 3 \
  --seed 20261017 --label merge-preparation --output scaling.json
```

For the optimized `dk` row, run on `46a4769`:

```bash
PYTHONPATH=core:. taskset -c 0-63 python3 scripts/benchmark_parallel_sampling.py \
  examples/dk --workers 1 2 4 8 16 32 64 --shots 262144 --repeats 3 \
  --seed 20261020 --label optimized-dk --output dk-scaling.json
```

Use each additional task's shot count from the workload table. On a different
machine, select an appropriate CPU mask from its actual topology. The exact
recorded three worker orders and rotated task orders are in the evidence;
the short command above uses the existing driver's standard rotation instead.
All timing processes explicitly set `PYTHONPATH` to their chosen source snapshot
to prevent importing an unrelated installed package.
