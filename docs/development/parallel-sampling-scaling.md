# Parallel sampling scaling, 2026-10-16

This follow-up measures the existing `bh2-T`, `bh20-T`, `rp2-d3`, and `wg-T`
tasks at 1, 8, 16, 32, and 64 workers on `ghz`. The original parallel
implementation, commit `2fe4b8b`, reaches its best sampling time at 32 workers
for all four tasks; increasing to 64 workers makes each task slower.
The optimized implementation improves sampling by 1.08x to 1.43x and complete
CLI invocations by 1.10x to 1.30x in alternating before/after comparisons at
32 and 64 workers. Fresh-process sampling still favors 32 workers for these
262,144-shot workloads. First-use worker preparation and serial accumulation
limit further scaling.

## Method and environment

Fresh-process scaling measurements launch the existing `scripts/run_task.py` in
a fresh Python process, with 262,144 shots. The runner uses the public `magna.run`
interface. The example configurations, physical noise proposal, and
`affine_support` sampler are unchanged. Three repetitions use seeds 20261016,
20261017, and 20261018, rotating worker order between repetitions.

The machine has two Intel Xeon Platinum 8358P processors, 64 physical cores,
128 logical CPUs, two NUMA nodes, and 503 GiB RAM. Builds use Python 3.10.12
and GCC 11.4.0 with C++17, `-O3 -DNDEBUG -pthread`. Runs are unpinned on a
shared server. The baseline one-minute load average ranged from 84.48 to 94.96.
Separate single-CPU diagnostic and validation builds overlapped some measurements;
benchmark jobs and CPU-intensive test suites did not overlap. Other users'
workloads were left running. The optimized full sweep had one-minute loads of
86.48 to 100.70. A request for 64 workers does not reserve 64 physical cores.

Sampling time includes additional worker runtime preparation, random seed
generation, scheduling, sampling, and ordered statistical accumulation. CLI wall
time additionally includes interpreter startup, task loading, initial compilation
and preparation, JSON serialization, and process teardown. Tables use medians.

The baseline snapshot is `/tmp/magna-parallel-20261016-1252/candidate`; it was
preserved without edits. Experiment outputs are under
`/tmp/magna-scaling-20261016-1349` on `ghz`. Both implementations load the same
absolute example paths so that source-path-dependent fingerprints are comparable.
The [compact evidence](benchmark-results/parallel-sampling-scaling-20261016.json)
retains individual observations, exact scientific-result fingerprints, phase
measurements, environment and native-binary provenance. The optimized interpreter
and extension import paths were checked explicitly against the new candidate
directory, excluding an accidental import of the baseline or an installed wheel.

## Original implementation

| Task | 1 worker (s) | 8 workers (s) | 16 workers (s) | 32 workers (s) | 64 workers (s) |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 6.0863 | 1.5028 | 1.0634 | 0.9052 | 0.9874 |
| `bh20-T` | 13.8467 | 2.5957 | 1.7510 | 1.3987 | 1.8895 |
| `rp2-d3` | 4.0245 | 1.3574 | 0.9954 | 0.9112 | 1.1144 |
| `wg-T` | 4.0514 | 1.4724 | 1.0184 | 0.9753 | 1.1422 |

The 64-worker sampling regressions relative to 32 workers are 9.1%, 35.1%,
22.3%, and 17.1%, respectively. These runs do not support selecting 64 workers
merely because the machine has 64 physical cores.

## Optimized fresh-process scaling

The release implementation retains worker threads with the compiled circuit,
claims short contiguous groups of draws, generates each seed window while holding
the Python interpreter lock once, and reuses statistical scratch storage. It keeps
the same random seeds, draw ordering, arithmetic update order, and 4,096-draw bound.

| Task | 1 worker (s) | 8 workers (s) | 16 workers (s) | 32 workers (s) | 64 workers (s) |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 7.8691 | 1.5234 | 0.9358 | 0.8754 | 1.0283 |
| `bh20-T` | 15.4202 | 2.5918 | 1.6806 | 1.3002 | 2.0589 |
| `rp2-d3` | 4.5156 | 1.1289 | 0.7448 | 0.5868 | 0.6063 |
| `wg-T` | 6.4559 | 1.2783 | 0.8539 | 0.7002 | 0.9605 |

Relative to one worker in this same sweep, 16-worker sampling speedups are
6.06x to 9.18x, 32-worker speedups are 7.69x to 11.86x, and 64-worker speedups
are 6.72x to 7.65x. These ratios describe the measured shared-server conditions.
Serial times changed materially between the two sweeps, so comparing those
separate tables does not reliably isolate the optimization's effect.

For the direct comparison below, each old/new pair uses the same task, worker
count, shots and seed. Version order alternates; each configuration has three
pairs. All eight sampling and CLI medians improve. CPU seconds per complete
invocation fall by approximately 5% to 13%.

| Task | Workers | Before sampling (s) | After sampling (s) | Sampling gain | CLI gain |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 32 | 0.8541 | 0.6850 | 1.25x | 1.20x |
| `bh2-T` | 64 | 1.0125 | 0.7064 | 1.43x | 1.29x |
| `bh20-T` | 32 | 1.4093 | 1.2062 | 1.17x | 1.10x |
| `bh20-T` | 64 | 1.6936 | 1.5697 | 1.08x | 1.18x |
| `rp2-d3` | 32 | 0.8188 | 0.5843 | 1.40x | 1.30x |
| `rp2-d3` | 64 | 0.8032 | 0.6421 | 1.25x | 1.14x |
| `wg-T` | 32 | 0.8799 | 0.6860 | 1.28x | 1.20x |
| `wg-T` | 64 | 1.0866 | 0.8900 | 1.22x | 1.15x |

A separate ablation retained only the pool and draw-group scheduling changes.
Its paired sampling gains ranged from 0.94x to 1.32x, including a 64-worker
`bh20-T` regression. Removing thread creation alone was insufficient; reducing
serial seed and statistics overhead was also useful.

A further isolated experiment released the pool's shared mutex before notifying
its workers. Three alternating pairs at 32 and 64 workers gave current/variant
sampling ratios of 0.98x and 1.04x for `bh2-T`, and 0.95x and 0.75x for
`bh20-T`. This variant was rejected; the measured release retains its tested
wake-up ordering. These results also caution against attributing the remaining
64-worker cost to that mutex without more evidence.

## Diagnostic evidence

A separate `strace -f -c -e trace=clone,clone3,futex` run of the same public CLI,
with `bh2-T`, 262,144 shots and 64 workers, recorded exactly 4,096 successful
`clone3` calls. The implementation creates and joins 64 threads for each of
64 windows of 4,096 draws. The optimized equivalent records only 64 `clone3`
calls. Trace timings are excluded from speedup tables because tracing changes
scheduling. Futex calls increase from 1,725 to 17,469: persistent workers sleep
and wake between windows. These perturbed counts describe the synchronization
mechanism, not its uninstrumented runtime cost.

The benchmark driver records differences in
`resource.getrusage(resource.RUSAGE_CHILDREN)` before and after each CLI
subprocess. The resulting CPU seconds include user and system time. Context
switches cover the complete invocation, not just its sampling interval. These
observations were initially collected with an external copy of the same driver;
the resource capture is now available in the existing committed driver on Unix.

| Task | CPU seconds, 8 workers | CPU seconds, 32 workers | CPU seconds, 64 workers | Voluntary switches, 32 workers | Voluntary switches, 64 workers |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 8.86 | 10.43 | 12.00 | 8,751 | 29,119 |
| `bh20-T` | 17.06 | 23.26 | 30.39 | 21,253 | 77,927 |
| `rp2-d3` | 7.82 | 10.17 | 10.77 | 7,434 | 26,328 |
| `wg-T` | 8.60 | 10.99 | 13.42 | 9,796 | 36,948 |

The rising CPU cost and switch counts establish substantial parallel overhead.
They do not, by themselves, distinguish allocation, cache-coherence, scheduler,
or memory-bandwidth costs. Hardware performance counters were unavailable:
`perf` is not installed and the host's `perf_event_paranoid` value is 4.

Separate instrumented copies of both releases timestamp setup, seed generation,
worker execution and ordered result accumulation while still running the normal
CLI. One diagnostic run per configuration is excluded from release medians.
Worker execution includes dispatch, waiting and diagnostic merging; workers can
already be sampling during dispatch.

| Task | Workers | Before seed (s) | After seed (s) | Before accumulation (s) | After accumulation (s) |
|---|---:|---:|---:|---:|---:|
| `bh2-T` | 32 | 0.1423 | 0.1190 | 0.3046 | 0.2132 |
| `bh2-T` | 64 | 0.1447 | 0.1099 | 0.3033 | 0.2207 |
| `bh20-T` | 32 | 0.1298 | 0.0918 | 0.3438 | 0.1878 |
| `bh20-T` | 64 | 0.1382 | 0.0777 | 0.3455 | 0.1850 |

The remaining serial seed and accumulation phases consume about 0.26 to 0.33
seconds in these diagnostic runs. Their absolute cost does not shrink with worker
count, which limits speedup even after optimizing worker execution. Accumulation
remains ordered to retain exactly reproducible results; a floating-point tree
reduction would change that contract.

First-window cost is especially important for `bh20-T`. In the optimized
diagnostic run, the first window's worker phase took 0.3336 seconds with 32
workers and 1.0013 seconds with 64 workers. The remaining 63 worker phases
together took 0.6726 and 0.5284 seconds, respectively: warmed 64-worker execution
was faster, but the first-use cost outweighed that advantage. This phase includes
lazy worker plan preparation and cache warm-up, not just thread launch.

A separate ordinary 64-worker `bh20-T` invocation reached about 1.05 GiB peak
resident memory with either implementation (1,096,776 KiB before and 1,095,704
KiB after). The pool does not remove private worker plans. Host RAM capacity was
not exhausted; NUMA, SMT and memory-bandwidth effects were not isolated by these
measurements and should not be presented as established causes.

## Reusing a compiled circuit

An additional experiment loads each unchanged task with `scripts.run_task.load`,
calls public `compile_circuit`, and then calls public `run` four times on that
same object. Each task/worker configuration has its own fresh process and compiled
circuit. The first run uses seed 20261016; the subsequent three use seeds
20261016, 20261017 and 20261018. Every run has 262,144 shots. No private sampling
entry point is called.

The first-run column is one observation, while the reuse column is the median
of three. Both columns report sampling only; initial `compile_circuit` time,
interpreter startup and teardown are excluded. Shared-server variation remains,
so differences between the columns should not all be attributed to cache warming.

| Task | Workers | First sampling (s) | Reused sampling median (s) |
|---|---:|---:|---:|
| `bh2-T` | 1 | 6.9879 | 4.1540 |
| `bh2-T` | 16 | 0.8648 | 0.8686 |
| `bh2-T` | 32 | 0.6631 | 0.6221 |
| `bh2-T` | 64 | 0.5677 | 0.5662 |
| `bh20-T` | 1 | 15.1395 | 15.5634 |
| `bh20-T` | 16 | 1.5223 | 1.4640 |
| `bh20-T` | 32 | 1.5069 | 1.0794 |
| `bh20-T` | 64 | 2.3037 | 1.0815 |

For `bh20-T`, reuse removes much of the 64-worker first-use penalty, but its
32- and 64-worker medians are essentially tied. `bh2-T` gains about 10% from
32 to 64 warmed workers. Reusing a compiled object is useful for repeated
sampling; selecting 64 workers still requires measuring the intended workload.
All 32 results match their corresponding fresh-process scientific results exactly.

## Requested record storage

A separate public-API comparison runs the existing `bh2-T` circuit with its
original task plus `sample_records: [0]`, requesting 65,536 draws and 32 workers.
Its ideal-state path is resolved to the same source file when loading the task
as a dictionary. Three alternating old/new pairs use the same three seeds as
the main experiment. This is an additional requested-output case, distinct from
the unchanged benchmark configurations above.

| Measurement | Before median (s) | After median (s) | Gain |
|---|---:|---:|---:|
| Sampling | 15.2227 | 0.2857 | 53.29x |
| Public `run` call, including preparation | 15.3187 | 0.3860 | 39.68x |

The original implementation reserved capacity for exactly one additional record
on every draw, repeatedly moving all previously stored record containers. The
optimized geometric capacity growth removes that quadratic behavior. Every
requested record and all other scientific fields match exactly for all three
seeds. The public-call wall time excludes interpreter startup and JSON writing;
it is not the complete CLI wall metric used in the main tables. The evidence
stores record-array hashes and counts rather than duplicating the large arrays.

## Correctness and validation

All 60 optimized fresh-process observations match the corresponding original
result exactly across 23 scientific fields, including estimates, scientific
representations, standard errors, weights, efficiency, sampled records,
fingerprints and numerical metadata. This includes all 12 serial observations.
Within each full sweep, the 8-, 16-, 32- and 64-worker outputs also match exactly
for every task and seed: 36 comparisons per implementation. Timing, worker count
and mutable diagnostic counters are intentionally excluded from this equality
check. Parallel and serial random streams remain distinct.

The final production source passed all 374 regression tests with zero skips in
the complete dependency environment, including standalone C++ checks and the
normal wheel build/install tests. All 35 focused parallel and native file-runtime
tests passed. A separate extension built with UBSan passed 27 parallel and
weighted-noise tests; standalone foundation checks also exercised statistics
failure atomicity under UBSan. Documentation checks and the strict MkDocs build
passed. The diagnostic and rejected ablation builds are separate from this tested
production binary.

## Reproduce

With each revision built separately, invoke the existing benchmark driver
against the same example directories:

```sh
PYTHONPATH=core:. python3 scripts/benchmark_parallel_sampling.py \
  /path/to/shared/examples/bh2-T /path/to/shared/examples/bh20-T \
  /path/to/shared/examples/rp2-d3 /path/to/shared/examples/wg-T \
  --shots 262144 --workers 1 8 16 32 64 --repeats 3 --seed 20261016 \
  --label scaling --output scaling.json
```

One ordinary 64-worker run is:

```sh
PYTHONPATH=core:. python3 scripts/run_task.py examples/bh20-T \
  --shots 262144 --workers 64 --seed 20261016 --output result.json
```

To reuse the same preparation and worker pool through the public interface:

```python
from pathlib import Path
from magna import compile_circuit, run
from scripts.run_task import load

circuit, settings, _ = load(Path("examples/bh20-T"))
compiled = compile_circuit(circuit, **settings)
first = run(compiled, shots=262144, seed=20261016, workers=64)
again = run(compiled, shots=262144, seed=20261017, workers=64)
```
