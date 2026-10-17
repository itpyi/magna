# Decoder callback overhead, 2026-10-17

This follow-up optimizes the existing `ControllerBinding.invoke` path used by
public `magna.run` and the task runner. The `dk` decoder itself and all public
sampling interfaces remain unchanged. The baseline is `5533543`, whose production
code matches the [eight-task scaling report](parallel-sampling-merge-summary.md).
The new comparisons below are fresh paired measurements, not ratios against that
earlier session's timings.
The optimized snapshot was subsequently committed as `46a4769`; its 128 recorded
production inputs match that commit after newline normalization. The
[consolidated benchmark summary](parallel-sampling-merge-summary.md) uses the
candidate measurements from this report as round B, alongside native rows from round A.

## Diagnosis and implementation

The bundled `dk` decoder handles six syndrome bits and returns fourteen correction
bits. The Python adapter used to build a source-location error prefix, deserialize
an empty JSON configuration, and validate both tuples with Python generator loops
on every invocation. These operations run while the callback holds the GIL.
See [CPython's threading rules](https://docs.python.org/3.10/c-api/init.html#thread-state-and-the-global-interpreter-lock).

In 65,536-shot serial profiles through the ordinary task CLI, cumulative time in
`ControllerBinding.invoke` fell from 1.790 s to 0.371 s.
The decoder body was about 0.13 s in both profiles. Profiling amplifies Python
function-call costs; these figures diagnose overhead and are not production
speedup measurements. The timed sweeps below run without a profiler or callback
instrumentation.

The retained changes are:

- Check bit tuples in the existing native binding layer, preserving exact `int`
  or `bool` type checks, rejection of non-bits and oversized integers, and the
  public wrapper's error messages and output normalization.
- Allocate a fresh empty dict directly for the canonical empty configuration.
  Nonempty JSON configurations still receive an independent deep snapshot per call.
- Format source-location errors only when an error occurs, preserving exception
  chaining and the declared output-width check.

Every required decoder invocation still runs. No decoded values are memoized,
no circuit is substituted, and no GIL scheduling parameter or public option is
introduced. Callbacks that release the GIL can still overlap.

A diagnostic public-API wrapper observed 8,192 different Python thread states for
8,192 parallel calls. This follows the attach/delete behavior in
[CPython 3.10.12](https://github.com/python/cpython/blob/v3.10.12/Python/pystate.c#L1383-L1467).
An experimental window-scoped reuse reduced that count to 8/64/128 at 4/32/64
workers, but did not produce a clear timing gain. A separate GIL-entry gate also
showed no consistent gain. Neither threading change is retained. Their single-run
pilot data remain in the evidence; they are exploratory ablations, not precise
comparisons. The measurable opportunity was shortening the work performed under
the GIL, rather than assuming thread-state allocation was the dominant cost.

## Paired benchmark method

All timed observations invoke the existing `scripts/benchmark_parallel_sampling.py`
driver, which launches `scripts/run_task.py` in a fresh process. Both versions read
the same absolute `examples/dk` and `examples/bh20-T` input paths from the baseline
snapshot. Packaged noise, decoder, resource limits, and `affine_support` settings
are unchanged. Each observation uses 262,144 shots; seeds are 20261020, 20261021,
and 20261022. Each worker/task pair alternates version order, and worker and task
order rotate between repetitions. There are 42 `dk` and 18 `bh20-T` observations.

Sampling seconds include worker first use, seeds, scheduling, sampling, and
ordered accumulation. CLI wall time, preparation, CPU time, switches, RSS, and
individual results are also retained in the JSON. Tables use medians of three;
the gain is baseline median divided by candidate median at the same worker count.
Observed ranges and the three seed-paired ratios are supplied below.
Three repetitions do not establish a confidence interval.

Host: ghz, two Intel Xeon Platinum 8358P CPUs, 64 physical cores, 128 logical CPUs,
two NUMA nodes, 503.51 GiB RAM. Both versions use Python 3.10.12 and
GCC 11.4 with C++17, `-O3 -DNDEBUG -pthread`. Every process uses the same CPU 0–63
affinity mask, one logical CPU per physical core. The cores are not reserved;
other users and their SMT siblings remain active. Available memory was
160.26–179.91 GiB, and one-minute load average was
76.22–79.85. Timings ran from 2026-10-17 13:49:20
to 2026-10-17 13:57:29 Asia/Shanghai. This project's builds and tests ran
outside that interval. Small differences should not be interpreted as stable
gains or losses; serial timings in particular varied substantially on this shared host.

## dk sampling results

Seconds are lower-is-better; throughput is thousand shots/second.

| Workers | Before (s) | After (s) | Before/after gain | After throughput | After scaling vs 1 worker |
|---:|---:|---:|---:|---:|---:|
| 1 | 14.558 | 12.349 | 1.18x | 21.23 | 1.00x |
| 2 | 10.320 | 8.541 | 1.21x | 30.69 | 1.45x |
| 4 | 7.625 | 4.815 | 1.58x | 54.44 | 2.56x |
| 8 | 8.302 | 4.506 | 1.84x | 58.17 | 2.74x |
| 16 | 8.799 | 5.452 | 1.61x | 48.08 | 2.26x |
| 32 | 10.689 | 5.991 | 1.78x | 43.75 | 2.06x |
| 64 | 10.731 | 5.999 | 1.79x | 43.69 | 2.06x |

The best measured candidate throughput is at eight workers (2.74x its own
one-worker median). At 32 and 64 workers, same-worker before/after gains are
1.78x and 1.79x, respectively: approximately 44% less sampling time. The candidate
still loses throughput above eight workers, so this change reduces the callback
cost without removing the Python scaling limit. These are workload-specific
observations; the noisy serial baseline also affects the scaling column.

| Workers | Before range (s) | After range (s) | Three seed-paired gains |
|---:|---:|---:|---|
| 1 | 8.940–14.770 | 11.804–12.623 | 0.76x, 1.20x, 1.15x |
| 2 | 9.732–10.635 | 8.256–9.433 | 1.21x, 1.18x, 1.13x |
| 4 | 7.494–8.595 | 4.428–5.370 | 1.40x, 1.78x, 1.72x |
| 8 | 7.207–8.371 | 4.151–4.704 | 1.53x, 1.84x, 2.02x |
| 16 | 8.379–10.108 | 4.453–5.728 | 1.88x, 1.76x, 1.61x |
| 32 | 10.554–10.884 | 5.750–6.045 | 1.84x, 1.77x, 1.82x |
| 64 | 10.494–11.218 | 5.973–6.001 | 1.79x, 1.75x, 1.88x |

## Resource observations

Each cell gives before / after medians. Decoder time sums overlapping worker
intervals, including time waiting to acquire the GIL. It is not a wall-time
percentage or an isolated measure of the decoder's CPU cost.

| Workers | Process CPU (s) | Context switches | Summed decoder time (s) | Peak RSS (MiB) |
|---:|---:|---:|---:|---:|
| 1 | 14.72 / 12.50 | 18 / 25 | 3.78 / 1.37 | 25.8 / 25.9 |
| 2 | 20.30 / 16.64 | 27,017 / 36,394 | 6.62 / 4.31 | 26.7 / 26.7 |
| 4 | 21.30 / 17.00 | 277,273 / 145,826 | 15.85 / 5.11 | 27.4 / 27.5 |
| 8 | 21.59 / 17.59 | 276,600 / 277,003 | 49.64 / 20.01 | 29.4 / 29.3 |
| 16 | 22.89 / 19.30 | 274,798 / 278,722 | 120.29 / 67.55 | 32.8 / 32.9 |
| 32 | 26.12 / 20.67 | 272,357 / 282,240 | 308.76 / 161.95 | 40.2 / 40.3 |
| 64 | 26.69 / 21.30 | 279,033 / 288,267 | 617.31 / 326.22 | 54.7 / 54.7 |

The remaining scaling limit includes serialized Python execution and GIL handoffs,
alongside native scheduling and ordered accumulation. These experiments do not
assign an exact percentage to each remaining cause, and they do not establish
that arbitrary Python decoders will obtain the same improvement.

## Native-only control

`bh20-T` does not invoke the changed callback wrapper. These shared-host observations
check for a large unrelated regression; they are not evidence for accelerating its
unchanged sampling path.

| Workers | Before (s) | After (s) | Before/after ratio |
|---:|---:|---:|---:|
| 1 | 11.466 | 15.605 | 0.73x |
| 32 | 0.941 | 0.956 | 0.98x |
| 64 | 0.946 | 0.997 | 0.95x |

### Same-core serial confirmation

The broad CPU mask permits the scheduler to choose a different physical core for
each serial process. Because its serial control varied substantially, a separate
confirmation pins both versions to CPU 0 only. It uses the same public driver,
262,144 shots, three additional seeds (20261030–20261032), and alternating version
order. These twelve observations are reported separately, not substituted into
the original table. They ran after the regression and documentation checks.

| Task | Before (s) | After (s) | Median ratio | Three seed-paired ratios |
|---|---:|---:|---:|---|
| dk | 13.918 | 11.690 | 1.19x | 1.18x, 1.19x, 1.20x |
| bh20-T | 7.837 | 12.824 | 0.61x | 1.01x, 0.60x, 1.03x |

Pinning a single core did not remove the native control's variability: both
versions produced approximately 7.7 s and 15 s observations. The 0.61x ratio of
medians above alone does not establish a code regression; two pairs
are close to parity while the middle pair spans different timing regimes.

### Fixed-hash native control

A final diagnostic keeps CPU 0 and sampling seed 20261032 fixed, explicitly sets
`PYTHONHASHSEED` to 0, 1, or 2, and repeats all three hash seeds with the source
version order reversed. Every observation still uses the same public CLI and
262,144 shots. This tests whether an unrelated process-level source of variation
can explain the native-only result. All twelve observations are retained below;
they are separate from the original sweep and the three-seed confirmation.
Per-second CPU 0/64 counters and CPU 0 frequency are included in the evidence.

| Repetition | Python hash seed | Before (s) | After (s) | Before/after ratio |
|---:|---:|---:|---:|---:|
| 1 | 0 | 15.236 | 15.089 | 1.01x |
| 1 | 1 | 15.215 | 15.226 | 1.00x |
| 1 | 2 | 15.059 | 15.021 | 1.00x |
| 2 | 0 | 15.522 | 15.465 | 1.00x |
| 2 | 1 | 15.244 | 15.388 | 0.99x |
| 2 | 2 | 15.394 | 15.250 | 1.01x |

All six ratios in this final control are within approximately 1% of parity.
The earlier large serial slowdown did not reproduce with this design. No clear
dependence on the three hash seeds emerged, so Python hashing is not established
as the cause of the earlier variability. During these twelve observations, CPU 0
was fully busy and its SMT sibling CPU 64 was 95–100% busy (median 100%); CPU 0's
reported frequency was 2.883–3.067 GHz. The experiment was still sharing a physical
core. Earlier sibling activity was not recorded, so attributing the earlier
timing transitions specifically to SMT contention would require further evidence.
The original 32/64-worker control differences are retained above; this final
serial check does not establish the absence of small changes at every worker count.

## Correctness and reproduction

All 60 observations completed. All 30 matched before/after pairs agree exactly
on every result field except timings, diagnostic resources, and workers; native
query counts also match. This includes the serial stream. Another 36 checks
confirm equality between parallel worker counts within each version. Serial and
parallel streams remain intentionally different.
All twelve same-core confirmation observations completed, and their six additional
before/after pairs also agree exactly, including query counts.
The twelve fixed-hash native observations completed and all six corresponding
before/after pairs agree exactly as well: 42 before/after pairs in total.

Focused regressions cover fresh empty and nested configurations, exact bit types
and ranges, callback exception causes and source locations, callbacks that release
the GIL, repeated runs, shutdown, fork, signal interruption, and resource limits.
The complete regression suite ran 382 tests: 370 passed and
12 installed-wheel tests were skipped because `MAGNA_INSTALLED_PYTHON` was unset.
The documentation checks and strict MkDocs build also passed. Commands and results are recorded
in the [machine-readable evidence](benchmark-results/decoder-callback-20261017.json).
The later [final merge review](parallel-sampling-final-review.md) closes the
installed-wheel validation gap: all 12 installation tests and all 27 parallel
regressions pass against a fresh wheel built from `46a4769`.
The initial regression attempt could not build the test-only oracle extension
because its default `clang++` was unavailable; setting `CXX=g++` selected the same
compiler family as the production build for the successful rerun.

To reproduce one configuration through the public runner:

```bash
PYTHONPATH=core:. taskset -c 0-63 python3 scripts/benchmark_parallel_sampling.py \
  examples/dk --shots 262144 --workers 1 2 4 8 16 32 64 \
  --repeats 3 --seed 20261020 --output decoder-scaling.json
```

Select a CPU mask appropriate to the actual host. The evidence contains the exact
paired commands and ordering, all results, binary/native-source hashes, source
digests, profiles, and exploratory pilots. The raw experiment is retained under
`/tmp/magna-decoder-20261017` on ghz.
