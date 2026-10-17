# Parallel sampling: final merge review

Reviewed production version: `46a4769` on `feat/parallel-sampling`, compared with
local main `56847478c840395735146ef1e1ccdc3d8aedc10f`. Review date: 2026-10-17.
The subsequent report integration changes documentation and evidence only.

**Finding: no mandatory code fix or unresolved merge blocker was identified in
this review.** This is a bounded assessment of the inspected code and recorded
checks, not a claim of exhaustive concurrency or platform coverage. No merge or
push was performed, and main was not changed.

## Review scope

The source review covered the complete production diff from main: the existing
Python/CLI `workers` argument, native batch dispatch, persistent thread ownership,
shared prepared plans, worker-local mutable state and resource ownership,
cross-worker query budgets, ordered accumulation, callback/GIL exception transfer,
signal cancellation, idle-pool shutdown and POSIX fork handling, and decoder
argument/configuration handling. The public default remains `workers=1`.

Existing regressions cover deterministic parallel streams across worker counts
and progress partitions, physical and importance sampling, both history methods,
shared preparation and reconfiguration, query/time limits, callback errors and
overlap, interpreter shutdown, fork, and signal interruption. No new production
change was needed during this review.

The reporting issue was corrected: every throughput, speedup, elapsed-time,
variability and memory table, plus the figure, now selects optimized `dk` from
round B. Every performance row states its round. The seven native rows retain
round A and its measured commit. Same-worker optimization gains use B's paired
baseline; A and B are not pooled to calculate an optimization gain.

## Validation evidence

| Check | Result and scope |
|---|---|
| Source identity | All 128 recorded production inputs match `46a4769` and the clean wheel-build snapshot after newline normalization |
| Full source regression, retained from the decoder follow-up | 382 tests collected: 370 passed; 12 installation tests skipped because no installed interpreter was selected; includes the existing native UBSan checks |
| New clean wheel build | PEP 517 build from `git archive 46a4769`; GCC 11.4 / CPython 3.10.12 on ghz |
| New installed-interface acceptance | All 12 tests passed, no skips; executed with the isolated wheel interpreter and `-I` |
| New installed parallel regressions | All 27 tests passed, no skips; the existing parallel test module ran against the installed wheel with `-I` |
| Installed payload identity | All 31 package files exactly match the newly built wheel; package and native module resolve inside the new environment |
| Scientific consistency | The original A evidence and B paired evidence remain unchanged; the selected rows have 120 exact cross-worker comparisons, and B has 42 exact before/after pairs including its controls |
| Table integration | All 56 aggregates recomputed from the 168 selected observations; all 280 primary performance-table cells match the evidence |
| Documentation | Local link/language checks and strict MkDocs build pass after report integration; figure visually checked |
| Git hygiene | Whitespace/error-marker checks pass; the integration contains no production-code changes |

The previous source regression is reused because its production inputs match the
reviewed version. The fresh installation checks close its twelve skipped tests;
this is not presented as a new single invocation of all 382 tests. The 27 wheel
parallel tests additionally verify that packaging preserves the tested behavior.

The wheel SHA-256 is
`7388f0ee068dd534fe260ffd561f8581bc3a65104580f960c40de4d44c703780`.
The installed native binary SHA-256 is
`3da9d3967ca77c8058e292dc7401e083b2a82503076f583c84c94e4abe7746fa`.
This packaging build is separate from the release binary used for performance
measurements; it does not replace those timing records.

Exact commands, timestamps, installed paths, payload hashes and test output are
in the [final-check evidence](benchmark-results/parallel-sampling-final-check-20261017.json).
Raw build/install logs and the wheel remain under
`/tmp/magna-final-check-20261017` on ghz.

### Installation test prerequisites

The external decoder fixture must be installed separately from
`tests/fixtures/basic/decoder_plugin`. The first isolated acceptance run had three
failures because this fixture was absent; after installing it, all twelve tests
passed. The fixture deliberately remains outside the production wheel. The
initial failure log and the successful commands are both retained in the evidence.

The server lacks `ensurepip` and has pip 22, so the environment was created with
`venv --without-pip`; its interpreter executed the existing pip package for an
offline wheel install. Runtime checks used `-I`, with `PYTHONPATH` removed. No
server system package or unrelated environment was changed.

For an environment with working `venv` and pip, build and install the production
wheel and the decoder fixture, then select that interpreter explicitly:

```bash
MAGNA_INSTALLED_PYTHON=/absolute/path/to/wheel-env/bin/python \
  /absolute/path/to/wheel-env/bin/python -I -m unittest discover \
  -s tests/core -p test_installed_interface.py -v
/absolute/path/to/wheel-env/bin/python -I -m unittest discover \
  -s tests/core -p test_parallel_sampling.py -v
```

## Remaining limits

- The consolidated benchmark is explicitly a combination of two measured
  versions/rounds. The seven native tasks were not rerun as a full sweep on
  `46a4769`. Their old timings are historical evidence, not newly measured
  performance claims for the optimized commit.
- ghz is shared. Load, SMT contention and serial timing variability limit small
  performance claims. B's original native control has small 32/64-worker losses;
  its later fixed-core/fixed-hash serial pairs differ by approximately 1%. These
  observations do not prove that every workload is regression-free.
- Python callbacks still serialize under the GIL. `dk` peaks at eight workers
  in B; more workers are not a universal speed improvement. Cached thread pools
  retain their largest size until the compiled object is released, as documented.
- This final runtime acceptance covers Linux x86-64 / CPython 3.10 / GCC 11.4.
  It does not add Windows, macOS, other Python-version, or ThreadSanitizer coverage.

These limits are documented and do not identify a mandatory code fix from the
evidence collected here. No claim of a fresh same-version eight-task benchmark
should be made without running that additional experiment.
