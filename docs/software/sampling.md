# Sampling methods

A run draws random faults, and for tasks with external decoders also draws the
measurement bits those decoders read. Two independent choices control these draws:
the *noise proposal* (physical or importance) and the *history sampler*
(`affine_support` or `exact_marginal`). Everything else in a draw is exact.

## Physical sampling

By default faults are drawn from the noise declared in the circuit.
No configuration is needed:

```python
from magna import run

result = run("physical.magna", ideal_state="ideal.magna", shots=1000, seed=7)
```

Each draw contributes its exact accepted mass and error mass; estimates are plain means.
Physical sampling is the right choice when failures are frequent enough to appear
in the available shots.

## Importance sampling

When logical failures come from rare combinations of faults, most physical draws contain
no fault and contribute nothing to the error estimate.
Importance sampling multiplies every eligible error probability by a common `proposal_scale`
and corrects each draw by its exact likelihood ratio, so the estimates stay unbiased.

```python
from magna import run

result = run(
    "physical.magna",
    ideal_state="ideal.magna",
    sampling={"method": "importance", "proposal_scale": "10"},
    shots=1000,
    seed=7,
)
```

On the command line, save the same object as `sampling.json`:

```sh
magna run physical.magna --ideal-state ideal.magna --sampling sampling.json --shots 1000 --seed 7
```

`sampling` is run configuration: it changes neither the circuit's noise model nor the task.
`{"method": "physical"}` is the explicit default.

### What the scale changes

For every eligible noise category with physical total error probability `p`,
the proposal uses `q = proposal_scale * p`. Relative rates between channels and the
proportions of alternatives within a channel are preserved.

| Category | Under the proposal |
|---|---|
| Unguarded Pauli channel or correlated chain with `0 < p < 1` | Total probability scaled |
| Measurement report flip with `0 < p < 1` | Scaled |
| Probability exactly 0 or 1 | Unchanged |
| Channel guarded by a `rec` or `ctrl` bit ([conditional noise](noise.md#conditional-noise)) | Unchanged |
| Decoder-input bits, ideal target, recovery | Unchanged |

`proposal_scale` is an exact rational, given as a decimal or fraction string.
Every scaled probability must stay below one; otherwise the configuration is rejected.
The result's `sampling` section reports how many categories were scaled and how many were left unchanged.

### Weights

A category with physical probability `p` and proposal probability `q` contributes
a factor `p/q` when its error occurs and `(1-p)/(1-q)` when it does not.
The product over categories is the draw's noise weight `w`.
Estimates are `mean(w * acceptance)` and `mean(w * error)`, and the conditional error is their ratio.
Weighted acceptance is not clipped: finite-sample estimates can exceed one even for an always-accepted task.
`weights.ess` reports `(sum w)^2 / sum(w^2)`.

### Choosing the scale

A good scale makes the typical draw contain about as many faults as a typical failure needs.
Too small a scale wastes draws on fault-free runs; too large a scale produces heavy-tailed
weights and expensive multi-fault draws.
Compare the standard error of your target quantity at equal computation time,
and state the scale with every importance-sampled result.

`tests/fixtures/importance-sampling` has an exact error of 0.001 and a supplied scale of 100.

## History samplers

For tasks with external decoders, guarded noise or `sample_records`, some measurement bits
must be sampled. Two methods are available; both are unbiased and both use the same task.

| Method | How it draws | Weight |
|---|---|---|
| `affine_support` (default) | Proposes a history uniformly from a linear family known to contain every possible history, then evaluates that history exactly | Physical history probability divided by proposal probability |
| `exact_marginal` | Draws each needed bit from its exact conditional probability | None |

The affine-support method avoids a conditional marginal query for every sampled bit, at the price
of a reweighting that adds variance; which one wins depends on the task, and kappa and the pilot
below measure it. Select with `--history-sampler exact_marginal` or `options={"history_sampler": "exact_marginal"}`.
The runner never switches methods on its own.
Tasks whose feedback is all affine and that request no records sample only faults,
and this setting has no effect; `compilation.sampled_history` in the inspect report says which applies.

With `affine_support`, the complete weight of a draw is the noise weight times the history weight;
`weights.ess` includes both factors. A history whose proposal lands on an interference zero
contributes zero weight and stays in the averages.

### Kappa and progress

```sh
magna run physical.magna --task task.json --shots 10000 --progress-every 1000
```

Each progress line reports *kappa*: the estimated variance of the current conditional error
under `affine_support` relative to `exact_marginal` with the same noise proposal.
Kappa 3 means roughly three times as many draws are needed for the same precision;
per-draw time is a separate factor.
The final result carries `efficiency.kappa` with its components and a `status`.
Kappa is `null` when fewer than two draws completed, acceptance is zero, or the diagnostic cannot be formed.
It is a running point estimate, not a confidence bound.

### Pilot comparison

To decide between the two history samplers before a long run:

```sh
magna compare-samplers physical.magna --task task.json --pilot-shots 256 --seed 7 --output pilot.json
```

```python
from magna import compare_samplers

pilot = compare_samplers("physical.magna", task="task.json", pilot_shots=256, seed=7)
print(pilot["gamma"], pilot["recommended_sampler"])
```

The pilot runs the requested number of draws with each method, using seeds `seed` and `seed + 1`.
Gamma is kappa times the affine-support time per draw divided by the exact-marginal time per draw;
above one it favours `exact_marginal`.
The recommendation is advice only. The next production run uses whatever you configure.
Both pilot results, including their preparation cost and resource summaries, are kept under `pilots`.
A failed pilot reports `status: "failed"`; an inconclusive one `status: "inconclusive"`.

## Parallel sampling

Use the same run interface with a positive `workers` count:

```python
from magna import compile_circuit, run

compiled = compile_circuit("physical.magna", task="task.json")
result = run(compiled, shots=10000, seed=7, workers=4)
```

```sh
magna run physical.magna --task task.json --shots 10000 --seed 7 --workers 4
python scripts/run_task.py examples/bh2-T --shots 10000 --workers 4 --output result.json
```

The portable task runner also accepts `"workers": 4` in `run.json`; the command-line
option overrides it. Parallelism applies to physical and importance sampling and
both history samplers. It does not change the exact per-draw calculation or the
estimator. Results and requested records are accumulated in draw order.

Each native worker owns its mutable execution state and prepared networks. Workers
are reused by repeated runs of the same compiled circuit. The compiled circuit retains
the largest worker cache it has used, so its memory cost grows with that count.
Small runs may be slower because of worker preparation
and scheduling. Start with a worker count no larger than the available CPU cores,
then compare `timings.sampling_seconds` and `timings.total_seconds` on your task.

Python decoder callbacks run with the Python interpreter lock. Their call order across
draws is unspecified; callbacks should return a deterministic result from their inputs
and configuration, without depending on shared mutable state or invocation order.
Callbacks that release the interpreter lock can overlap and must be thread-safe.
Accessing the same compiled circuit from a callback or signal handler while a parallel
batch is active raises `RuntimeError`.
Heavy Python decoders can limit the speedup of native sampling.

`max_queries` is shared across the run, and `max_seconds` measures elapsed wall time.
Other resource bounds apply to each worker's networks and caches. On failure, statistics
retain the completed prefix in draw order. Workers may already have evaluated later
draws; those results are discarded, and their work is included in resource counters.

## Seeds and reproducibility

With one worker, the same package version and seed reproduce faults, histories and
decoder calls exactly; the default `workers=1` preserves the sequential random stream.
Parallel runs use a separate random stream per draw. With deterministic callbacks,
the same version and seed reproduce the estimates and records across worker counts
greater than one and across `progress_every` settings. Parallel and sequential runs
need not draw the same faults or histories. Resource-limit failures can depend on
thread scheduling, and callback invocation order is not reproducible in parallel.
Changing the version can change the seeded stream even when the physical distribution is unchanged,
so compare versions through estimates and uncertainties, not individual draws.
