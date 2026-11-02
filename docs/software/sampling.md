# Sampling methods

A run draws random faults, and for tasks with external decoders also draws the
measurement bits those decoders read. Two independent choices control these draws:
the *noise proposal* (physical or importance) and the *history sampler*
(`affine_support` or `exact_marginal`). Everything else in a draw is exact.

The experimental [CUDA sampler](cuda.md) runs physical and importance sampling of supported tasks on an NVIDIA GPU.

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

Set `workers` to a positive integer to use multiple CPU cores (default 1):

```python
from magna import run

result = run("physical.magna", task="task.json", shots=10000, seed=7, workers=4)
```

```sh
magna run physical.magna --task task.json --shots 10000 --seed 7 --workers 4
python scripts/run_task.py examples/bh2-T --shots 10000 --workers 4 --output result.json
```

The task runner also accepts `"workers": 4` in `run.json`; the CLI overrides it.
Parallel sampling supports physical and importance sampling and both history samplers.

To compare worker counts through the same task runner, save a benchmark report:

```sh
python scripts/benchmark_parallel_sampling.py examples/bh2-T --workers 1 2 4 8 --shots 10000 --repeats 3 --output scaling.json
```

Start with no more workers than available CPU cores and compare
`timings.sampling_seconds` on your task; small runs may be slower. More workers
can increase memory use; lowering `workers` on an existing compiled circuit may
not release that memory. See [resource limits](options.md) when setting budgets.

Python decoders can limit speedup. For reproducible results, decoders should return
the same output for the same inputs and configuration, without relying on call
order or shared mutable state. Decoders that release Python's interpreter lock
must be thread-safe. Calling the same compiled circuit from a decoder or signal
handler during parallel sampling raises `RuntimeError`.

On POSIX, only use `fork` when sampling and decoder calls are idle.

## Seeds and reproducibility

The default `workers=1` keeps the existing seeded results. For completed parallel
runs with deterministic decoders, the same package version, environment and seed
reproduce estimates and records across worker counts greater than one and across
`progress_every` settings. Results with one worker need not match parallel results.
Decoder call order and runs that hit resource limits are not reproducible in this way.
Changing the version can change the seeded stream even when the physical distribution is unchanged,
so compare versions through estimates and uncertainties, not individual draws.
