# Running and reading results

`magna run` and `run(...)` in Python return one JSON-compatible result object.
This page explains how to read it. The full field list is in the [schema reference](schemas.md#result-schema).

## Check the status first

```python
result = run("physical.magna", task="task.json", shots=1000, seed=7)
assert result["status"] == "complete", result["failures"]
```

| Field | Meaning |
|---|---|
| `status` | `complete`, or `failed` when preparation or any draw failed |
| `requested`, `completed`, `failed`, `not_attempted` | Shot accounting |
| `failures` | A list of `{phase, shot, type, message}` entries; `phase` is `preparation` or `sampling`, and `type` is the exception name, such as `ResourceLimit` for an exceeded [resource option](options.md) |

When a run fails part way, the main `estimates` are set to `null` and the provisional
aggregates move to `partial_statistics`. Do not use partial statistics as if they were complete.
A preparation failure (resource budget exceeded during compilation) reports `completed` 0,
`failed` 0 and every shot as `not_attempted`; the reason is the `preparation` entry in `failures`.

## Read the estimates

```json
"estimates": {"acceptance": 0.9921, "error": 0.00098, "conditional_error": 0.000988, "moments": []},
"standard_errors": {"acceptance": 0.0028, "error": 0.00031, "conditional_error": 0.00031, "moments": []}
```

| Estimate | Meaning |
|---|---|
| `acceptance` | Mean accepted mass: the probability that a run is accepted |
| `error` | Mean joint accepted-and-wrong mass |
| `conditional_error` | `error / acceptance`: the logical error rate of accepted runs; with an ideal target, the infidelity |
| `moments` | Accepted joint Pauli moments for the task's `observables`, not divided by acceptance |

Acceptance and error are means of exact per-draw contributions, each weighted by its
importance weight when a proposal is used. The conditional error is the ratio of the two aggregates,
not the mean of per-draw ratios. Its standard error uses the paired delta method.
`conditional_error` is `null` when the aggregate acceptance is zero.

Standard errors describe variation over the sampled faults and decoder inputs.
A task with no noise, no external decoder and no requested records has zero standard error.
With fewer than two completed draws, standard errors are `null`.

## Very small and very large values

Every estimate has a scientific twin, `estimates_scientific` and `standard_errors_scientific`,
in which each number is `{"mantissa": m, "exponent2": e}` meaning `m * 2**e`.
These fields preserve values below or above the range of ordinary doubles.
If an ordinary field would underflow or overflow, it is `null` and its name is listed in
`numerics.underflow_fields` or `numerics.overflow_fields`; read its scientific twin instead.
A run accumulates its statistics as draws complete and keeps no per-draw data,
so its memory does not grow with the number of shots. For exact per-draw values,
draw them one at a time with `CompiledCircuit.sample`.

Read rare-event results from the scientific fields, and do not treat `null` as zero:
`null` means out of range or undefined, never an exact zero.

## Weights and efficiency

| Field | Meaning |
|---|---|
| `weights.mean` | Mean importance weight (1 under physical sampling with integrated histories) |
| `weights.ess` | Effective sample size `(sum w)^2 / sum(w^2)` of the weights alone |
| `efficiency.kappa` | Estimated variance of the conditional error under `affine_support` relative to `exact_marginal` with the same noise proposal; `null` when not estimable |
| `history_sampler` | The configured history sampler, `affine_support` (default) or `exact_marginal`. It is reported for every run, but it draws bits only when `resources.compilation.sampled_history` is true |

A weight ESS close to the shot count means the proposal did not distort the sample much;
it is not by itself a precision guarantee for the error estimate.
Use the standard error of the quantity you care about.
[Sampling methods](sampling.md) explains kappa and the choice of history sampler.

## Provenance and cost

| Group | Fields |
|---|---|
| Identity | `schema_version`, `software_version`, `fingerprints` of circuit, task, controllers, options, sampling, ideal state and recovery |
| Execution | `route` (`linear` or `nonlinear`), `method`, `seed`, requested `workers`, `sampling` configuration, `evaluation` (output mapping, target and recovery identity, metric convention) |
| Timing | `timings.compile_seconds`, `preparation_seconds` (reference compilation), `sampling_seconds` (the draw loop), `query_seconds`, `decoder_seconds`, `support_seconds` |
| Resources | `resources.compilation` with the prepared query families and `resource_summary`, and the configured `limits` |

Report preparation and sampling time separately; preparation is paid once per task.
When `run` receives a compiled circuit, `timings.total_seconds`, `query_seconds`, `decoder_seconds`
and the `resources` counters cover only that call, while `preparation_seconds` is the compiled circuit's
one-time preparation.

For parallel runs, `total_seconds` and `sampling_seconds` are elapsed wall time.
Sampling time includes startup overhead. `compile_seconds`,
`query_seconds`, `decoder_seconds` and `support_seconds` sum the work of all workers,
so they can exceed wall time. The compilation report does not include all memory
used by parallel sampling. See [parallel sampling](sampling.md#parallel-sampling).

## Inspect before running

`magna inspect` (or `compile_circuit(...).inspect()`) returns the parsed circuit,
the task, the route, the decoder bindings with their record dependencies, and the
`compilation` report with its resource summary. Use it to confirm that a task is what you
intended and to see its resource requirement before committing to a long run.
`compilation.sampled_history` tells you whether any measurement bits will be sampled
(external decoders, guarded noise or `sample_records`), and `compilation.conditional_noise`
whether guarded channels are present.

## Progress reports

```sh
magna run physical.magna --task task.json --shots 10000 --progress-every 1000
```

Progress lines go to standard error, so standard output keeps only the summary, or the JSON with `--json`.
Each line shows the completed count and the running kappa estimate, followed by a reminder that
high kappa favours `--history-sampler exact_marginal`.
In Python pass `progress_every=1000` to `run`.
