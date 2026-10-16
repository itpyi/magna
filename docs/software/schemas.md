# Schemas

## Task schema

Version 1. All fields are optional except that at most one error definition may be given.

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | `1` | Schema identifier |
| `outputs` | list of int | Physical qubits kept as quantum output, in order; default all qubits |
| `error_observables` | list of int | `OBSERVABLE_INCLUDE` indices whose parities must all be zero for success; requires empty `outputs` and `observables` |
| `ideal_state` | path or object | Target preparation file, or `{"circuit": path, "outputs": [...], "postselection": {"w": bit}}` |
| `recovery` | path | Noiseless terminal recovery file; requires `ideal_state` |
| `sample_records` | list of int | Absolute record indices to sample and return |
| `observables` | list of `[x, z]` | Output Pauli masks in output order |

Paths are relative to the task file. `ideal_state` and `error_observables` are mutually exclusive
error definitions, and only `ideal_state` may be combined with `recovery`.
Acceptance is declared by `POSTSELECT` in the circuit, not in the task.
See [tasks and targets](tasks.md).

## Controller binding schema

```json
{
  "schema_version": 1,
  "bindings": {
    "name": {"import": "package.module:function", "version": "1.0", "config": {}}
  }
}
```

See [classical feedback](controllers.md#register-functions-from-a-json-file).

## Sampling schema

`{"method": "physical"}` or `{"method": "importance", "proposal_scale": "10"}`,
where the scale is a positive rational written as a decimal or fraction string.
See [sampling methods](sampling.md#importance-sampling).

## Result schema

Version 2, returned by `run` and written by `magna run`.
The enumeration result and the `magna validate` report carry the same version.

| Field | Contents |
|---|---|
| `schema_version`, `software_version` | Identifiers |
| `status` | `complete` or `failed` |
| `requested`, `completed`, `failed`, `not_attempted` | Shot counts |
| `seed`, `workers`, `route`, `method`, `history_sampler` | Execution identity; `workers` is the requested native thread count (default 1); `route` is `linear` or `nonlinear` |
| `estimates` | `acceptance`, `error`, `conditional_error`, `moments` |
| `standard_errors` | Standard errors of the same fields; `null` with fewer than two draws |
| `estimates_scientific`, `standard_errors_scientific` | The same values as `{"mantissa", "exponent2"}` objects |
| `numerics` | `underflow_fields` and `overflow_fields`: the ordinary fields that are out of range and `null` |
| `partial_statistics` | Provisional aggregates of a failed run; present only when `status` is `failed` |
| `weights` | `mean`, `mean_scientific`, `ess`, the weight `convention` and `ess_scope` |
| `efficiency` | `kappa` and the two variance coefficients whose ratio it is (`variance_coefficient` for `affine_support`, `exact_marginal_variance_coefficient` for `exact_marginal`), `ess`, `rho`, `mean_weight`, zero-weight count, `status`, a `comparison` note, and scientific twins |
| `sampling` | The noise proposal, the counts of scaled, guarded and deterministic categories, and `exact_marginal_records`, which is set only under `exact_marginal` |
| `evaluation` | `output_mapping`, `ideal_state` and `recovery` identity, `error_metric`, `observables` convention |
| `normalization`, `uncertainty` | Text describing the estimator conventions |
| `fingerprints` | Hashes of `circuit`, `task`, `controllers`, `options`, `sampling`, `ideal_state`, `recovery` |
| `timings` | `total_seconds`, `compile_seconds`, `preparation_seconds`, `sampling_seconds`, `query_seconds`, `decoder_seconds`, `support_seconds` |
| `resources` | `queries`, `compilations`, configured `limits` and the `compilation` report |
| `failures` | List of `{phase, shot, type, message}` |
| `sampled_records` | Per-draw values of the task's `sample_records`, or `null` |

See [running and reading results](results.md).

Version 2 renamed `sampling.born_records` to `sampling.exact_marginal_records` and
`efficiency.born_variance_coefficient` to `efficiency.exact_marginal_variance_coefficient`,
reworded `efficiency.comparison` and `weights.convention` for `exact_marginal`, and removed the
always-empty `resources.last_query_diagnostics`. For an `error_observables` task, the circuit
fingerprint and `evaluation` now describe the circuit as written ([tasks](tasks.md#choose-the-error-definition)).

## Per-draw sample

`CompiledCircuit.sample(rng)` returns one draw. Its `acceptance`, `error` and `moments` are the
draw's contributions, whose means are the run's estimates: each exact mass divided by the history
mass and multiplied by the complete importance weight `weight_scientific`
(`noise_weight_scientific` is the noise factor alone). `sampled_records`, `sampled_parities`,
`decoder_calls` and `recovery_decoder_calls` describe the drawn history.
The `exact` section holds the unweighted exact scalars `history_mass`, `acceptance`, `error`,
`moments` and `target_success`, the noise likelihood ratio in hexadecimal,
the history `proposal_probability` as a numerator/denominator pair under `affine_support`
and, when it differs from one, the `target_normalization`.

## Format identifier

Every circuit file currently reports the format identifier `magna-0.1`.
