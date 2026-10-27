# Resource options and failures

Compilation prepares a stabilizer decomposition for each [reference network](concepts.md#one-circuit-one-amplitude)
the task needs. Options bound that work and the run time. Pass them as `options={...}` in Python
or as `--options options.json` on the command line.

```json
{"max_path_bits": 65536, "max_doubled_path_bits": 131072, "max_quotient_rank": 12, "max_terms": 4096, "history_sampler": "affine_support"}
```

These are the `options` of every packaged task's `run.json` (see `examples/bh2-T/run.json`),
and they are a reasonable starting point for circuits of similar size.
Add `"max_seconds": 600`, for example, to bound the wall-clock time of a run.

## Options

| Option | Scope | Meaning |
|---|---|---|
| `max_path_bits` | compilation | Path variables of one compiled network |
| `max_doubled_path_bits` | compilation | Path variables of a network that pairs the circuit with its conjugate, as probabilities and fidelities do |
| `max_quotient_rank` | compilation | Phase coordinates left after elimination |
| `max_terms` | compilation | Stabilizer terms of one prepared decomposition |
| `max_table_bytes` | compilation | Memory of prepared lookup tables |
| `max_plans` | runtime | Number of compiled networks kept for reuse |
| `max_queries` | runtime | Number of network evaluations in a run |
| `max_seconds` | runtime | Wall-clock time, checked between evaluations |
| `history_sampler` | runtime | `affine_support` (default) or `exact_marginal`; see [sampling methods](sampling.md#history-samplers) |

Integer limits must be nonnegative (`max_plans` and `max_queries` at least one);
`max_seconds` is a positive number. Unknown option names are rejected.
A running query is not interrupted, so a run can overshoot `max_seconds` until its
active queries finish (up to one per worker in a parallel run).
`max_queries` and `max_seconds` count from the start of each `run` call; when `run` also compiles,
the count includes that preparation.
With [parallel sampling](sampling.md#parallel-sampling), `max_queries` and `max_seconds`
apply to the whole run. Other resource limits apply separately to each worker;
they do not cap total memory use.

## Reading the resource report

`magna inspect` returns `compilation` with:

| Field | Meaning |
|---|---|
| `status` | `complete` after preparation |
| `query_families` | Prepared reference networks grouped by stage: `protocol` (named `recovery` when the task has a terminal recovery) and `target`. Each entry names its `purpose` and reports its phase coordinates, stabilizer terms, decomposition bounds, timing and which resource guarantees were established within the limits; see [the inspect report](architecture/compiler.md#reading-the-inspect-report) |
| `resource_summary` | The maximum and the sum of the prepared decomposition bounds |
| `compiler_diagnostics` | Per stage: how the non-Clifford gates were arranged and the size of the compiled circuit; see [noncommuting rotation axes](architecture/compiler.md#noncommuting-rotation-axes) |
| `sampled_history`, `conditional_noise` | Whether measurement bits are sampled and whether guarded channels are present |
| `reference_scope`, `guarded_noise_coverage` | Short fixed descriptions of what the prepared networks cover |
| `preparation_seconds` | Time spent preparing |

Resources are prepared once per task. Faults, feedback decisions and importance weights
are applied to the prepared decomposition during sampling without re-preparing it,
so the resource report does not change from draw to draw.

## What happens on failure

| Situation | Behaviour |
|---|---|
| Budget exceeded during preparation | `compile_circuit` raises [`ResourceLimit`](api.md#errors), naming the option, its value and the amount needed; `run` returns `status: "failed"` with a `preparation` failure entry, `completed` 0, `failed` 0 and `not_attempted` equal to the shot count |
| Budget or time exceeded during sampling | The run stops; completed draws are kept as `partial_statistics`, the main `estimates` are `null`, and `failures` records the shot and reason |
| Decoder failure during a draw (wrong number of outputs, a non-bit value, an exception) | The run stops with `status: "failed"` and keeps the completed draws as `partial_statistics`; the CLI exits with status 1 |
| Invalid circuit, task or configuration, including a decoder name without a binding or a binding that cannot be imported | An exception before any work (`CircuitSyntaxError` or `ValueError`); the CLI exits with status 2 |

The simulator never falls back to an approximation when a limit is hit.
Raise the limit, simplify the target preparation, or reduce the number of output qubits.

`validate` checks syntax and configuration without preparing decompositions;
`inspect` does the full preparation and is the way to measure a task's resource need.
