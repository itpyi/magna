# Python API

The file interface mirrors the command line and is the normal way to use the simulator.
Everything it needs, including its [errors](#errors), is imported from `magna`.
The [operation-level interface](#operation-level-interface), described at the end,
evaluates individual fully specified outcomes and is imported from `magna.operations`.

## File interface

```python
from magna import load_circuit, load_controllers, compile_circuit, run, enumerate_errors, compare_samplers

registry = load_controllers("controllers.json")
circuit = load_circuit("physical.magna", controllers=registry)
compiled = compile_circuit(circuit, task="task.json", options={"max_terms": 4096})
result = run(compiled, shots=1000, seed=7)
```

### Functions

| Function | Returns |
|---|---|
| `load_circuit(path, *, controllers=None, max_instructions=1000000, max_qubits=65536)` | A `CircuitFile` |
| `parse_circuit(text, *, controllers=None, source="<string>", max_instructions=1000000, max_qubits=65536)` | A `CircuitFile` from text; a file that exceeds either budget raises `ResourceLimit` |
| `load_controllers(path)` | A `ControllerRegistry` from a bindings JSON |
| `compile_circuit(circuit, *, task=None, options=None, controllers=None, ideal_state=None, recovery=None, sampling=None)` | A `CompiledCircuit` with all queries prepared |
| `run(circuit, *, shots=1, seed=0, task=None, options=None, controllers=None, ideal_state=None, recovery=None, sampling=None, progress_every=None, workers=1)` | A [result](results.md) dictionary; `circuit` may also be a `CompiledCircuit` |
| `enumerate_errors(circuit, *, max_weight, task=None, ideal_state=None, recovery=None, controllers=None, options=None, max_patterns=100000, max_branches=4096, max_record_bits=64, include_patterns=False)` | An [enumeration result](enumeration.md) |
| `compare_samplers(circuit, *, pilot_shots, seed=0, task=None, options=None, controllers=None, ideal_state=None, recovery=None, sampling=None)` | A pilot report with `gamma` and `recommended_sampler` |

Argument conventions:

- `circuit` is a path or a `CircuitFile`; `controllers` is needed only when passing a path and the circuit uses external functions.
- `run` also takes a `CompiledCircuit`, which already fixes the task, options, controllers, targets and sampling; leave those arguments out. Repeated runs reuse its preparation, and each result counts only the work of its own call.
- `task` is a dictionary or a JSON path. Paths inside a task JSON are relative to that file.
- `ideal_state` and `recovery` are paths, `CircuitFile` objects, or (for `ideal_state`) an object `{"circuit": ..., "outputs": [...], "postselection": {...}}`. Give each in the task or as an argument, not both. Paths passed as arguments are relative to the working directory.
- `sampling` is a dictionary or JSON path: `{"method": "physical"}` or `{"method": "importance", "proposal_scale": "10"}`.
- `options` is a dictionary of [resource options](options.md), including `history_sampler`.
- `workers` is a positive integer selecting native sampling threads (default 1). See
  [parallel sampling](sampling.md#parallel-sampling) for reproducibility and resource costs.

### CompiledCircuit

| Member | Meaning |
|---|---|
| `inspect()` | The inspection report: parsed circuit, task, route, bindings and the `compilation` resource report |
| `sample(rng)` | One draw with a caller-owned `random.Random`; returns the per-draw record described in [schemas](schemas.md#per-draw-sample) |
| `route` | `linear` or `nonlinear` |
| `task`, `options`, `ideal_state`, `recovery` | The resolved configuration |

Preparation happens in `compile_circuit`. It draws no randomness and calls no decoder.
A [resource option](options.md) that is too small for the task raises `ResourceLimit`; see [errors](#errors).
Repeated `run(compiled, ...)` and `sample` calls reuse the prepared decompositions.
A compiled circuit serves one thread at a time: a call from a second thread while another
is running raises `RuntimeError`. Use `run(compiled, workers=4, shots=10000)` to let
the runtime manage parallel sampling with independent worker state.

### CircuitFile and ControllerRegistry

`CircuitFile` exposes the parsed instructions, `num_qubits`, `num_records`, `analysis`
(control expressions and external instructions), `to_text()`, `write(path)` and `to_dict()`.
Invalid syntax raises `CircuitSyntaxError` with a `SourceLocation`.

`ControllerRegistry().register(name, callback, *, version, config=None)` binds a Python function
(`version` is required and keyword-only);
see [classical feedback](controllers.md#register-a-function-in-python).

### Errors

| Exception | Raised when |
|---|---|
| `CircuitSyntaxError` | The circuit text is invalid, or a decoder name has no binding; a `ValueError` carrying a `SourceLocation` |
| `ValueError` | A task, option, target or decoder binding is invalid |
| `ResourceLimit` | A [resource option](options.md) is too small for the task |
| `MemoryError` | The machine runs out of memory |
| `RuntimeError` | The simulator fails internally; please report it with the circuit and task |

`CircuitSyntaxError`, `SourceLocation` and `ResourceLimit` are imported from `magna`.
`ResourceLimit` is a `RuntimeError`. Its message names the option, and its attributes carry the same information:
`option` is the option name, `limit` its value, and `needed` a lower bound on what the task requires,
or `None` when the work stopped before that was known.

```python
from magna import ResourceLimit

try:
    compiled = compile_circuit(circuit, task="task.json", options={"max_terms": 16})
except ResourceLimit as exc:
    print(f"raise {exc.option} above {exc.limit}; the task needs at least {exc.needed}")
```

`run` reports an exceeded resource option or lack of memory, during preparation or sampling,
as `status: "failed"` instead of raising; `enumerate_errors` does the same once enumeration starts.
Both record the exception name and message in `failures`.

## Operation-level interface

For a circuit described as Python dictionaries and a fully specified outcome,
`Simulator` evaluates exact probabilities and compiled forms directly.
These classes and the exact value types are imported from `magna.operations`.

```python
from magna.operations import InstrumentIR, SampleAssignment, Simulator, StabilizerKet

ir = InstrumentIR(
    inputs=(0,), outputs=(0,), initial=StabilizerKet.positive(1),
    operations=(
        {"id": "t", "kind": "T", "qubits": [0]},
        {"id": "m", "kind": "MX", "qubits": [0], "record": "m"},
    ),
)
probability = Simulator(ir).probability(SampleAssignment(true={"m": 0}))
```

| Class | Role |
|---|---|
| `InstrumentIR` | Fixed preparation, ordered operations, output wires and declared controls |
| `SampleAssignment` | True record values, original faults and report flips for one outcome |
| `Simulator(ir, options=None)` | `probability(assignment)`, `compile(assignment)`, `evaluate_observable(...)`, `sample_one(...)` |
| `BoundaryTask`, `BoundarySampler` | Event queries and causal sampling on an `InstrumentIR` |
| `NoiseModel` | Original-site noise draws with exact likelihood ratios |
| `ExactScalar`, `StabilizerKet`, `StabilizerExpansion` | Exact value types |

The [operation-level reference](architecture/operations.md) and
[exact value reference](architecture/values.md) describe these classes in detail.
