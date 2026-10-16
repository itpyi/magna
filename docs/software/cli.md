# Command line

```text
magna COMMAND CIRCUIT [options]
```

`magna --help` lists the commands, `magna COMMAND --help` lists the options of one command,
and `magna --version` prints the version.

## Commands

| Command | Action | Result |
|---|---|---|
| `validate` | Parse the circuit and check the task, bindings and causal references | Circuit summary with its format identifier |
| `inspect` | Compile the task and report the route, dependencies and prepared resources | Inspection report |
| `run` | Sample the task | [Result](results.md) |
| `enumerate` | Sum fault patterns up to `--max-weight` exactly | [Enumeration result](enumeration.md) |
| `compare-samplers` | Run a pilot of both history samplers | Pilot report with Gamma and a recommendation |

Each command prints a short human-readable summary, for example

```text
run: complete, 2000 of 2000 draws, seed 7
  acceptance         1 ± 0
  error              0.001 ± 0.00070693
  conditional error  0.001 ± 0.00070693
  route linear
  time 0.0155 s (preparation 0.000181 s, sampling 0.0135 s)
```

`--json` prints the full JSON result instead, and `--output FILE` also writes it to a file.
Scripts that read the result should use one of these two options.
Progress reports go to standard error.

## Options

Every command accepts:

| Option | Meaning |
|---|---|
| `--task FILE` | Task JSON; see [tasks](tasks.md) |
| `--controllers FILE` | Decoder bindings JSON; see [classical feedback](controllers.md#register-functions-from-a-json-file) |
| `--ideal-state FILE` | Ideal target circuit, if not given in the task |
| `--recovery FILE` | Terminal recovery circuit, if not given in the task |
| `--options FILE` | Resource budgets and history sampler; see [resource options](options.md) |
| `--output FILE` | Also write the full JSON result to a file |
| `--json` | Print the full JSON result instead of the summary |

Options of individual commands:

| Option | Commands | Meaning |
|---|---|---|
| `--sampling FILE` | validate, inspect, run, compare-samplers | Noise proposal, `{"method": "importance", "proposal_scale": "10"}` |
| `--history-sampler NAME` | validate, inspect, run | `affine_support` (default) or `exact_marginal`; overrides the options file |
| `--shots N` | run | Number of draws (default 1) |
| `--workers N` | run | Native sampling threads (positive integer, default 1) |
| `--seed N` | run, compare-samplers | Random seed (default 0) |
| `--progress-every N` | run | Print progress and kappa every N completed draws |
| `--pilot-shots N` | compare-samplers | Draws per history sampler (at least 2); required |
| `--max-weight K` | enumerate | Maximum number of faults per pattern; required |
| `--max-patterns N` | enumerate | Pattern budget (default 100000) |
| `--max-branches N` | enumerate | Decoder-history budget (default 4096) |
| `--include-patterns` | enumerate | List every pattern in the JSON result |

An option given to a command that does not use it is an error.
Paths given on the command line are relative to the working directory.
Paths inside a task JSON are relative to that JSON file.
Give the ideal state and the recovery in one place only.

## Examples

```sh
magna validate factory.magna
magna inspect factory.magna --task task.json
magna run factory.magna --task task.json --shots 1000 --seed 7 --output result.json
magna run factory.magna --task task.json --shots 10000 --workers 4 --output result.json
magna run physical.magna --ideal-state ideal.magna --recovery recovery.magna --controllers controllers.json --task task.json --shots 1000
magna run physical.magna --task task.json --sampling sampling.json --shots 2000 --seed 7 --json
magna enumerate physical.magna --task task.json --max-weight 2
magna compare-samplers physical.magna --task task.json --pilot-shots 256 --seed 7
```

## Exit status

| Status | Meaning |
|---|---|
| 0 | The command completed; with `--json`, read `status` inside the JSON |
| 1 | The simulation ran but failed: an exceeded resource option or time limit, a decoder failure, lack of memory or an internal error; with `--json`, `status` is `"failed"` |
| 2 | Invalid input: syntax error, missing binding, unknown task field, unknown option or bad configuration |

For invalid input the message goes to standard error, naming the offending key or value.
With `--json`, standard output instead receives `{"status": "invalid", "error": ...}`.
A decoder named in `--controllers` whose module or function cannot be imported is invalid input too.

## Running from a source checkout

Without installing, build the native extension and use the checkout launcher from the repository root:

```sh
./scripts/python.sh scripts/build_native_core.py
./scripts/python.sh -m magna.file_api run physical.magna --task task.json
```
