# CUDA sampler

The experimental CUDA sampler runs physical and importance sampling of supported tasks on an NVIDIA GPU.
It reports the same acceptance, error and conditional error as `run`, with every per-draw
probability still exact, and draws hundreds of times faster than the CPU sampler.
Use it when a task needs more draws than the CPU sampler completes in the time you have.

On an RTX 4090 D, 262,144 draws of four example tasks took the following times,
compared with `run` on one core of a Ryzen 9 9950X (median of three runs, batch size 65,536,
first-call compilation excluded):

| Task | Sampling | `run` (s) | CUDA sampler (ms) | Speed-up |
|---|---|---:|---:|---:|
| `bh2-T` | importance, scale 1000/7 | 1.20 | 1.80 | 665x |
| `bh100-T` | importance, scale 500/77 | 9.82 | 38.6 | 254x |
| `color3` | physical | 0.99 | 1.47 | 671x |
| `rp2-d5` | physical | 1.46 | 6.25 | 233x |

## Install

You need an NVIDIA GPU with a driver that supports CUDA 12. Install magna with the `cuda12` extra,
which adds NumPy and CuPy:

```sh
python -m pip install ".[cuda12]"
```

CuPy also needs the CUDA runtime libraries. If no CUDA 12 toolkit is installed,
`python -m pip install "cupy-cuda12x[ctk]>=14,<15"` installs them as Python packages;
[CuPy's installation guide](https://docs.cupy.dev/en/stable/install.html) lists the other options.
Building magna itself needs no CUDA compiler: the sampler compiles its GPU code the first time it runs.

## Run a task

Compile the task as usual, then pass it to `CudaSampler`:

```python
from magna import compile_circuit
from magna.cuda_sampler import CudaSampler

compiled = compile_circuit(
    "examples/bh2-T/physical.magna",
    task="examples/bh2-T/task.json",
    sampling={"method": "importance", "proposal_scale": "1000/7"},
)
sampler = CudaSampler(compiled)
result = sampler.run(1_048_576, seed=7)
print(result["estimates"], result["standard_errors"])
```

`CudaSampler.run(shots, *, seed=0, offset=0, batch_size=65536)` attempts `shots` draws.
Draws are numbered from `offset`, so `run(n, seed=7, offset=n)` continues a run of `n` draws with the next `n`.

The GPU processes the draws in batches of at most `batch_size` (up to 1,000,000);
larger batches use more GPU memory.

The first call also compiles the GPU code, so reuse one sampler for repeated runs.
After changing the circuit, task, sampling or options, compile again and create a new sampler.
Use a sampler from one thread at a time.

## Supported tasks

The CUDA sampler supports a task when:

- it samples no measurement history: `compilation.sampled_history` in the
  [inspect report](sampling.md#history-samplers) is false. Decoder calls,
  [conditional noise](noise.md#conditional-noise) and `sample_records` all sample a history;
- it has no observables;
- it uses physical sampling, or importance sampling in which every scaled noise location
  has the same total error probability and there are between 2 and 512 of them.
  A noise location is one target qubit or qubit pair of a noise instruction, one correlated-error chain,
  or one noisy measurement;
- if the ideal state uses postselection, its postselection probability is a power of two.

Affine feedback, correlated errors, Pauli channels and measurement report errors are all supported.
For an unsupported task, `CudaSampler(compiled)` raises `ValueError` with the reason.

The example tasks run as follows:

| Examples | Physical sampling | Importance sampling |
|---|---|---|
| `bh2`–`bh100`, `-S` and `-T` | Yes | Yes |
| `color3`, `color5`, `rp2-d3`, `rp2-d3-S`, `rp2-d5`, `rp2-d5-S`, `wg-S`, `wg-T` | Yes | No: more than 512 noise locations |
| `dk`, `logical-distillation`, `tqd3`, `tqd4`, `tqd5` | No: decoder calls | No |

## Results

`run` returns a dictionary:

| Field | Meaning |
|---|---|
| `status` | `complete` |
| `requested`, `completed` | Number of draws |
| `seed`, `offset`, `batch_size` | The arguments of `run` |
| `estimates` | `acceptance`, `error` and `conditional_error` |
| `standard_errors` | Standard errors of the same three estimates |
| `timings.sampling_seconds` | Time spent sampling |

The estimates and standard errors mean the same as in the [result of `run`](results.md#read-the-estimates),
including the weighting of importance-sampled draws.
`error` and `conditional_error` are `None` when the task has no ideal state.
The result has no scientific fields, so a value outside the range of ordinary doubles raises `ValueError`
instead of becoming `null`.

## Random numbers

The CUDA sampler uses its own random number generator, Philox4x32-10, indexed by the draw number.
The same seed and offset give the same faults whatever the batch size;
estimates can still differ in the last digits, because batches are summed in a different order.
CPU and GPU runs with the same seed draw different faults.

## Multiple GPUs on one host

Use `MultiGpuCudaSampler` for long sampling jobs on explicitly selected GPUs.
It supports the same tasks and sampling methods as `CudaSampler`. Save this as a Python script:

```python
from magna import compile_circuit
from magna.cuda_multi import MultiGpuCudaSampler

if __name__ == "__main__":
    compiled = compile_circuit(
        "examples/bh2-T/physical.magna", task="examples/bh2-T/task.json",
        sampling={"method": "importance", "proposal_scale": "1000/7"},
    )
    with MultiGpuCudaSampler(compiled, devices=(0, 1, 2, 3)) as sampler:
        result = sampler.run(16_777_216, seed=7)
        print(result["estimates"], result["standard_errors"])
```

Select GPUs allocated to your job; IDs are logical IDs after `CUDA_VISIBLE_DEVICES` filtering.
The main guard is required because workers start with Python's `spawn` method.
Repeated calls reuse the workers. Use the context manager or call `close()` when finished.

`run(shots, *, seed=0, offset=0, batch_size=65536)` takes a total shot count across all devices.
The batch size is per device, with the same 1,000,000 limit as the single-GPU sampler.
Results add `devices`, `shards` (draw intervals and elapsed times), and
`timings.total_seconds`, `timings.startup_seconds`, and `timings.sampling_seconds`.
Shard times overlap. Changing device count preserves fault draws, but the last digits
of estimates and standard errors can differ because reduction order changes.

The optional constructor argument `timeout` limits each run, including worker startup.
A worker failure or timeout closes the pool and raises; create a new sampler before retrying.
Worker cleanup may add time beyond the timeout.
