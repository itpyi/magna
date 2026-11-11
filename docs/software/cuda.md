# CUDA sampler

The experimental CUDA sampler runs physical and importance sampling of supported tasks on an NVIDIA GPU.
It reports the same acceptance, error and conditional error as `run`, with every per-draw
probability still exact. Speed depends on the task and the time spent in its decoder.
Use it when a task needs more draws than the CPU sampler completes in the time you have.

On an RTX 4090 D, 262,144 draws of four example tasks without decoder calls took the following times,
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

`CudaSampler.run(shots, *, seed=0, offset=0, batch_size=None)` attempts `shots` draws.
Draws are numbered from `offset`, so `run(n, seed=7, offset=n)` continues a run of `n` draws with the next `n`.

The GPU processes the draws in batches of at most `batch_size` (up to 1,000,000);
`None` chooses 4096 for tasks with decoder calls and 65536 otherwise.
Host and GPU memory grow approximately with batch size; choose a smaller batch when memory is limited.

The first call also compiles the GPU code, so reuse one sampler for repeated runs.
After changing the circuit, task, sampling or options, compile again and create a new sampler.
Use a sampler from one thread at a time.

Decoders are passed to `compile_circuit` through `controllers`, as for `run`, and run on the CPU.
Tasks with decoder calls use the default [history sampler](sampling.md#history-samplers), `affine_support`;
for `exact_marginal`, `CudaSampler` raises `ValueError`.

## Supported tasks

The CUDA sampler supports a task when:

- its noise is unconditional and `sample_records` is empty;
- it has no observables;
- it uses physical sampling, or importance sampling in which every scaled noise location
  has the same total error probability and there are between 2 and 512 of them.
  A noise location is one target qubit or qubit pair of a noise instruction, one correlated-error chain,
  or one noisy measurement;
- if the task has no decoder calls and the ideal state uses postselection, its postselection probability is a power of two;
- if the task has decoder calls, it uses the `affine_support` history sampler.

Affine feedback, correlated errors, Pauli channels and measurement report errors are all supported.
For an unsupported task, `CudaSampler(compiled)` raises `ValueError` with the reason.

The example tasks run as follows:

| Examples | Physical sampling | Importance sampling |
|---|---|---|
| `bh2`–`bh100`, `-S` and `-T` | Yes | Yes |
| `color3`, `color5`, `rp2-d3`, `rp2-d3-S`, `rp2-d5`, `rp2-d5-S`, `wg-S`, `wg-T` | Yes | No: more than 512 noise locations |
| `dk`, `logical-distillation` | Yes | Yes |
| `tqd3`, `tqd4`, `tqd5` | Yes | No: more than 512 noise locations |

## Results

`run` returns a dictionary:

| Field | Meaning |
|---|---|
| `status` | `complete` |
| `requested`, `completed` | Number of draws |
| `seed`, `offset` | The arguments of `run` |
| `batch_size` | The batch size used |
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

`MultiGpuCudaSampler` divides a run among several GPUs on one machine.
It runs the tasks that `CudaSampler` supports, except tasks with decoder calls.
Use it when a run takes too long on one GPU.

Write the code in a script, under `if __name__ == "__main__":`.
Each GPU gets its own Python process, and that process imports your script when it starts.

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

`devices` lists the GPUs to use, numbered as CUDA sees them:
with `CUDA_VISIBLE_DEVICES=2,3`, use `devices=(0, 1)`.

`run` takes the same arguments as `CudaSampler.run`, except that `batch_size` defaults to 65536 and cannot be `None`.
`shots` is the total number of draws, divided evenly among the GPUs, and `batch_size` applies to each GPU.
The first run takes longer, because each process compiles the task and the GPU code.
Later runs reuse the processes. Leaving the `with` block, or calling `close()`, stops them.

The same seed and offset give the same faults on any number of GPUs.
As with batches, the estimates can still differ in the last digits.

The result has the fields of `CudaSampler.run`, plus these:

| Field | Meaning |
|---|---|
| `devices` | The GPUs given to the sampler |
| `shards` | One entry for each GPU used, with its `device`, `offset`, `shots` and `sampling_seconds` |
| `timings.total_seconds` | Time for the whole run |
| `timings.startup_seconds` | Time spent starting the processes |

`timings.sampling_seconds` covers all GPUs together. The GPUs run at the same time,
so the times in `shards` do not add up to it.

To limit how long a run can take, pass `timeout` in seconds:
`MultiGpuCudaSampler(compiled, devices=(0, 1), timeout=600)`. The limit includes the first run's compilation.
If a run times out or a GPU fails, `run` raises an error and the sampler closes; create a new one to try again.
