# magna: Magic-state-factory Assessment via Global Network Algebra

*Copia magiarum quam simulare possumus **magna** est!*

magna evaluates noisy quantum protocols with non-Clifford gates, measurements,
Pauli noise and classical feedback, and reports exact or statistically estimated acceptance
probabilities, logical error rates and target-state fidelities.
It compiles the whole fixed-input circuit into one amplitude and contracts it globally,
so the cost is set by the requested output rather than by the number of qubits alone,
and every per-draw quantity is exact.

Protocols are Stim-style circuit files with classical control registers, external decoder calls
and in-circuit postselection on detectors. A task selects the output qubits and the
error definition, usually an ideal target state written as a small noiseless circuit.
Three estimators share one task: physical Monte Carlo sampling, importance sampling with exact
reweighting, and exact low-weight fault enumeration.

## Install and run

```sh
python -m pip install .
magna run tests/fixtures/basic/t_interference.magna --task tests/fixtures/basic/t_interference.task.json
python scripts/run_task.py examples/bh2-T --shots 1024 --output bh2.json
```

The first command reports the exact acceptance `(2 + sqrt(2)) / 4` of a one-qubit `H T H` circuit.
The second runs 14-to-2 Bravyi–Haah magic-state distillation and reports the joint
infidelity of the two accepted output states against two ideal T states.
Every directory in `examples/` is a complete task (circuit, target, configuration, decoder code)
from the accompanying paper that `scripts/run_task.py` runs through the public interface;
copy a directory and the runner next to an installed simulator to run it anywhere.
The TQD tasks call their authors' decoder, which is not distributed here;
`python scripts/fetch_author_decoder.py` downloads it at the pinned commit and checks its hashes.

## Documentation

[User manual](https://itpyi.github.io/magna/)

- [Overview](docs/software/index.md), [installation](docs/software/install.md) and [tutorial](docs/software/getting-started.md)
- [Concepts](docs/software/concepts.md): what is exact, what is sampled, and how results are defined
- User guide: [circuit files](docs/software/circuit-format.md), [noise](docs/software/noise.md),
  [classical feedback and decoders](docs/software/controllers.md), [tasks and targets](docs/software/tasks.md),
  [results](docs/software/results.md), [sampling methods](docs/software/sampling.md), [exact enumeration](docs/software/enumeration.md)
- Reference: [command line](docs/software/cli.md), [Python API](docs/software/api.md),
  [resource options](docs/software/options.md), [schemas](docs/software/schemas.md)
- [Examples](docs/software/examples.md): RP² cultivation, magic-state distillation, TQD memory, and the full task list
- [How it works](docs/software/architecture/index.md): the pipeline and the repository map

## Develop

```sh
./scripts/python.sh scripts/build_native_core.py
./scripts/python.sh scripts/project_checks.py --suite current
./scripts/python.sh scripts/project_docs.py check
```

The package source is in `core/magna` (Python interface plus the C++ kernel in `native/`),
tests in `tests/core`, tutorial fixtures in `tests/fixtures`.
[CONTRIBUTING.md](CONTRIBUTING.md) explains how to propose a change,
and [CHANGELOG.md](CHANGELOG.md) lists the changes in each release.

## Citing magna

If you use magna in your research, please cite the paper that introduces it:

Yifei Wang and Yingfei Gu, *Magna: simulating magic state factories as cheaply as Clifford circuits*,
[arXiv:2610.22315](https://arxiv.org/abs/2610.22315) (2026).

```bibtex
@misc{wang2026magna,
  title         = {{Magna}: simulating magic state factories as cheaply as {Clifford} circuits},
  author        = {Wang, Yifei and Gu, Yingfei},
  year          = {2026},
  eprint        = {2610.22315},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2610.22315},
}
```

To make your results reproducible, also state the magna version you used (`magna --version`).

## License

magna is distributed under the [Mozilla Public License 2.0](LICENSE).
Some example circuits and decoder code come from other projects under the MIT and Apache 2.0 licenses;
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists them with their licenses.
