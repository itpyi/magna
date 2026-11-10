<!-- Thank you for contributing. CONTRIBUTING.md explains each section. -->

## Purpose

<!-- What does this pull request change, and why?
Link the agreed issue by writing "Closes #" and its number. A change to the kernel, sampling, the public interface
or a file format, and a new feature, need one; CONTRIBUTING.md explains which changes go straight to a pull request. -->

## Scope

Parts of the repository this change is meant to touch:

- [ ] Kernel foundation `core/magna/native/foundation/`
- [ ] Kernel compiler `core/magna/native/compile/`
- [ ] Kernel evaluation `core/magna/native/evaluate/`
- [ ] Kernel runtime `core/magna/native/runtime/`
- [ ] CUDA sampler `core/magna/native/cuda/`
- [ ] Bindings `core/magna/native/bindings/`
- [ ] Python interface `core/magna/*.py`, `core/magna/data/`
- [ ] User manual `docs/`, `mkdocs.yml`
- [ ] Project documents `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `RELEASING.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`
- [ ] Tests `tests/`
- [ ] Examples `examples/`
- [ ] Scripts, packaging and CI `scripts/`, `setup.py`, `pyproject.toml`, `MANIFEST.in`, `requirements.txt`, `.gitignore`, `.github/`

Changed files outside these parts, in backticks with a reason for each (or "none"):

<!-- Example: `core/magna/native/evaluate/trace_query.hpp`: the new sampler reuses a suffix of the boundary rows. -->

## Interface, format and results

<!-- Keep one answer per line and explain any change. -->

- Public interface (Python API, command line, result fields): unchanged / changed
- Circuit or task file format: unchanged / changed
- Results of a seeded run: unchanged / changed
- `CHANGELOG.md`: updated / not needed

## Tests

<!-- Tests added or changed, and the commands you ran. -->

## Measurements

<!-- Only if you report performance or accuracy numbers: commit, machine and command for each. -->

## AI tools (optional)

<!-- Which AI tools helped, if any. -->
