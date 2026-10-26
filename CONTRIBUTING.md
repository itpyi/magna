# Contributing to magna

Bug reports, documentation fixes, new examples and kernel improvements are welcome.
This page explains how to build and test magna, what a pull request should contain,
and what happens before it is merged.

## Issues

Report a bug with the circuit file, task, command and output that show it, cut down as far as you can,
together with the magna version (`pip show magna-quantum`, or the commit you built),
your Python version and your operating system.
Questions and feature ideas are welcome as issues too.
For a larger change, especially a new kernel feature, you may open an issue first to discuss the design.
This is optional, but it can save rework.

## Build and test

You need Python 3.11 or later with development headers, and a C++17 compiler.
Work in a virtual environment in `.venv`, which `scripts/python.sh` uses automatically:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./scripts/python.sh scripts/build_native_core.py
./scripts/python.sh scripts/project_checks.py --suite current
./scripts/python.sh scripts/project_docs.py check
```

Rebuild after every change under `core/magna/native/`.
The build and the tests compile with `clang++`. Without it, build with `--cxx g++`
and run the tests with the environment variable `CXX=g++`.
The test suite also compiles the native headers on their own and runs some of them under
the undefined-behaviour sanitizer. Install [Stim](https://github.com/quantumlib/Stim)
(`.venv/bin/python -m pip install stim`) to include the tests that compare gates and records with it.

The installed-wheel tests are skipped unless you point them at an installation.
This recipe starts from fresh folders, so a second run tests the new build rather than the old one:

```sh
rm -rf /tmp/magna-wheel /tmp/magna-dist
python3 -m venv /tmp/magna-wheel
/tmp/magna-wheel/bin/python -m pip wheel --no-deps -w /tmp/magna-dist .
/tmp/magna-wheel/bin/python -m pip install /tmp/magna-dist/*.whl ./tests/fixtures/basic/decoder_plugin
MAGNA_INSTALLED_PYTHON=/tmp/magna-wheel/bin/python ./scripts/python.sh -m unittest discover -s tests/core -p test_installed_interface.py
```

[How it works](docs/software/architecture/index.md) describes the pipeline
and what each directory under `core/magna/native/` is responsible for.

## Design rules

magna is a general simulator: protocols enter as circuit files and decoders as registered functions.
Changes to the package keep to these rules.

- Algorithms belong in the C++ kernel. Python adapts inputs and results and connects your decoder functions.
- The kernel contains no protocol-specific code: no circuit generators, stored answers or experiment-tuned settings.
- Decoders and other classical controllers see only reported measurement data, their static configuration
  and their own earlier decisions. A correction that fails is evaluated as the physical output it produces;
  it never stops the simulation or adds postselection.
- Existing decoder functions keep working without changes.
- The public interface (Python API, command line, result fields, circuit and task files) stays stable.
  A change to it updates the user manual in `docs/software/` in the same pull request.

## Pull requests

Keep one pull request to one change: a feature, an optimization and a refactor are three pull requests.
The pull request template asks for the following.

- **Purpose.** What changes and why.
- **Scope.** The parts of the repository the change is meant to touch, and a reason for each changed file outside them.
  Reviewers check every changed file against this.
- **Interface, format and results.** Whether the public interface, the file formats or any numerical result changes.
  An optimization or refactor should leave the results of a seeded run unchanged; say so if they change.
- **Tests.** New behaviour comes with tests; a bug fix comes with a test that fails without the fix.
- **Changelog.** A change that users would notice adds a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
- **Measurements.** A performance or accuracy number names the commit it was measured on
  (normally the pull request's final commit), the machine and the command that reproduces it.
- **AI tools (optional).** You may say which AI tools helped. Either way you are responsible for every line
  and should be able to answer review questions about it.

## Review

1. **CI.** Every push builds and tests on Linux and macOS with Python 3.11 and 3.14,
   and tests an installed wheel. A first-time contributor's CI starts after a maintainer approves it.
2. **Scope and code.** A maintainer checks that each changed file serves the stated purpose, then reviews the code.
3. **Benchmarks.** For a change to the C++ kernel or to sampling, the maintainers run a fixed benchmark suite,
   kept outside this repository, on a fixed machine, comparing `main` with the pull request.
   Every case that becomes 5% or more slower, or needs 5% or more additional peak memory,
   is reported in the pull request, and the maintainers decide whether the change is worth it.
   Benchmarks you run yourself are welcome as additional information.
4. **Merge.** Pull requests are squash-merged, so each becomes one commit on `main`.
   [RELEASING.md](RELEASING.md) explains how merged changes become a release.

## License

magna is distributed under the Mozilla Public License 2.0 (see [LICENSE](LICENSE)).
Contributions are accepted under the same license.
