# Installation

## Requirements

- Python 3.11 or newer, with development headers
- A C++17 compiler (the platform compiler is used; set `CXX` to choose another)

The package contains a compiled C++ extension, so installation builds native code.

## Install from a checkout

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

This installs the `magna` Python package and the `magna` command.

Some packaged tasks need extra packages for their decoders (SciPy, PyMatching, Stim).
Each task directory lists them in its `requirements.txt`; install those when you run that task.

## Install a released version

Every release has a tag, listed on the [releases page](https://github.com/itpyi/magna/releases).
To install one without cloning the repository, name its tag, here `vX.Y.Z`:

```sh
python -m pip install "magna-quantum @ git+https://github.com/itpyi/magna.git@vX.Y.Z"
```

This installs the package and the `magna` command, not the examples.

## Check the installation

```sh
magna run tests/fixtures/basic/t_interference.magna --task tests/fixtures/basic/t_interference.task.json
```

The command prints a short summary whose acceptance is about 0.853553.
Add `--json` to see the full JSON result.

## Working inside the repository

After `pip install`, run the repository's scripts with plain `python`, for example
`python scripts/run_task.py examples/bh2-T`.
Developers who run from the source tree without installing use the launcher
`./scripts/python.sh`, which puts the checkout on the path; build the native extension
in place first with `./scripts/python.sh scripts/build_native_core.py`.
The CLI is then `./scripts/python.sh -m magna.file_api`.
Run the test suite with `./scripts/python.sh scripts/project_checks.py --suite current`
and check the manual's links with `./scripts/python.sh scripts/project_docs.py check`.

## Read this manual locally

```sh
./scripts/serve_docs.py
```

The script creates an isolated environment in `build/docs-venv`, installs MkDocs on the first run,
and opens the manual in your browser at `http://localhost:8000/` or the next free port.
Edits reload automatically; press Ctrl-C to stop.
Use `--port 8080` to pick a port or `--no-browser` to skip opening the browser.

To build static HTML after the first run:

```sh
build/docs-venv/bin/python -m mkdocs build --strict
```

The site is written to `build/software-docs`.
