# Changelog

Notable changes to magna, newest first.
[RELEASING.md](RELEASING.md#version-numbers) explains what the version numbers mean.

## Unreleased

### Added

- The README explains how to cite magna.

### Changed

- Python decoder functions are called with less overhead, so sampling with a fast decoder is faster.

## 0.2.0 (2026-10-26)

### Added

- A contribution guide, a release guide, this changelog, a pull request template, and tests on Linux and macOS
  with Python 3.11 and 3.14, including a test of the installed wheel, for every pull request.
- The installation guide shows how to install a released version from its tag.

### Changed

- Python 3.11 or newer is required. Python 3.10 reached its end of life in October 2026.

### Fixed

- The C++ extension builds on Linux with GCC 15. Some kernel headers used standard-library features
  whose headers only the macOS standard library included indirectly.

## 0.1.0 (2026-10-15)

First public release: the `magna` Python package and command, the user manual,
and the example tasks of the accompanying paper.
