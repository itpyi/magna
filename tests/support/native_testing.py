"""Build and import the test-only ``_native_testing`` extension.

The module compiles the production bindings (``core/magna/native/bindings/module.hpp``)
together with the independent oracles in ``tests/native/python/``, using the production compiler
flags, into ``build/native-testing/<source digest>/``.
The installed package never contains it; it shares ``ResourceLimit`` with ``magna._native``.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'core' / 'magna' / 'native'
ORACLES = ROOT / 'tests' / 'native' / 'python'
FLAGS = ['-std=c++17', '-O3', '-DNDEBUG', '-Wall', '-Wextra', '-Wno-missing-field-initializers',
         '-Wno-unused-function', '-fPIC', '-shared', '-pthread']
NAME = '_native_testing'


def _units():
    production = sorted(p for p in NATIVE.rglob('*.cpp') if p.parent.name != 'bindings')
    return [ORACLES / 'testing_module.cpp', *production]


def _digest(units):
    sources = sorted({*units, *NATIVE.rglob('*.hpp'), *ORACLES.glob('*.hpp')})
    digest = hashlib.sha256(' '.join([*FLAGS, sys.version, sysconfig.get_config_var('EXT_SUFFIX')]).encode())
    for path in sources:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load():
    """Return the oracle module, compiling it once per source digest."""
    if NAME in sys.modules:
        return sys.modules[NAME]
    import magna._native  # noqa: F401  The oracles reuse its ResourceLimit class.
    units = _units()
    folder = ROOT / 'build' / 'native-testing' / _digest(units)
    output = folder / (NAME + sysconfig.get_config_var('EXT_SUFFIX'))
    if not output.exists():
        folder.mkdir(parents=True, exist_ok=True)
        flags = FLAGS + (['-undefined', 'dynamic_lookup'] if sys.platform == 'darwin' else [])
        partial = folder / (NAME + '.partial')
        command = [os.environ.get('CXX', 'clang++'), *flags, '-I' + sysconfig.get_path('include'),
                   '-I' + str(NATIVE), *map(str, units), '-o', str(partial)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError('building the test-only native module failed:\n' + result.stderr)
        partial.replace(output)
    spec = importlib.util.spec_from_file_location(NAME, output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[NAME] = module
    return module
