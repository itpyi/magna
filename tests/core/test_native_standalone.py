"""Compile native owners without Python and compare with independent finite sums."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

class NativeStandaloneTest(unittest.TestCase):
    def test_reference_runtime_headers_are_self_contained(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler, 'C++17 compiler required for native-only acceptance')
        for header in ('evaluate/reference_network.hpp', 'runtime/affine_proposal.hpp',
                       'runtime/file_execution.hpp', 'runtime/file_enumeration.hpp',
                       'runtime/file_parallel.hpp'):
            with self.subTest(header=header):
                result = subprocess.run([compiler, '-std=c++17', '-fsyntax-only', '-x', 'c++',
                    '-I'+str(ROOT/'core/magna/native'), '-'],
                    input=f'#include "{header}"\n', capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_foundation_without_cpython(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler, 'C++17 compiler required for native-only acceptance')
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'foundation'
            build = subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=undefined',
                '-fno-sanitize-recover=all', '-I'+str(ROOT/'core/magna/native'),
                str(ROOT/'tests/native/foundation.cpp'),
                *(str(p) for p in sorted((ROOT/'core/magna/native').rglob('*.cpp'))
                    if 'bindings' not in p.name), '-o', str(executable)],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable)], capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('checks passed', result.stdout)
