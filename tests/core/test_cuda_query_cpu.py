"""CPU-only exact query controls and deterministic compiler differential checks."""
import builtins
from pathlib import Path
from random import Random
import subprocess
import sys
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy is required')

from magna import _native, compile_circuit, parse_circuit
from magna._cuda_experiment import CudaExperiment
from magna._cuda_queries import GpuQueries
from magna.cuda_sampler import CudaSampler
from magna.scalar import ExactScalar
from tests.support.cuda_reference import cpu_evaluate, cpu_frames, draw, evaluate_frames, tuples


class CpuQueryControlTests(unittest.TestCase):
    def test_native_cuda_bridge_is_built_even_without_cupy(self):
        # Missing registration is a build failure, not an optional-device skip.
        for name in ('make_cuda_experiment', 'make_cuda_queries',
                     '_cuda_query_cpu_evaluate', 'cuda_noise_metadata', 'cuda_noise_validate',
                     '_cuda_noise_cpu_sample', '_cuda_draw_noise', '_cuda_map_frames',
                     '_cuda_evaluate_frames', '_cuda_evaluate_noise'):
            with self.subTest(binding=name):
                self.assertTrue(callable(getattr(_native, name, None)), name)

    def test_default_cpu_sampling_needs_neither_numpy_nor_cupy(self):
        # An isolated interpreter cannot reuse this module's imported NumPy.
        # Select this checkout explicitly so an installed/parent checkout cannot
        # accidentally satisfy the test instead of the package under review.
        core = Path(__file__).resolve().parents[2] / 'core'
        source = r'''
import builtins
import sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def without_gpu_dependencies(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'cupy'):
        raise AssertionError('default CPU path imported optional dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_gpu_dependencies
from magna import compile_circuit, parse_circuit, run
compiled = compile_circuit(parse_circuit('H 0\nT 0'),
                           ideal_state=parse_circuit('H 0\nT 0'))
result = run(compiled, shots=8, seed=123)
assert result['status'] == 'complete' and result['completed'] == 8
assert result['estimates']['acceptance'] == 1
assert result['estimates']['error'] == 0
assert not any(name.startswith(('magna.cuda_', 'magna._cuda_')) for name in sys.modules)
'''
        completed = subprocess.run([sys.executable, '-I', '-c', source, str(core)],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_cpu_reference_does_not_import_cupy(self):
        original_import = builtins.__import__

        def no_cupy(name, *args, **kwargs):
            if name.split('.')[0] == 'cupy':
                raise AssertionError('CPU query reference tried to import CuPy')
            return original_import(name, *args, **kwargs)

        experiment = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0\nZ_ERROR(1/3) 0'),
                                                   ideal_state=parse_circuit('H 0\nT 0')))
        frames = np.stack([experiment.base_frame, experiment.base_frame ^ experiment.columns[0]])
        with patch('builtins.__import__', side_effect=no_cupy):
            queries = GpuQueries(experiment)
            result = cpu_evaluate(queries, frames)
        self.assertIsNone(queries._cp)
        self.assertIsNone(queries._plans)
        self.assertEqual(tuples(result), ((((1, 0, 0, 0), 0), ((1, 0, 0, 0), 0)),
                                           (((1, 0, 0, 0), 0), ((0, 0, 0, 0), 0))))

    def test_pending_options_edits_are_rejected_before_cpu_or_gpu_access(self):
        compiled = compile_circuit(parse_circuit('H 0\nT 0'), ideal_state=parse_circuit('H 0\nT 0'))
        experiment = CudaExperiment(compiled)
        query = GpuQueries(experiment)
        compiled.options['max_quotient_rank'] = 0
        with self.assertRaisesRegex(ValueError, 'options changed'):
            cpu_evaluate(query, experiment.base_frame[None, :])
        # The failure must occur before CuPy is imported.
        sampler = CudaSampler.__new__(CudaSampler)
        sampler._experiment, sampler._queries = experiment, query
        with self.assertRaisesRegex(ValueError, 'options changed'):
            sampler.run(1)
        with self.assertRaisesRegex(ValueError, 'options changed'):
            GpuQueries(experiment)
        self.assertIsNone(query._cp)

    def test_random_small_circuits_match_original_exact_evaluator(self):
        """Seed 264917: 240 circuits, 7,680 frames, Gauss and odd-phase sums.

        Each circuit includes a true T target, positive or inverted selection,
        and measurement-report noise. Physical draws and arbitrary full-frame
        updates are checked against the original native probability path.
        """
        rng = Random(264917)
        modes = set()
        compared = 0
        for trial in range(240):
            n = 1 + trial % 3
            lines = []
            for step in range(8):
                kind = rng.choice(['H', 'T', 'T_DAG', 'S', 'X', 'Y', 'Z',
                                   'SQRT_X', 'R', 'RX', 'CX', 'CZ', 'CCZ'])
                if kind == 'CCZ':
                    if n < 3:
                        kind = 'T'
                    else:
                        lines.append('CCZ 0 1 2')
                        continue
                if kind in ('CX', 'CZ'):
                    if n < 2:
                        kind = 'H'
                    else:
                        a, b = rng.sample(range(n), 2)
                        lines.append(f'{kind} {a} {b}')
                        continue
                qubit = rng.randrange(n)
                lines.append(f'{kind} {qubit}')
                if step % 2 == 0:
                    lines.append(f'DEPOLARIZE1(1/4) {qubit}')
            for qubit in range(max(1, n-1)):
                lines.extend([f'M(1/8) {qubit}', 'DETECTOR rec[-1]'])
            lines.append('POSTSELECT det[0]' if trial % 2 else 'POSTSELECT !det[0]')
            text = '\n'.join(lines)
            with self.subTest(trial=trial, circuit=text):
                experiment = CudaExperiment(compile_circuit(parse_circuit(text), task={'outputs': [n-1]},
                                                           ideal_state=parse_circuit('H 0\nT 0')))
                query = GpuQueries(experiment)
                modes.update(p['mode'] for p in query.metadata)
                batch = draw(experiment, [rng.getrandbits(64) for _ in range(12)])
                physical = cpu_frames(experiment, batch)
                raw = np.zeros((20, experiment.width), dtype=np.uint64)
                bits = experiment.metadata['frame_bits']
                for row in raw:
                    for axis in range(2):
                        for word in range(experiment.words):
                            live = min(64, max(0, bits-64*word))
                            row[axis*experiment.words + word] = rng.getrandbits(live)
                frames = np.concatenate([physical, raw])
                actual = tuples(cpu_evaluate(query, frames))
                original = evaluate_frames(experiment, frames)
                norm = ExactScalar(*experiment.metadata['target_normalization'])
                for values, expected in zip(actual, original, strict=True):
                    acceptance, target = map(lambda value: ExactScalar(*value), values)
                    self.assertEqual(acceptance, ExactScalar(*expected[1]))
                    self.assertEqual(acceptance*norm-target, ExactScalar(*expected[2]))
                    compared += 1
                self.assertIsNone(query._cp)
        self.assertEqual(compared, 7680)
        self.assertTrue({1, 3}.issubset(modes))


if __name__ == '__main__':
    unittest.main()
