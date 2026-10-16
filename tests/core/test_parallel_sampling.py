"""Parallel sampling through the same public Python and command-line entry points."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
from random import Random
import runpy
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from magna import ControllerRegistry, compile_circuit, parse_circuit, run
from magna.file_api import main


# Timing and work counters may depend on cache warmup and worker scheduling.
# These fields describe the draws and their statistical interpretation instead.
DRAW_FIELDS = (
    'status', 'requested', 'completed', 'failed', 'not_attempted', 'failures',
    'estimates', 'standard_errors', 'estimates_scientific',
    'standard_errors_scientific', 'weights', 'efficiency', 'sampled_records',
    'sampling', 'history_sampler', 'fingerprints', 'route', 'numerics',
)


class ParallelSamplingTests(unittest.TestCase):
    def assert_same_draws(self, actual, expected):
        for key in DRAW_FIELDS:
            self.assertEqual(actual[key], expected[key], key)

    def test_workers_requires_a_positive_integer(self):
        circuit = parse_circuit('I 0')
        for workers in (0, -1, True, False, 1.0, '2', None):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, 'workers must be a positive integer'):
                    run(circuit, workers=workers)

    def test_one_worker_preserves_the_existing_seeded_stream(self):
        circuit = parse_circuit('H 0\nT 0\nZ_ERROR(1/10) 0\nMX(1/7) 0')
        for method in ('affine_support', 'exact_marginal'):
            with self.subTest(method=method):
                kwargs = dict(shots=71, seed=391, task={'sample_records': [0]},
                              ideal_state=parse_circuit('H 0'),
                              options={'history_sampler': method})
                expected = run(circuit, **kwargs)
                actual = run(circuit, workers=1, **kwargs)
                self.assertEqual(expected['status'], 'complete', expected['failures'])
                self.assert_same_draws(actual, expected)

    def test_parallel_seed_and_record_order_are_independent_of_partition(self):
        circuit = parse_circuit('H 0\nT 0\nZ_ERROR(1/10) 0\nMX(1/7) 0\n'
                                'CX rec[-1] 1\nH 1\nT 1\nM 1')
        for method in ('affine_support', 'exact_marginal'):
            for sampling in (None, {'method': 'importance', 'proposal_scale': '2'}):
                with self.subTest(method=method, sampling=sampling):
                    kwargs = dict(shots=79, seed=-391, task={'sample_records': [0, 1]},
                                  ideal_state=parse_circuit('I 0 1'), sampling=sampling,
                                  options={'history_sampler': method})
                    expected = run(circuit, workers=2, **kwargs)
                    self.assertEqual(expected['status'], 'complete', expected['failures'])
                    self.assertEqual(len(expected['sampled_records']), kwargs['shots'])
                    for workers, interval in ((2, None), (3, None), (7, 11), (3, 1)):
                        with redirect_stderr(StringIO()):
                            actual = run(circuit, workers=workers, progress_every=interval, **kwargs)
                        self.assert_same_draws(actual, expected)
                    changed = run(circuit, workers=2, **dict(kwargs, seed=392))
                    self.assertNotEqual(changed['sampled_records'], expected['sampled_records'])

    def test_deterministic_quantum_moments_match_the_exact_state(self):
        # T H|0> has Bloch vector (1/sqrt(2), 1/sqrt(2), 0).
        circuit = parse_circuit('H 0\nT 0')
        kwargs = dict(shots=23, task={'observables': [[1, 0], [1, 1], [0, 1]]},
                      ideal_state=parse_circuit('H 0'))
        serial = run(circuit, **kwargs)
        for workers in (2, 3, 5):
            with self.subTest(workers=workers):
                result = run(circuit, workers=workers, **kwargs)
                self.assert_same_draws(result, serial)
                self.assertEqual(result['estimates']['acceptance'], 1)
                self.assertAlmostEqual(result['estimates']['error'], (1 - 1/math.sqrt(2))/2)
                for value, expected in zip(result['estimates']['moments'],
                                           (1/math.sqrt(2), 1/math.sqrt(2), 0)):
                    self.assertAlmostEqual(value, expected)

    def test_fewer_shots_than_workers_and_partial_final_partition(self):
        circuit = parse_circuit('X 0\nM 0')
        for shots in (1, 2, 5, 17):
            with self.subTest(shots=shots), redirect_stderr(StringIO()):
                result = run(circuit, shots=shots, workers=8, progress_every=3,
                             task={'sample_records': [0]})
                self.assertEqual((result['status'], result['completed']), ('complete', shots))
                self.assertEqual(result['sampled_records'], [[1]] * shots)
                self.assertEqual(result['failed'] + result['not_attempted'], 0)

    def test_physical_and_importance_estimates_match_independent_noise_law(self):
        # The reported bit is the XOR of independent Bernoulli(1/5), Bernoulli(1/7).
        circuit = parse_circuit('X_ERROR(1/5) 0\nM(1/7) 0\nOBSERVABLE_INCLUDE(0) rec[-1]')
        for sampling, record_mean in ((None, 2/7),
                ({'method': 'importance', 'proposal_scale': '2'}, 16/35)):
            with self.subTest(sampling=sampling):
                result = run(circuit, shots=4096, seed=818, workers=4, sampling=sampling,
                             task={'error_observables': [0], 'sample_records': [0]})
                self.assertEqual(result['status'], 'complete', result['failures'])
                for field, expected in (('acceptance', 1), ('error', 2/7), ('conditional_error', 2/7)):
                    error = abs(result['estimates'][field] - expected)
                    self.assertLessEqual(error, 6*result['standard_errors'][field] + 1e-12, field)
                records = [row[0] for row in result['sampled_records']]
                self.assertEqual(set(records), {0, 1})
                self.assertAlmostEqual(sum(records)/len(records), record_mean, delta=.05)
                if sampling is None:
                    self.assertEqual(result['estimates']['error'], sum(records)/len(records))

    def test_nonlinear_python_controller_keeps_records_and_feedback_consistent(self):
        calls = []
        registry = ControllerRegistry()
        def conjunction(bits, config):
            calls.append(tuple(bits))
            return (bits[0] & bits[1],)
        registry.register('and', conjunction, version='1')
        circuit = parse_circuit('H 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nCX ctrl[0] 2\nM 2\n'
                                'OBSERVABLE_INCLUDE(0) rec[-1]', controllers=registry)
        for method in ('affine_support', 'exact_marginal'):
            with self.subTest(method=method):
                calls.clear()
                result = run(circuit, shots=257, seed=37, workers=4,
                             options={'history_sampler': method},
                             task={'error_observables': [0], 'sample_records': [0, 1, 2]})
                self.assertEqual(result['status'], 'complete', result['failures'])
                self.assertEqual(len(calls), result['completed'])
                for a, b, c in result['sampled_records']:
                    self.assertEqual(c, a & b)
                self.assertEqual(result['estimates']['acceptance'], 1)
                self.assertEqual(result['estimates']['error'],
                                 sum(row[2] for row in result['sampled_records'])/result['completed'])

    def test_compiled_circuit_can_switch_between_serial_and_parallel_runs(self):
        circuit = parse_circuit('H 0\nT 0\nM(1/7) 0\nCX rec[-1] 1\nX_ERROR(1/5) 1')
        kwargs = dict(task={'outputs': [1], 'sample_records': [0]}, ideal_state=parse_circuit('I 0'))
        compiled = compile_circuit(circuit, **kwargs)
        parallel = run(circuit, shots=31, seed=9, workers=3, **kwargs)
        serial = run(circuit, shots=31, seed=9, workers=1, **kwargs)
        queries = {}
        for workers in (3, 1, 2, 1, 3):
            result = run(compiled, shots=31, seed=9, workers=workers)
            self.assert_same_draws(result, serial if workers == 1 else parallel)
            if workers in queries:
                self.assertEqual(result['resources']['queries'], queries[workers])
            queries[workers] = result['resources']['queries']

    def test_controller_failure_is_reported_and_compiled_circuit_remains_usable(self):
        fail = [True]
        registry = ControllerRegistry()
        def decoder(bits, config):
            if fail[0]:
                raise ValueError('parallel decoder failure')
            return bits
        registry.register('echo', decoder, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\n'
                                'CONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(circuit, task={'outputs': [1], 'sample_records': [0]})
        failed = run(compiled, shots=19, workers=4)
        self.assertEqual((failed['status'], failed['completed'], failed['failed']), ('failed', 0, 1))
        self.assertEqual(failed['not_attempted'], 18)
        self.assertEqual(failed['sampled_records'], [])
        self.assertEqual(failed['failures'][0]['phase'], 'sampling')
        self.assertEqual(failed['failures'][0]['shot'], 0)
        self.assertEqual(failed['failures'][0]['type'], 'ValueError')
        self.assertIn('parallel decoder failure', failed['failures'][0]['message'])
        self.assertIsNone(failed['estimates'])
        self.assertIn('partial_statistics', failed)
        fail[0] = False
        for workers in (4, 1, 2):
            result = run(compiled, shots=19, workers=workers)
            self.assertEqual((result['status'], result['completed']), ('complete', 19), result['failures'])

    def test_query_budget_is_shared_and_partial_results_keep_a_completed_prefix(self):
        circuit = parse_circuit('X_ERROR(1/3) 0\nM 0')
        compiled = compile_circuit(circuit, task={'sample_records': [0]}, options={'max_queries': 11})
        for _ in range(2):
            result = run(compiled, shots=100, workers=4)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failures'][0]['type'], 'ResourceLimit')
            self.assertIn('quantum-query budget', result['failures'][0]['message'])
            self.assertEqual(result['failures'][0]['shot'], result['completed'])
            self.assertEqual(len(result['sampled_records']), result['completed'])
            self.assertEqual(result['completed'] + result['failed'] + result['not_attempted'], 100)
            self.assertLessEqual(result['resources']['queries'], 11)
            self.assertIsNone(result['estimates'])
            self.assertIn('estimates', result['partial_statistics'])

    def test_time_budget_applies_to_all_workers_and_can_be_removed_for_reuse(self):
        delay = [.02]
        registry = ControllerRegistry()
        def decoder(bits, config):
            time.sleep(delay[0])
            return bits
        registry.register('slow', decoder, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\n'
                                'CONTROL[slow] det[-1] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(circuit)
        compiled.options['max_seconds'] = .05
        failed = run(compiled, shots=100, workers=4)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['failures'][0]['type'], 'ResourceLimit')
        self.assertIn('time budget', failed['failures'][0]['message'])
        self.assertEqual(failed['completed'] + failed['failed'] + failed['not_attempted'], 100)
        compiled.options.pop('max_seconds')
        delay[0] = 0
        result = run(compiled, shots=17, workers=4)
        self.assertEqual((result['status'], result['completed']), ('complete', 17), result['failures'])

    def test_reconfiguring_a_compiled_circuit_updates_cached_workers(self):
        circuit = parse_circuit('H 0\nT 0\nZ_ERROR(1/10) 0\nMX 0')
        kwargs = dict(task={'sample_records': [0]}, ideal_state=parse_circuit('H 0'))
        compiled = compile_circuit(circuit, **kwargs)
        self.assertEqual(run(compiled, shots=23, workers=4)['status'], 'complete')
        for method in ('exact_marginal', 'affine_support'):
            compiled.options['history_sampler'] = method
            actual = run(compiled, shots=41, seed=21, workers=4)
            expected = run(circuit, shots=41, seed=21, workers=4,
                           options={'history_sampler': method}, **kwargs)
            self.assert_same_draws(actual, expected)
        compiled.options['max_queries'] = 1
        failed = run(compiled, shots=100, workers=4)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['failures'][0]['type'], 'ResourceLimit')
        self.assertLessEqual(failed['resources']['queries'], 1)
        compiled.options.pop('max_queries')
        self.assertEqual(run(compiled, shots=23, workers=4)['status'], 'complete')

    def test_preparation_failure_preserves_existing_result_contract(self):
        circuit = parse_circuit('H 0\nT 0\nH 0\nT 0\nH 0\nM 0')
        result = run(circuit, shots=17, workers=3, options={'max_terms': 0})
        self.assertEqual((result['status'], result['completed'], result['failed'], result['not_attempted']),
                         ('failed', 0, 0, 17))
        self.assertEqual(result['failures'][0]['phase'], 'preparation')
        self.assertIsNone(result['failures'][0]['shot'])

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'POSIX interval timer required')
    def test_signal_handler_cannot_reenter_a_runtime_used_by_parallel_workers(self):
        registry = ControllerRegistry()
        def decoder(bits, config):
            time.sleep(.2)
            return bits
        registry.register('slow', decoder, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\n'
                                'CONTROL[slow] det[-1] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(circuit)
        rejected = []
        def handler(*_):
            try:
                compiled.sample(Random(5))
            except RuntimeError as exc:
                rejected.append(str(exc))
        previous = signal.signal(signal.SIGALRM, handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, .05)
            result = run(compiled, shots=4, workers=2)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        self.assertEqual(result['status'], 'complete', result['failures'])
        self.assertEqual(len(rejected), 1)
        self.assertIn('in use by parallel workers', rejected[0])
        self.assertEqual(run(compiled, shots=1)['status'], 'complete')

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'POSIX interval timer required')
    def test_main_thread_signal_interrupts_parallel_sampling(self):
        class Stop(Exception):
            pass
        def handler(*_):
            raise Stop()
        compiled = compile_circuit(parse_circuit('H 0\nT 0\nH 0\nM 0\n'
                                                 'DETECTOR rec[-1]\nPOSTSELECT det[0]'))
        previous = signal.signal(signal.SIGALRM, handler)
        started = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, .1)
            with self.assertRaises(Stop):
                run(compiled, shots=10**7, workers=4)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(run(compiled, shots=2, workers=2)['status'], 'complete')


class ParallelCommandLineTests(unittest.TestCase):
    def cli(self, *args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main([str(arg) for arg in args])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_existing_run_command_accepts_workers_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            circuit = Path(directory)/'protocol.magna'
            circuit.write_text('X_ERROR(1/5) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]')
            task = Path(directory)/'task.json'
            task.write_text(json.dumps({'error_observables': [0], 'sample_records': [0]}))
            code, output, progress = self.cli('run', circuit, '--task', task, '--shots', 29,
                                             '--seed', 17, '--workers', 3, '--progress-every', 11, '--json')
            self.assertEqual(code, 0, output + progress)
            actual = json.loads(output)
            expected = run(circuit, task=task, shots=29, seed=17, workers=3)
            self.assertEqual(actual['estimates'], expected['estimates'])
            self.assertEqual(actual['sampled_records'], expected['sampled_records'])
            for completed in (11, 22, 29):
                self.assertIn(f'[{completed}/29]', progress)
            code, output, _ = self.cli('run', circuit, '--workers', 0, '--json')
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output)['status'], 'invalid')
            self.assertIn('workers must be a positive integer', json.loads(output)['error'])

    def test_portable_task_runner_reads_workers_and_accepts_an_override(self):
        runner = Path(__file__).resolve().parents[2]/'scripts/run_task.py'
        with tempfile.TemporaryDirectory() as directory:
            task_directory = Path(directory)
            (task_directory/'protocol.magna').write_text('X_ERROR(1/5) 0\nM 0')
            (task_directory/'task.json').write_text(json.dumps({'sample_records': [0]}))
            (task_directory/'run.json').write_text(json.dumps({
                'circuit': 'protocol.magna', 'task': 'task.json',
                'shots': 13, 'seed': 71, 'workers': 3,
            }))
            for override, workers in (([], 3), (['--workers', '2'], 2), (['--workers', '1'], 1)):
                output = StringIO()
                # load() temporarily adds the chosen decoder directory to sys.path.
                with patch.object(sys, 'argv', [str(runner), directory, *override]), \
                        patch.object(sys, 'path', sys.path.copy()), redirect_stdout(output):
                    with self.assertRaises(SystemExit) as stopped:
                        runpy.run_path(str(runner), run_name='__main__')
                self.assertEqual(stopped.exception.code, 0)
                result = json.loads(output.getvalue())
                self.assertEqual((result['status'], result['completed'], result['workers']),
                                 ('complete', 13, workers))
                expected = run(task_directory/'protocol.magna', task=task_directory/'task.json',
                               shots=13, seed=71, workers=workers)
                self.assertEqual(result['sampled_records'], expected['sampled_records'])


if __name__ == '__main__':
    unittest.main()
