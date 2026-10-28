"""Real spawn/IPC tests without CUDA, plus opt-in device parity tests."""
from copy import deepcopy
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna import compile_circuit, parse_circuit
from magna.cuda_sampler import CudaSampler
from magna.cuda_multi import MultiGpuCudaSampler, _compile_snapshot, _partitions, _snapshot
from tests.support.cuda_multi import cpu_worker, fail_worker, crash_worker, hang_worker, startup_failure
from tests.support.cuda_reference import cpu_run, cpu_sample, cpu_evaluate, contributions
from tests.support.cuda import require_cuda


def task(*, importance=False, target=True):
    return compile_circuit(parse_circuit('H 0\nT 0\nX_ERROR(1/8) 0\nZ_ERROR(1/8) 0'),
                           ideal_state=parse_circuit('H 0\nT 0') if target else None,
                           sampling={'method': 'importance', 'proposal_scale': '2'} if importance else None)


def assert_statistics(test, actual, expected, rtol=2e-12):
    for section in ('estimates', 'standard_errors'):
        for name, value in expected[section].items():
            if value is None:
                test.assertIsNone(actual[section][name])
            else:
                np.testing.assert_allclose(actual[section][name], value, rtol=rtol, atol=1e-15)


class MultiGpuControlTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch('magna.cuda_multi._worker_main', cpu_worker)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_partitions_cover_extreme_intervals_once(self):
        for shots in (0, 1, 3, 8, 31, 1001, 2**53):
            for offset in (0, 2**32-5, 2**64-max(1, shots)):
                parts = _partitions(shots, offset, (7, 2, 4, 1, 6, 5, 0, 3))
                self.assertEqual(sum(p['shots'] for p in parts), shots)
                cursor = offset
                for part in parts:
                    self.assertEqual(part['offset'], cursor)
                    self.assertGreater(part['shots'], 0)
                    cursor += part['shots']
                self.assertEqual(cursor, offset+shots)
                if parts:
                    self.assertLessEqual(max(p['shots'] for p in parts)-min(p['shots'] for p in parts), 1)

    def test_real_process_shards_match_serial_physical_and_importance(self):
        for importance in (False, True):
            compiled = task(importance=importance)
            expected = cpu_run(CudaSampler(compiled), 257, seed=91, offset=2**32-19, batch_size=17)
            for devices in ((0,), (3, 1), (7, 2, 0), tuple(range(8))):
                with self.subTest(importance=importance, devices=devices):
                    with MultiGpuCudaSampler(compiled, devices=devices, timeout=30) as pool:
                        actual = pool.run(257, seed=91, offset=2**32-19, batch_size=31)
                        self.assertEqual(actual['completed'], 257)
                        assert_statistics(self, actual, expected)

    def test_aggregate_covariance_against_independent_per_draw_numpy(self):
        compiled = task()
        single = CudaSampler(compiled)
        frames = cpu_sample(single._noise, 193, 7, offset=13)
        exact = cpu_evaluate(single._queries, frames)
        a, e = contributions(exact.coefficients, exact.powers, single._norm_power)
        ratio = e.mean()/a.mean()
        expected = dict(estimates=dict(acceptance=a.mean(), error=e.mean(), conditional_error=ratio),
                        standard_errors=dict(acceptance=a.std(ddof=1)/np.sqrt(len(a)),
                                             error=e.std(ddof=1)/np.sqrt(len(a)),
                                             conditional_error=(e-ratio*a).std(ddof=1)/np.sqrt(len(a))/a.mean()))
        with MultiGpuCudaSampler(compiled, devices=(4, 1, 6), timeout=30) as pool:
            assert_statistics(self, pool.run(193, seed=7, offset=13, batch_size=11), expected)

    def test_reuse_zero_and_fewer_shots_than_devices(self):
        compiled = task(target=False)
        with MultiGpuCudaSampler(compiled, devices=(7, 2, 0), timeout=30) as pool:
            zero = pool.run(0)
            self.assertEqual(pool._workers, {})
            self.assertEqual(zero['shards'], [])
            self.assertIsNone(zero['estimates']['acceptance'])
            one = pool.run(1, seed=2, offset=2**64-1)
            self.assertEqual(list(pool._workers), [7])
            self.assertIsNone(one['standard_errors']['acceptance'])
            pool.run(17)
            pids = {d: w.process.pid for d, w in pool._workers.items()}
            for shots in (0, 2, 29):
                result = pool.run(shots, seed=19, offset=3, batch_size=7)
                expected = cpu_run(CudaSampler(compiled), shots, seed=19, offset=3)
                assert_statistics(self, result, expected)
                self.assertEqual(pids, {d: w.process.pid for d, w in pool._workers.items()})
        self.assertFalse(set(pids.values()) & {p.pid for p in mp.active_children()})
        pool.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pool.run(1)

    def test_invalid_arguments_never_spawn_and_leave_pool_reusable(self):
        with MultiGpuCudaSampler(task(), devices=(0, 1)) as pool:
            cases = [dict(shots=-1), dict(shots=True), dict(shots=2**53+1),
                     dict(shots=2, offset=2**64-1), dict(shots=1, seed=-1),
                     dict(shots=1, batch_size=True), dict(shots=1, batch_size=np.bool_(True)),
                     dict(shots=1, batch_size=0), dict(shots=1, batch_size=1_000_001)]
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                    pool.run(**kwargs)
                self.assertEqual(pool._workers, {})
            self.assertEqual(pool.run(1)['completed'], 1)

    def test_invalid_devices_and_timeout(self):
        compiled = task()
        for devices in ((), (0, 0), (-1,), (True,), (1.0,), ('0',), (2**31,)):
            with self.subTest(devices=devices), self.assertRaises((ValueError, TypeError)):
                MultiGpuCudaSampler(compiled, devices=devices)
        for timeout in (-1, 0, float('nan'), float('inf'), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                MultiGpuCudaSampler(compiled, devices=(0,), timeout=timeout)

    def test_single_gpu_rejects_boolean_batch_before_cuda(self):
        sampler = CudaSampler(task())
        with patch.object(sampler, '_chunks', side_effect=AssertionError('must not sample')):
            for batch in (True, False, np.bool_(True), np.bool_(False)):
                with self.subTest(batch=batch), self.assertRaisesRegex(TypeError, 'batch_size'):
                    sampler.run(1, batch_size=batch)

    def test_failures_and_hard_exit_close_all_children(self):
        for worker, message in ((fail_worker, 'injected device'), (crash_worker, 'worker'),
                                (startup_failure, 'initialization')):
            before = {p.pid for p in mp.active_children()}
            with patch('magna.cuda_multi._worker_main', worker):
                pool = MultiGpuCudaSampler(task(), devices=(0, 1), timeout=20)
                with self.assertRaisesRegex(RuntimeError, message):
                    pool.run(17)
                self.assertTrue(pool._closed)
                self.assertEqual(pool._workers, {})
                self.assertEqual(before, {p.pid for p in mp.active_children()})

    def test_timeout_reaps_hung_processes(self):
        with patch('magna.cuda_multi._worker_main', hang_worker):
            pool = MultiGpuCudaSampler(task(), devices=(0, 1), timeout=2)
            before = {p.pid for p in mp.active_children()}
            with self.assertRaises(TimeoutError):
                pool.run(10)
            self.assertTrue(pool._closed)
            self.assertEqual(before, {p.pid for p in mp.active_children()})

    def test_interrupt_reaps_started_workers(self):
        pool = MultiGpuCudaSampler(task(), devices=(0, 1), timeout=30)
        pool.run(2)
        pids = {w.process.pid for w in pool._workers.values()}
        with patch.object(pool, '_collect', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                pool.run(100)
        self.assertFalse(pids & {p.pid for p in mp.active_children()})

    def test_concurrent_access_and_option_edits_rejected(self):
        compiled = task()
        with MultiGpuCudaSampler(compiled, devices=(0,)) as pool:
            with pool._lock:
                for operation in (lambda: pool.run(1), pool.close):
                    with self.assertRaisesRegex(RuntimeError, 'already in use'):
                        operation()
            compiled.options['max_terms'] = 17
            with self.assertRaisesRegex(ValueError, 'options changed'):
                pool.run(1)
            self.assertEqual(pool._workers, {})

    def test_unsupported_task_rejected_before_spawn(self):
        compiled = compile_circuit(parse_circuit('H 0\nM 0'), task={'sample_records': [0]})
        with self.assertRaises(ValueError):
            MultiGpuCudaSampler(compiled, devices=(0, 1))

    def test_snapshot_keeps_resolved_target_and_recovery_after_files_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'physical.magna').write_text('H 0\nX_ERROR(1/8) 0')
            (root/'ideal.magna').write_text('H 0')
            (root/'recovery.magna').write_text('X 0')
            (root/'task.json').write_text('{"ideal_state":"ideal.magna","recovery":"recovery.magna"}')
            compiled = compile_circuit(root/'physical.magna', task=root/'task.json')
            pool = MultiGpuCudaSampler(compiled, devices=(0, 1), timeout=30)
        with pool:
            assert_statistics(self, pool.run(31, seed=2), cpu_run(CudaSampler(compiled), 31, seed=2))

    def test_snapshot_target_mapping_and_classical_error_observables(self):
        mapped = compile_circuit(parse_circuit('H 0\nX_ERROR(1/8) 0'), ideal_state={
            'circuit': parse_circuit('H 0\nH 1'), 'outputs': [0], 'postselection': {1: 0}})
        classical = compile_circuit(parse_circuit('X_ERROR(1/8) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]'),
                                    task={'outputs': [], 'error_observables': [0]})
        for compiled in (mapped, classical):
            with self.subTest(task=compiled.task['raw']):
                snapshot = _snapshot(compiled)
                before = deepcopy(snapshot)
                rebuilt = _compile_snapshot(snapshot)
                self.assertEqual(snapshot, before)
                assert_statistics(self, cpu_run(CudaSampler(rebuilt), 51, seed=11),
                                  cpu_run(CudaSampler(compiled), 51, seed=11))
                with MultiGpuCudaSampler(compiled, devices=(0, 1), timeout=30) as pool:
                    assert_statistics(self, pool.run(51, seed=11), cpu_run(CudaSampler(compiled), 51, seed=11))


class MultiGpuDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = os.environ.get('MAGNA_TEST_CUDA_DEVICES')
        if not selected:
            raise unittest.SkipTest('set MAGNA_TEST_CUDA_DEVICES to explicitly allocate at least two GPUs')
        cls.devices = tuple(int(d) for d in selected.split(','))
        if len(cls.devices) < 2 or len(set(cls.devices)) != len(cls.devices):
            raise ValueError('MAGNA_TEST_CUDA_DEVICES must list at least two distinct device IDs')
        cls.cp = require_cuda()
        if any(d < 0 or d >= cls.cp.cuda.runtime.getDeviceCount() for d in cls.devices):
            raise ValueError('MAGNA_TEST_CUDA_DEVICES contains an unavailable device')

    def test_real_devices_physical_importance_and_reuse(self):
        devices = self.devices
        for importance in (False, True):
            compiled = task(importance=importance)
            expected = cpu_run(CudaSampler(compiled), 1003, seed=39, offset=2**32-7)
            with MultiGpuCudaSampler(compiled, devices=devices, timeout=180) as pool:
                for batch in (127, 256):
                    result = pool.run(1003, seed=39, offset=2**32-7, batch_size=batch)
                    assert_statistics(self, result, expected)
                    self.assertEqual(result['completed'], 1003)
                one = pool.run(1)
                self.assertEqual(len(one['shards']), 1)


if __name__ == '__main__':
    unittest.main()
