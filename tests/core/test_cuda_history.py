"""Decoder histories retain their probabilities, feedback and draw counts."""
import math
import unittest
from unittest.mock import patch

from magna import _native, ControllerRegistry, compile_circuit, parse_circuit
from tests.support.cuda import require_cuda
from tests.support.cuda_reference import history_draw

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna.cuda_sampler import CudaSampler
from magna._cuda_history_statistics import _scientific_moments, _summarize_scaled


class HistoryCapabilityTests(unittest.TestCase):
    def test_exact_marginal_decoder_is_rejected_at_construction(self):
        registry = ControllerRegistry()
        registry.register('copy', lambda bits, config: bits, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(circuit, options={'history_sampler': 'exact_marginal'})
        with self.assertRaisesRegex(ValueError, 'affine_support; use magna.run for exact_marginal'):
            CudaSampler(compiled)

    def test_deferred_history_features_name_the_restriction(self):
        cases = [
            ('H 0\nM 0', {'task': {'sample_records': [0]}}, 'sample_records'),
            ('H 0\nM 0\nX_ERROR(1/4) rec[-1] 1', {}, 'conditional noise'),
        ]
        for text, settings, reason in cases:
            with self.subTest(reason=reason):
                compiled = compile_circuit(parse_circuit(text), **settings)
                with self.assertRaisesRegex(ValueError, reason):
                    CudaSampler(compiled)


class DeviceHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()


    def test_irrational_history_probability_and_affine_weights(self):
        expected = (1 - math.sqrt(0.5)) / 2
        with self.subTest(method='affine_support'):
            observed = []
            registry = ControllerRegistry()
            def decode(bits, config):
                observed.append(bits[0])
                return bits
            registry.register('copy', decode, version='1')
            circuit = parse_circuit('H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n'
                'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
            compiled = compile_circuit(circuit, task={'outputs': [1]},
                ideal_state=parse_circuit('I 0'), options={'history_sampler': 'affine_support'})
            result = CudaSampler(compiled).run(8192, seed=1023, batch_size=4096)
            self.assertEqual(result['completed'], 8192)
            self.assertEqual(len(observed), 8192)
            for metric, truth in (('acceptance', 1.), ('error', expected),
                                  ('conditional_error', expected)):
                self.assertLessEqual(abs(result['estimates'][metric] - truth),
                    6 * result['standard_errors'][metric] + 1e-14)
            self.assertLess(abs(sum(observed) / len(observed) - .5), .04)
            self.assertGreater(result['standard_errors']['acceptance'], 0.)

    def test_nonunit_target_normalization_stays_in_history_contributions(self):
        target = parse_circuit(''.join(f'H {q}\nT {q}\nH {q}\n' for q in range(1, 5))+
                               ''.join(f'CX {q} 5\n' for q in range(1, 5)))
        observed = []
        registry = ControllerRegistry()
        def decode(bits, config):
            observed.append(bits[0])
            return bits
        registry.register('copy', decode, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\n'
            'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(circuit, task={'outputs': [1]},
            ideal_state={'circuit': target, 'outputs': [0], 'postselection': {5: 0}},
            options={'history_sampler': 'affine_support'})
        sampler = CudaSampler(compiled)
        self.assertEqual(sampler._experiment.metadata['target_normalization'], ((5, 0, 0, 0), -3))
        result = sampler.run(257, seed=94, batch_size=79)
        with self.subTest(method='affine_support'):
            self.assertEqual(result['completed'], 257)
            self.assertEqual(result['estimates']['acceptance'], 1.)
            self.assertEqual(result['standard_errors']['acceptance'], 0.)
            self.assertAlmostEqual(result['estimates']['error'], sum(observed)/257, places=14)

    def test_rejected_draws_and_failed_corrections_stay_in_the_average(self):
        with self.subTest(method='affine_support'):
            observed = []
            registry = ControllerRegistry()
            def decode(bits, config):
                observed.append(bits[0])
                return bits
            registry.register('copy', decode, version='1')
            circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\n'
                'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1\nPOSTSELECT det[0]',
                controllers=registry)
            compiled = compile_circuit(circuit, task={'outputs': [1]},
                ideal_state=parse_circuit('X 0'), options={'history_sampler': 'affine_support'})
            result = CudaSampler(compiled).run(1024, seed=18, batch_size=127)
            accepted = observed.count(0) / 1024
            self.assertEqual(len(observed), 1024)
            self.assertEqual(result['completed'], 1024)
            self.assertAlmostEqual(result['estimates']['acceptance'], accepted, places=14)
            self.assertAlmostEqual(result['estimates']['error'], accepted, places=14)
            self.assertAlmostEqual(result['estimates']['conditional_error'], 1., places=14)
            self.assertGreater(result['standard_errors']['acceptance'], 0.)

    def test_prior_decisions_and_histories_are_independent_of_chunk_size(self):
        traces, results = [], []
        for batch_size in (257, 29):
            observed = [[], []]
            registry = ControllerRegistry()
            def first(bits, config):
                self.assertEqual(config['visits'], 0)
                config['visits'] += 1
                observed[0].append(bits)
                return bits
            def second(bits, config):
                observed[1].append(bits)
                return (bits[0] ^ bits[1],)
            registry.register('first', first, version='1', config={'visits': 0})
            registry.register('second', second, version='1')
            circuit = parse_circuit('H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n'
                'CONTROL[first] det[0] -> ctrl[0]\nCX ctrl[0] 1\n'
                'H 2\nM 2\nDETECTOR rec[-1]\n'
                'CONTROL[second] det[1] ctrl[0] -> ctrl[1]\nCX ctrl[1] 1',
                controllers=registry)
            compiled = compile_circuit(circuit, task={'outputs': [1]},
                ideal_state=parse_circuit('I 0'), options={'history_sampler': 'affine_support'})
            results.append(CudaSampler(compiled).run(257, seed=91, offset=37,
                batch_size=batch_size))
            self.assertEqual(len(observed[0]), 257)
            self.assertEqual(len(observed[1]), 257)
            self.assertEqual([x[0] for x in observed[0]], [x[1] for x in observed[1]])
            traces.append(observed)
        with self.subTest(method='affine_support'):
            self.assertEqual(traces[0], traces[1])
            for group in ('estimates', 'standard_errors'):
                np.testing.assert_allclose(list(results[0][group].values()),
                    list(results[1][group].values()), rtol=1e-12, atol=1e-15)


    def test_importance_history_and_terminal_completion_match_native_draws(self):
        observed = []
        registry = ControllerRegistry()
        def decode(bits, config):
            observed.append(bits)
            return bits
        registry.register('copy', decode, version='1')
        circuit = parse_circuit('H 0\nT 0\nZ_ERROR(1/4) 0\nH 0\nM 0\nDETECTOR rec[-1]\n'
            'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1\nX_ERROR(1/4) 1', controllers=registry)
        compiled = compile_circuit(circuit, task={'outputs': [1]}, ideal_state=parse_circuit('I 0'),
            options={'history_sampler': 'affine_support'}, sampling={'method': 'importance', 'proposal_scale': '2'})
        sampler = CudaSampler(compiled)
        history = sampler._history
        original = history.terminal.evaluate
        def continuation(*args, **kwargs):
            result = original(*args, **kwargs)
            result.status[0, 0] = 1
            return result
        with patch.object(history.terminal, 'evaluate', side_effect=continuation), \
                patch.object(_native, 'cuda_history_replay', wraps=_native.cuda_history_replay) as replay:
            result = sampler.run(129, seed=817, offset=5, batch_size=129)
            self.assertEqual(replay.call_count, 1)
        self.assertEqual(len(observed), 129)
        gpu_inputs = list(observed)
        rows = [history_draw(sampler._experiment, history.handle, history.metadata, 817, j+5)
                for j in range(129)]
        self.assertEqual(gpu_inputs, observed[129:])
        expected = _summarize_scaled(_scientific_moments([row[2]['contribution'] for row in rows]))
        with self.subTest(method='affine_support'):
            self.assertEqual(result['completed'], 129)
            for group in expected:
                np.testing.assert_allclose(list(result[group].values()), list(expected[group].values()),
                                           rtol=1e-12, atol=1e-15)

    def test_pending_observations_and_auxiliary_state_cross_words(self):
        observed = []
        registry = ControllerRegistry()
        def decode(bits, config):
            observed.append(bits)
            return (bits[-1],)
        registry.register('last', decode, version='1')
        qubits = ' '.join(map(str, range(70)))
        text = f'H {qubits}\nT 69\nH 69\nM {qubits}\n'
        text += ''.join(f'DETECTOR rec[{j-70}]\n' for j in range(70))
        text += 'CONTROL[last] '+' '.join(f'det[{j}]' for j in range(70))+' -> ctrl[0]\nCX ctrl[0] 70'
        compiled = compile_circuit(parse_circuit(text, controllers=registry), task={'outputs': [70]},
            ideal_state=parse_circuit('I 0'), options={'history_sampler': 'affine_support'})
        sampler = CudaSampler(compiled)
        history = sampler._history
        self.assertEqual(history.sites[0]['pending_count'], 70)
        result = sampler.run(129, seed=841, offset=7, batch_size=129)
        self.assertEqual(len(observed), 129)
        gpu_inputs = list(observed)
        rows = [history_draw(sampler._experiment, history.handle, history.metadata, 841, j+7)
                for j in range(129)]
        self.assertEqual(gpu_inputs, observed[129:])
        expected = _summarize_scaled(_scientific_moments([row[2]['contribution'] for row in rows]))
        with self.subTest(method='affine_support'):
            for group in expected:
                np.testing.assert_allclose(list(result[group].values()), list(expected[group].values()),
                                           rtol=1e-12, atol=1e-15)

    def test_irrational_target_normalization_is_removed_from_contributions(self):
        registry = ControllerRegistry()
        registry.register('copy', lambda bits, config: bits, version='1')
        circuit = parse_circuit('H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n'
            'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        ideal = dict(circuit=parse_circuit('I 0\nH 1\nT 1\nH 1'), outputs=[0], postselection={1: 0})
        compiled = compile_circuit(circuit, task={'outputs': [1]}, ideal_state=ideal,
            options={'history_sampler': 'affine_support'})
        sampler = CudaSampler(compiled)
        history = sampler._history
        result = sampler.run(129, seed=118, batch_size=37)
        rows = [history_draw(sampler._experiment, history.handle, history.metadata, 118, j)
                for j in range(129)]
        expected = _summarize_scaled(_scientific_moments([row[2]['contribution'] for row in rows]))
        with self.subTest(method='affine_support'):
            for group in expected:
                np.testing.assert_allclose(list(result[group].values()), list(expected[group].values()),
                                           rtol=1e-12, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
