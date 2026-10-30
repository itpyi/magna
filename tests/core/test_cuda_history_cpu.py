"""Native history plans and exact replay do not require a CUDA installation."""
import math
import struct
import unittest

from magna import ControllerRegistry, _native, compile_circuit, parse_circuit
from magna.scalar import ExactScalar


def history(text, method='affine_support', ideal='I 0', outputs=(1,)):
    registry = ControllerRegistry()
    registry.register('copy', lambda bits, config: bits, version='1')
    compiled = compile_circuit(parse_circuit(text, controllers=registry),
        task={'outputs': list(outputs)}, ideal_state=parse_circuit(ideal),
        options={'history_sampler': method})
    experiment, _ = _native.make_cuda_experiment(compiled._execution.handle)
    handle, plan = _native.make_cuda_history(experiment)
    return compiled, handle, plan


def number(value):
    return float(ExactScalar(*value).real_decimal())


class CudaHistoryCpuTests(unittest.TestCase):
    def test_replay_preserves_importance_likelihoods(self):
        registry = ControllerRegistry()
        registry.register('copy', lambda bits, config: bits, version='1')
        circuit = parse_circuit('H 0\nZ_ERROR(1/4) 0\nM 0\nDETECTOR rec[-1]\n'
            'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1\nX_ERROR(1/4) 1', controllers=registry)
        for method in ('affine_support', 'exact_marginal'):
            compiled = compile_circuit(circuit, task={'outputs': [1]}, ideal_state=parse_circuit('I 0'),
                options={'history_sampler': method}, sampling={'method': 'importance', 'proposal_scale': '2'})
            experiment, _ = _native.make_cuda_experiment(compiled._execution.handle)
            handle, _ = _native.make_cuda_history(experiment)
            _, raw_weights, raw_offsets, raw_actions = _native._cuda_noise_cpu_sample(experiment, 32, 19, 3, True, True)
            weights = struct.unpack('<32d', raw_weights)
            offsets = struct.unpack('<33Q', raw_offsets)
            actions = struct.unpack('<'+'I'*(len(raw_actions)//4), raw_actions)
            for shot, weight in enumerate(weights):
                inputs = _native.cuda_history_replay(handle, 19, shot+3, (), ())
                row = _native.cuda_history_replay(handle, 19, shot+3, (inputs,), (inputs,))
                self.assertEqual(math.ldexp(*row[0]), weight)
                wrong = inputs[0] ^ (1 in actions[offsets[shot]:offsets[shot+1]])
                self.assertEqual(math.ldexp(*row[1]), weight*wrong)

    def test_constrained_irrational_entries_and_exact_terminal_caps(self):
        text = ('H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\n'
                'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1')
        for method in ('affine_support', 'exact_marginal'):
            compiled, handle, plan = history(text, method)
            self.assertEqual(plan['open_bits'], 1)
            self.assertGreaterEqual(len(plan['entries']), 2)
            self.assertEqual(len(plan['sites']), 1)
            for shot in range(16):
                first = _native._cuda_history_cpu_reference(handle, 2026, shot, ())
                self.assertEqual(len(first['frame']), 8*plan['frame_words'])
                support = _native._cuda_history_support_reference(handle, 0, first['frame'])
                self.assertFalse(support['zero'])
                self.assertEqual(support['rank'], 1)
                self.assertEqual(int.from_bytes(support['base'], 'little'), 0)
                self.assertEqual(int.from_bytes(support['columns'], 'little'), 1)
                inputs = first['inputs'][0]
                self.assertEqual(_native.cuda_history_replay(handle, 2026, shot, (), ()), inputs)
                complete = _native._cuda_history_cpu_reference(handle, 2026, shot, (inputs,))
                exact = complete['queries']
                history_mass, acceptance_mass, error_mass = complete['masses'][0]
                self.assertEqual(number(exact[0]), number(history_mass))
                self.assertEqual(number(exact[1]), number(acceptance_mass))
                self.assertAlmostEqual(number(exact[1])-number(exact[2]), number(error_mass), places=14)
                expected = _native.cuda_history_replay(handle, 2026, shot, (inputs,), (inputs,))
                self.assertEqual(expected, complete['contribution'])
                acceptance = math.ldexp(*expected[0])
                if method == 'exact_marginal':
                    self.assertEqual(acceptance, 1.)
                else:
                    self.assertEqual(complete['proposal_rank'], 1)
                    self.assertAlmostEqual(acceptance, 1 + (1-2*inputs[0])*math.sqrt(.5))

    def test_saved_ctrl_inputs_and_replay_share_the_addressed_site(self):
        text = ('H 0\nM 0\nDETECTOR rec[-1]\n'
                'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1\n'
                'H 2\nM 2\nDETECTOR rec[-1]\n'
                'CONTROL[copy] det[1] ctrl[0] -> ctrl[1] ctrl[2]\n'
                'CX ctrl[1] 1\nCZ ctrl[2] 1')
        compiled, handle, plan = history(text, 'exact_marginal')
        self.assertEqual([site['input_count'] for site in plan['sites']], [1, 2])
        for shot in range(16):
            first = _native.cuda_history_replay(handle, 19, shot, (), ())
            second = _native.cuda_history_replay(handle, 19, shot, (first,), (first,))
            self.assertEqual(second[1], first[0])
            reference = _native._cuda_history_cpu_reference(handle, 19, shot, (first,))
            self.assertEqual(len(reference['frame']), 8*plan['frame_words'])
            self.assertEqual(second, reference['inputs'][1])
            result = _native.cuda_history_replay(handle, 19, shot, (first, second), (first, second))
            self.assertEqual(result[0], (.5, 1))

    def test_postselected_target_normalization_is_exact_in_terminal_plans(self):
        registry = ControllerRegistry()
        registry.register('copy', lambda bits, config: bits, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\n'
            'CONTROL[copy] det[0] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        for method in ('affine_support', 'exact_marginal'):
            compiled = compile_circuit(circuit, task={'outputs': [1]},
                ideal_state={'circuit': parse_circuit('H 1\nT 1\nH 1'),
                             'outputs': [0], 'postselection': {1: 0}},
                options={'history_sampler': method})
            experiment, metadata = _native.make_cuda_experiment(compiled._execution.handle)
            handle, plan = _native.make_cuda_history(experiment)
            norm = ExactScalar(*metadata['target_normalization'])
            self.assertEqual([q['mode'] for q in plan['terminal'][:2]], [6, 5])
            self.assertEqual(plan['query_bases'][0]['mode'], 5)
            for shot in range(8):
                inputs = _native.cuda_history_replay(handle, 42, shot, (), ())
                complete = _native._cuda_history_cpu_reference(handle, 42, shot, (inputs,))
                p, a, f = map(lambda value: ExactScalar(*value), complete['queries'])
                old_p, old_a, old_e = map(lambda value: ExactScalar(*value), complete['masses'][0])
                self.assertEqual(p, old_p*norm)
                self.assertEqual(a, old_a*norm)
                self.assertEqual(a-f, old_e)
                self.assertEqual(complete['contribution'][0], (.5, 1))

    def test_compact_prefixes_share_one_query_and_preserve_conditional_masses(self):
        text = ('H 0\nT 0\nH 0\nM 0\nH 2\nT 2\nH 2\nM 2\n'
                'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                'CONTROL[copy] det[0] det[1] -> ctrl[0] ctrl[1]\n'
                'CX ctrl[0] 1\nCZ ctrl[1] 1')
        compiled, handle, plan = history(text, 'exact_marginal')
        self.assertEqual(len(plan['query_bases']), 1)
        self.assertEqual([p['disabled_caps'] for p in plan['sites'][0]['prefixes']], [2, 1, 0])
        probability = (ExactScalar((2, 1, 0, -1), -2), ExactScalar((2, -1, 0, 1), -2))
        for shot in range(12):
            row = _native._cuda_history_cpu_reference(handle, 76, shot, ())
            selected = row['inputs'][0]
            expected = [ExactScalar((1, 0, 0, 0)), probability[selected[0]],
                        probability[selected[0]]*probability[selected[1]]]
            for prefix, mass in enumerate(expected):
                actual = _native._cuda_history_prefix_cpu_evaluate(handle, 0, prefix, row['frame'])
                self.assertEqual(ExactScalar(*actual), mass)


if __name__ == '__main__':
    unittest.main()
