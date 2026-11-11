"""Whole-task parent ownership and independent finite physical probabilities."""
from random import Random
from fractions import Fraction
from itertools import product
import unittest

from magna import ControllerRegistry, compile_circuit, enumerate_errors, parse_circuit
from magna.file_api import NoiseDraw
from tests.support import file_hooks as hooks


class ReferenceNetworkTests(unittest.TestCase):
    def test_exact_mass_reuse_skips_terminal_sampling_query(self):
        # The first case has only a nonzero affine sector. The second has
        # one noncommuting, nonuniform query and then a dependent record.
        for prefix, queries in (("X 0\nM 0\n", 0), ("H 0\nT 0\nMX 0\n", 1)):
            registry = ControllerRegistry()
            registry.register('echo', lambda bits, config: bits, version='1')
            physical = parse_circuit(prefix +
                'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 1\n'
                'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[1]\n'
                'Z_ERROR(1) ctrl[1] 2', controllers=registry)
            compiled = compile_circuit(physical, task={'outputs': [1], 'sample_records': [0]},
                ideal_state=parse_circuit('I 0'), options={'history_sampler': 'exact_marginal'})
            parents = {row['purpose']: row for row in self.parents(compiled)}
            self.assertEqual(parents['acceptance']['retained_labels'], 0)
            self.assertEqual(parents['fidelity']['retained_labels'], 0)
            self.assertEqual(parents['sampling']['retained_labels'], queries)
            if not queries:
                self.assertIn(1, parents['acceptance']['reference_outcomes'])
            for seed in range(16):
                draw = compiled.sample(Random(seed))
                bit, = draw['sampled_records']
                self.assertEqual(draw['acceptance'], 1)
                self.assertEqual(draw['error'], bit)
                mass = draw['exact']['history_mass']
                value = tuple(Fraction(c)*Fraction(2)**mass['power'] for c in mass['coefficients'])
                self.assertEqual(value, (1, 0, 0, 0) if not queries else
                    (Fraction(1, 2), Fraction(1-2*bit, 4), 0, Fraction(2*bit-1, 4)))
            state = compiled._execution.state()
            self.assertEqual(state['completed_history_reuses'], 32)
            self.assertEqual(state['target_frame_reuses'], 16)
            evaluations = state['reference_network_evaluations']
            self.assertEqual(evaluations.get('sampling', 0), 16*queries)
            self.assertEqual(evaluations['acceptance'], 16)
            self.assertEqual(evaluations['fidelity'], 16)

    def test_shared_coordinates_preserve_parity_feedback_and_actual_offsets(self):
        # Observe only the circuit's declared controller inputs. Independent
        # oracle: S|+> has fair X outcomes; |+> has deterministic X outcome 0.
        for sampler in ('affine_support', 'exact_marginal'):
            seen = []
            registry = ControllerRegistry()
            def echo(bits, config):
                seen.append(tuple(bits))
                return bits
            registry.register('echo', echo, version='1')
            circuit = parse_circuit('X 0\nH 1 2\nM 0 1\nM 1\n'
                'CONTROL[XOR] rec[-3] rec[-2] -> ctrl[0]\n'
                'DETECTOR rec[-1]\n'
                'CONTROL[echo] ctrl[0] det[-1] -> ctrl[1] ctrl[2]\n'
                'CS ctrl[1] 2\nCX ctrl[2] 4\nMX 2\n'
                'DETECTOR rec[-1]\n'
                'CONTROL[echo] det[-1] -> ctrl[3]\nCX ctrl[3] 3\n'
                'DETECTOR rec[-4] rec[-2]\nPOSTSELECT det[2]', controllers=registry)
            compiled = compile_circuit(circuit, task={'outputs': [3]},
                ideal_state=parse_circuit('I 0'), options={'history_sampler': sampler})
            for flip0, flip2, flip3 in product((0, 1), repeat=3):
                flips = tuple(f'm{j}' for j, flip in ((0, flip0), (2, flip2), (3, flip3)) if flip)
                found = set()
                for seed in range(32):
                    seen.clear()
                    draw = compiled.sample(Random(seed), noise=NoiseDraw({}, flips))
                    s, reported_copy = seen[0];b, = seen[1]
                    a = reported_copy ^ flip2
                    self.assertEqual(s, 1 ^ flip0 ^ a)
                    if not s:
                        self.assertEqual(b, flip3)
                    expected_mass = Fraction(1, 4 if s else 2)
                    mass = draw['exact']['history_mass']
                    self.assertEqual(mass['coefficients'], [1, 0, 0, 0])
                    self.assertEqual(Fraction(2)**mass['power'], expected_mass)
                    accepted = int((1 ^ flip0) == reported_copy)
                    self.assertEqual(draw['acceptance'], accepted)
                    self.assertEqual(draw['error'], accepted*b)
                    found.add((a, b))
                expected = {(a, b ^ flip3) for a in (0, 1)
                            for b in range(2 if (1 ^ flip0 ^ a) else 1)}
                self.assertEqual(found, expected)

    def registry(self):
        registry = ControllerRegistry()
        registry.register('and', lambda bits, config: (bits[0] & bits[1],), version='1')
        return registry

    def parents(self, compiled):
        return [row for rows in compiled.inspect()['compilation']['query_families'].values()
                for row in rows]

    def test_three_bit_selection_order(self):
        physical = parse_circuit('H 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                                 'CONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                                 'CX ctrl[0] 2\nM 2\nDETECTOR rec[-1]\nPOSTSELECT det[2]',
                                 controllers=self.registry())
        task = {'outputs': [0], 'sample_records': [0, 1]}
        target = parse_circuit('I 0')
        compiled = compile_circuit(physical, task=task, ideal_state=target)
        self.assertEqual({p['purpose'] for p in self.parents(compiled)},
                         {'fidelity', 'acceptance', 'sampling_capped'})
        before = compiled.inspect()['compilation']
        rejected = 0
        for draw in hooks.samples(compiled, Random(20260928), 128):
            a, b = draw['sampled_records']
            self.assertEqual(draw['acceptance'], 1-(a & b))
            self.assertEqual(draw['error'], a*(1-(a & b)))
            rejected += not draw['acceptance']
        self.assertGreater(rejected, 0)
        self.assertEqual(before, compiled.inspect()['compilation'])
        state = compiled._execution.state()
        self.assertLessEqual(state['persistent_networks'], 3)
        self.assertEqual(state['reference_network_resource_constructions'], 0)
        self.assertEqual(state['reference_network_coordinate_constructions'], 0)
        self.assertEqual(state['reference_network_evaluations']['fidelity'], 128-rejected)
        self.assertEqual(state['reference_network_evaluations']['acceptance'], 128)
        exact = enumerate_errors(physical, task={k: v for k, v in task.items() if k != 'sample_records'},
                                 ideal_state=target, max_weight=0)
        self.assertEqual(exact['status'], 'complete', exact)
        self.assertEqual(exact['totals']['acceptance'], 3/4)
        self.assertEqual(exact['totals']['error'], 1/4)
        self.assertAlmostEqual(1-exact['totals']['error']/exact['totals']['acceptance'], 2/3)

    def test_affine_support_offset_and_report_flip(self):
        physical = parse_circuit('X 0\nH 1\nM(1) 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                                 'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nCX ctrl[0] 2',
                                 controllers=self.registry())
        compiled = compile_circuit(physical, task={'outputs': [2], 'sample_records': [0, 1]},
                                   ideal_state=parse_circuit('I 0'))
        for draw in hooks.samples(compiled, Random(17), 16):
            self.assertEqual(draw['sampled_records'][0], 0)
            self.assertEqual(draw['acceptance'], 1)
            self.assertEqual(draw['error'], 0)

    def test_number_of_parents_does_not_follow_prefix_count(self):
        for layers in (1, 3, 7):
            lines = []
            for j in range(layers):
                a, b = 2*j+1, 2*j+2
                lines.extend([f'H {a} {b}', f'M {a} {b}', 'DETECTOR rec[-2]', 'DETECTOR rec[-1]',
                              f'CONTROL[and] det[-2] det[-1] -> ctrl[{j}]', f'CX ctrl[{j}] 0'])
            compiled = compile_circuit(parse_circuit('\n'.join(lines), controllers=self.registry()),
                                       task={'outputs': [0]}, ideal_state=parse_circuit('I 0'))
            self.assertEqual(len(self.parents(compiled)), 3)
            list(hooks.samples(compiled, Random(21), 8))
            self.assertEqual(len(self.parents(compiled)), 3)
            self.assertLessEqual(compiled._execution.state()['persistent_networks'], 3)

    def test_affine_sector_enumerates_free_coordinates_not_redundant_labels(self):
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        lines = ['H 0']
        for j in range(12):
            lines.extend(['M 0', 'DETECTOR rec[-1]', f'CONTROL[echo] det[-1] -> ctrl[{j}]', f'CX ctrl[{j}] 1'])
        circuit = parse_circuit('\n'.join(lines), controllers=registry)
        result = enumerate_errors(circuit, task={'outputs': [1]}, ideal_state=parse_circuit('I 0'),
                                  max_weight=0, max_record_bits=1, max_branches=2)
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['totals']['acceptance'], 1)
        self.assertEqual(result['totals']['error'], 0)

    def test_affine_task_has_two_parents(self):
        compiled = compile_circuit(parse_circuit('H 0\nT 0\nH 1\nM 1\nCX rec[-1] 0\n'
                                                 'DETECTOR rec[-1]\nPOSTSELECT det[0]'),
                                   task={'outputs': [0]},
                                   ideal_state=parse_circuit('H 0\nT 0'))
        self.assertEqual({p['purpose'] for p in self.parents(compiled)}, {'fidelity', 'acceptance'})

    def test_all_three_nonzero_constrained_domains(self):
        from math import sqrt
        for sampler in ('affine_support', 'exact_marginal'):
            registry = ControllerRegistry()
            registry.register('echo', lambda bits, config: bits, version='1')
            physical = parse_circuit('H 0\nT 0\nMX 0\nDETECTOR rec[-1]\n'
                'CONTROL[echo] det[-1] -> ctrl[0]\nCZ ctrl[0] 0\nMX 0', controllers=registry)
            compiled = compile_circuit(physical, task={'outputs': [0], 'sample_records': [0, 1]},
                ideal_state=parse_circuit('H 0\nT 0'), options={'history_sampler': sampler})
            parents = self.parents(compiled)
            self.assertEqual(sorted(p['prepared_update_coordinates'] for p in parents), [2, 2, 4])
            self.assertEqual(sorted(p['prepared_update_entries'] for p in parents), [2, 2, 4])
            self.assertTrue(all(p['fixed_update_domain'] and p['entry_change_groups'] == 1 for p in parents))
            for seed in range(16):
                draw = compiled.sample(Random(seed))
                a, b = draw['sampled_records']
                self.assertEqual(b, 0)
                # Direct one-qubit oracle: X measurement of T|+>, followed
                # by its reported Z correction, ends in |+> for either branch.
                weight = 1+(1-2*a)/sqrt(2) if sampler == 'affine_support' else 1
                self.assertAlmostEqual(draw['acceptance'], weight)
                self.assertAlmostEqual(draw['error'], weight*(1-1/sqrt(2))/2)
            state = compiled._execution.state()
            self.assertEqual(state['reference_network_coordinate_constructions'], 0)
            self.assertEqual(state['reference_network_resource_constructions'], 0)
            self.assertEqual(state['branch_original_locations'], 0)

    def test_affine_batch_reuses_protocol_assignment_without_sampling_parent(self):
        compiled = compile_circuit(parse_circuit('H 0\nT 0\nZ_ERROR(0.25) 0'),
                                   task={'outputs': [0]}, ideal_state=parse_circuit('H 0\nT 0'))
        self.assertEqual({p['purpose'] for p in self.parents(compiled)}, {'fidelity', 'acceptance'})
        batch = list(hooks.raw_samples(compiled, Random(9), 16))
        self.assertEqual(compiled._execution.state()['target_frame_reuses'], 16)
        state = compiled._execution.state()
        self.assertEqual(state['branch_original_locations'], 0)
        self.assertGreater(state['protocol_transport_arithmetic_payload_bytes'], 0)
        self.assertGreater(state['branch_transport_arithmetic_payload_bytes'], 0)
        # Single-draw and batch entry points preserve the exact random stream.
        other = compile_circuit(parse_circuit('H 0\nT 0\nZ_ERROR(0.25) 0'),
                                task={'outputs': [0]}, ideal_state=parse_circuit('H 0\nT 0'))
        rng = Random(9)
        singles = [other._execution.sample(rng) for _ in range(16)]
        self.assertEqual(batch, singles)

    def test_cancelled_external_symbols_do_not_require_third_parent(self):
        circuit = parse_circuit('H 1 2\nM 1 2\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                                'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\n'
                                'CONTROL[XOR] ctrl[0] ctrl[1] -> ctrl[2]\nCX ctrl[2] 0',
                                controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [0]}, ideal_state=parse_circuit('I 0'))
        self.assertEqual(compiled.inspect()['route'], 'linear')
        self.assertEqual(len(self.parents(compiled)), 2)
        self.assertEqual(compiled.sample(Random(0))['decoder_calls'], 0)


if __name__ == '__main__':
    unittest.main()
