"""Independent finite instruments for causal conditional Pauli assignments.

The oracle sums true outcomes, report flips and mutually exclusive physical
channels directly, using two-entry complex output vectors. It shares no
compiler, marginal, transport or resource code with the simulator.
"""
from tests.support.readout_target import ZERO_TARGET
from fractions import Fraction
from itertools import product
from random import Random
import math
import unittest

import numpy as np

from magna import ControllerRegistry, compile_circuit, enumerate_errors, parse_circuit
from magna.file_api import NoiseDraw
from test_low_weight import rational
from tests.support import file_hooks as hooks

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1])
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j*math.pi/4)])


def dense_fixture(mode, feedback):
    registry = ControllerRegistry()
    registry.register('and', lambda bits, config: (bits[0] & bits[1],), version='1')
    if mode == 'affine':
        control = 'CONTROL[XOR] rec[-2] rec[-1] -> ctrl[0]'
        guard = lambda a, b: a ^ b
    elif mode == 'external':
        control = 'DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]'
        guard = lambda a, b: a & b
    else:
        control = ('DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[1]\n'
                   'CONTROL[XOR] ctrl[1] rec[-2] -> ctrl[0]')
        guard = lambda a, b: (a & b) ^ a
    text = ('H 0 1 2\nT 0 2\nH 0\nM(1/4) 0 1\n' + control + '\n'
            + feedback + ' ctrl[0] 2\n'
            'CORRELATED_ERROR(1/4) ctrl[0] Y2 X1\n'
            'ELSE_CORRELATED_ERROR(1/3) ctrl[0] Z2\n'
            'Z_ERROR(0) ctrl[0] 2\nZ_ERROR(1) ctrl[0] 2\n'
            'M(1/5) 1\nX_ERROR(1/3) rec[-1] 2\n'
            # Accept when records 0 and 2 agree, after any decoder-input detectors.
            f'DETECTOR rec[-3] rec[-1]\nPOSTSELECT det[{control.count("DETECTOR")}]')
    circuit = parse_circuit(text, controllers=registry)
    target = parse_circuit('H 0\nT 0')
    task = {'outputs': [2]}
    # Independently enumerate the complete physical instrument.
    control_state = H @ T @ H @ np.array([1, 0])
    ideal = T @ H @ np.array([1, 0])
    mass = acceptance = target_mass = 0.
    for a, b, ra, rb, rc in product(range(2), repeat=5):
        p = abs(control_state[a])**2 / 2
        for flip in (ra, rb):
            p *= 1/4 if flip else 3/4
        p *= 1/5 if rc else 4/5
        g = guard(a ^ ra, b ^ rb)
        branches = ((I, 0, 1.),) if not g else ((I, 0, .5), (Y, 1, .25), (Z, 0, .25))
        for pauli, flip_b, channel_p in branches:
            v = ideal.copy()
            if g:
                v = (X if feedback == 'CX' else S) @ v
            v = pauli @ v
            if g:
                v = Z @ v  # The probability-one repeated guard.
            c = b ^ flip_b ^ rc
            for last, last_p in (((I, 1.),) if not c else ((I, 2/3), (X, 1/3))):
                weight = p * channel_p * last_p
                mass += weight
                if (a ^ ra) == c:
                    acceptance += weight
                    target_mass += weight * abs(np.vdot(ideal, last @ v))**2
    return circuit, task, target, (mass, acceptance, acceptance-target_mass)


class ConditionalNoiseTests(unittest.TestCase):
    def parents(self, compiled):
        return [p for rows in compiled.inspect()['compilation']['query_families'].values() for p in rows]

    def test_exact_dense_instruments_and_sampling(self):
        for mode, feedback in product(('affine', 'external', 'mixed'), ('CX', 'CS')):
            with self.subTest(mode=mode, feedback=feedback):
                circuit, task, target, expected = dense_fixture(mode, feedback)
                exact = enumerate_errors(circuit, task=task, ideal_state=target, max_weight=7)
                self.assertEqual(exact['status'], 'complete', exact)
                self.assertAlmostEqual(expected[0], 1)
                self.assertEqual(exact['totals']['covered_probability'], 1)
                self.assertAlmostEqual(exact['totals']['acceptance'], expected[1], places=12)
                self.assertAlmostEqual(exact['totals']['error'], expected[2], places=12)
                compiled = compile_circuit(circuit, task=task, ideal_state=target)
                initial = compiled.inspect()['compilation']
                before = compiled.compilations
                draws = list(hooks.samples(compiled, Random(291), 512))
                self.assertAlmostEqual(sum(d['acceptance'] for d in draws)/len(draws), expected[1], delta=.065)
                self.assertAlmostEqual(sum(d['error'] for d in draws)/len(draws), expected[2], delta=.065)
                self.assertEqual(compiled.compilations, before)
                self.assertEqual(compiled.inspect()['compilation'], initial)
                state = compiled._execution.state()
                self.assertTrue(state['conditional_noise'])
                self.assertTrue(state['sampled_history'])
                self.assertEqual(len(self.parents(compiled)), 3)
                self.assertEqual(state['reference_network_preparations'], 3)
                self.assertLessEqual(state['persistent_networks'], 3)
                self.assertEqual(state['reference_network_resource_constructions'], 0)
                self.assertEqual(state['reference_network_coordinate_constructions'], 0)
                nonzero = sum(any(d['exact']['acceptance']['coefficients']) for d in draws)
                self.assertEqual(state['reference_network_evaluations']['fidelity'], nonzero)
                self.assertEqual(state['reference_network_evaluations']['acceptance'], 512)

    def test_supplied_candidate_respects_guard_and_causal_chain(self):
        calls = []
        registry = ControllerRegistry()
        def echo(bits, config):
            calls.append(bits[0])
            return bits
        registry.register('echo', echo, version='1')
        circuit = parse_circuit('H 0\nM 0\nX_ERROR(1/3) rec[-1] 1\nM 1\n'
                                'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nX_ERROR(1/2) ctrl[0] 2',
                                controllers=registry)
        ids = [i.op_id for i in circuit.instructions if i.name == 'X_ERROR']
        noise = NoiseDraw({oid: ({'kind': 'X', 'qubits': [q]},) for oid, q in zip(ids, (1, 2))}, ())
        compiled = compile_circuit(circuit, task={'outputs': [2], 'sample_records': [0, 1]},
                                   ideal_state=parse_circuit('I 0'))
        observed = set()
        for seed in range(32):
            draw = compiled.sample(Random(seed), noise=noise)
            a, b = draw['sampled_records']
            observed.add(a)
            self.assertEqual(a, b)
            self.assertEqual(calls[-1], b)
            self.assertEqual(draw['decoder_calls'], 1)
            self.assertEqual(draw['acceptance'], 1)
            self.assertEqual(draw['error'], a)
            self.assertEqual(draw['exact']['history_mass'], {'coefficients': [1, 0, 0, 0], 'power': -1})
        self.assertEqual(observed, {0, 1})
        exact = enumerate_errors(circuit, task={'outputs': [2]}, ideal_state=parse_circuit('I 0'), max_weight=2)
        self.assertEqual(rational(exact['totals']['exact']['error']), Fraction(1, 12))
        truncated = enumerate_errors(circuit, task={'outputs': [2]}, ideal_state=parse_circuit('I 0'), max_weight=1)
        self.assertEqual(truncated['totals']['covered_probability'], 5/6)
        self.assertEqual(truncated['totals']['error'], 0)

    def test_noncommuting_guard_chain_and_target_route(self):
        circuit = parse_circuit('H 0\nT 0\nH 0\nM 0\nX_ERROR(1/3) rec[-1] 1\nM 1\n'
                                'X_ERROR(1/2) rec[-1] 2\nM 2')
        task = {'outputs': [2], 'ideal_state': ZERO_TARGET}
        exact = enumerate_errors(circuit, task=task, max_weight=2)
        self.assertEqual(exact['status'], 'complete', exact)
        self.assertAlmostEqual(exact['totals']['acceptance'], 1)
        self.assertAlmostEqual(exact['totals']['error'], (2-math.sqrt(2))/24)
        for method, purpose in [('affine_support', 'sampling_capped'), ('exact_marginal', 'sampling')]:
            compiled = compile_circuit(circuit, task=task, options={'history_sampler': method})
            self.assertEqual({p['purpose'] for p in self.parents(compiled)}, {'acceptance', 'fidelity', purpose})
            list(hooks.samples(compiled, Random(13), 16))
            self.assertEqual(compiled._execution.state()['reference_network_preparations'], 3)

    def test_importance_weights_only_tilt_unconditional_channels(self):
        circuit = parse_circuit('H 0\nM 0\nX_ERROR(1/3) rec[-1] 1\nZ_ERROR(1/10) 1')
        compiled = compile_circuit(circuit, task={'outputs': [1]}, ideal_state=parse_circuit('I 0'),
                                   sampling={'method': 'importance', 'proposal_scale': '5'})
        self.assertEqual(compiled.noise_schedule.guarded, 1)
        self.assertEqual(compiled.noise_schedule.tilted, 1)
        ids = [i.op_id for i in circuit.instructions if i.name.endswith('_ERROR')]
        for seed in range(16):
            candidate = NoiseDraw({ids[0]: ({'kind': 'X', 'qubits': [1]},)}, (), (9, 5))
            draw = compiled.sample(Random(seed), noise=candidate)
            self.assertEqual(draw['exact']['noise_likelihood_ratio_hex'], ['0x9', '0x5'])

    def test_parity_guard_does_not_require_all_raw_records(self):
        circuit = parse_circuit('H 0 1\nM 0 1\nCONTROL[XOR] rec[-2] rec[-1] -> ctrl[0]\n'
                                'X_ERROR(1) ctrl[0] 2\nZ_ERROR(1) ctrl[0] 2')
        compiled = compile_circuit(circuit, task={'outputs': [2]}, ideal_state=parse_circuit('I 0'))
        for draw in hooks.samples(compiled, Random(1), 16):
            self.assertEqual(draw['sampled_parities'], 1)
            self.assertEqual(draw['sampled_records'], [])
        self.assertEqual(len(self.parents(compiled)), 3)

    def test_declared_constant_guards_and_channel_endpoints(self):
        for p, constant in product((0, 1), (0, 1)):
            text = ('M 0\nCONTROL[XOR] rec[-1] rec[-1] -> ctrl[0]\n'
                    + ('CONTROL[NOT] ctrl[0] -> ctrl[1]\n' if constant else '')
                    + f'X_ERROR({p}) ctrl[{constant}] 1')
            compiled = compile_circuit(parse_circuit(text), task={'outputs': [1]}, ideal_state=parse_circuit('I 0'))
            self.assertEqual(len(self.parents(compiled)), 3)
            self.assertEqual(compiled.sample(Random(0))['error'], p*constant)

    def test_two_qubit_channel_order_and_mutual_exclusion(self):
        # Stim ordering IX,IY,IZ,XI,XX,...; XY has probability 1/3, YZ 1/6.
        probabilities = ['0']*15
        probabilities[5], probabilities[10] = '1/3', '1/6'
        circuit = parse_circuit('H 0\nM 0\nPAULI_CHANNEL_2(' + ','.join(probabilities) + ') rec[-1] 1 2')
        exact = enumerate_errors(circuit, task={'outputs': [1, 2]}, ideal_state=parse_circuit('I 0 1'), max_weight=1)
        self.assertEqual(exact['status'], 'complete', exact)
        self.assertEqual(rational(exact['totals']['exact']['error']), Fraction(1, 4))

    def test_shared_t_tdag_noise_has_no_guarded_route_or_virtual_noise(self):
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        for control in ('rec[-1]', 'ctrl[0]'):
            middle = 'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n' if control == 'ctrl[0]' else ''
            circuit = parse_circuit('H 0 1\nM 0\n'+middle+'T 1\nCS_DAG '+control+' 1\nZ_ERROR(1/4) 1', controllers=registry)
            compiled = compile_circuit(circuit, task={'outputs': [1]}, ideal_state=parse_circuit('H 0\nT 0'))
            self.assertFalse(compiled._execution.state()['conditional_noise'])
            self.assertEqual(compiled.noise_schedule.category_count, 1)
            self.assertEqual(len(self.parents(compiled)), 2 if control == 'rec[-1]' else 3)
            exact = enumerate_errors(circuit, task={'outputs': [1]}, ideal_state=parse_circuit('H 0\nT 0'), max_weight=1)
            self.assertAlmostEqual(exact['totals']['error'], 3/8)


if __name__ == '__main__':
    unittest.main()
