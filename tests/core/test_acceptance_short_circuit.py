"""Zero accepted mass skips outputs without changing histories or exact metadata."""
from random import Random
import unittest

from magna import ControllerRegistry, compile_circuit, enumerate_errors, parse_circuit
from magna.scalar import ExactScalar, ONE


ZERO = ExactScalar()


class AcceptanceShortCircuitTests(unittest.TestCase):
    def test_zero_outputs_preserve_optional_error_and_target_normalization(self):
        physical = parse_circuit('I 0\nM 1\nDETECTOR rec[-1]\nPOSTSELECT !det[0]')
        target = dict(circuit=parse_circuit('I 0\nH 1'), outputs=[0], postselection={1: 0})
        task = {'outputs': [0], 'observables': [[0, 1], [1, 0], [0, 0]]}
        for ideal in (None, target):
            with self.subTest(target=ideal is not None):
                compiled = compile_circuit(physical, task=task, ideal_state=ideal)
                before = compiled._execution.state()['queries']
                masses = compiled._execution.sample(Random(4))['masses']
                self.assertEqual(ExactScalar(*masses['history']), ONE)
                self.assertEqual(ExactScalar(*masses['acceptance']), ZERO)
                self.assertEqual([ExactScalar(*v) for v in masses['moments']], [ZERO]*3)
                for key in ('error', 'target'):
                    if ideal is None:
                        self.assertIsNone(masses[key])
                    else:
                        self.assertEqual(ExactScalar(*masses[key]), ZERO)
                self.assertEqual(ExactScalar(*masses['target_normalization']),
                                 ONE if ideal is None else ExactScalar((1, 0, 0, 0), -1))
                state = compiled._execution.state()
                self.assertEqual(state['queries']-before, 1)
                self.assertEqual(state['reference_network_evaluations'].get('fidelity', 0), 0)
                self.assertEqual(state['target_frame_reuses'], 0)
                totals = enumerate_errors(physical, task=task, ideal_state=ideal, max_weight=0)['totals']
                self.assertEqual(totals['acceptance'], 0)
                self.assertEqual(totals['moments'], [0]*3)
                self.assertEqual(totals['error'], None if ideal is None else 0)

    def test_fractional_acceptance_still_queries_every_output(self):
        physical = parse_circuit('I 0\nH 1\nM 1\nDETECTOR rec[-1]\nPOSTSELECT det[0]')
        target = dict(circuit=parse_circuit('I 0\nH 1'), outputs=[0], postselection={1: 0})
        compiled = compile_circuit(physical,
            task={'outputs': [0], 'observables': [[0, 1], [1, 0], [0, 0]]}, ideal_state=target)
        before = compiled._execution.state()['queries']
        draw = compiled.sample(Random(4))
        self.assertEqual((draw['acceptance'], draw['error'], draw['moments']), (.5, 0, [.5, 0, .5]))
        state = compiled._execution.state()
        self.assertEqual(state['queries']-before, 5)
        self.assertEqual(state['reference_network_evaluations']['fidelity'], 1)

    def test_rejection_preserves_rng_and_both_decoder_calls(self):
        for sampler in ('affine_support', 'exact_marginal'):
            with self.subTest(sampler=sampler):
                traces = [[], []]
                compiled = []
                for selected in (False, True):
                    registry = ControllerRegistry()
                    def echo(bits, config, seen=traces[int(selected)]):
                        seen.append(bits)
                        return bits
                    registry.register('echo', echo, version='1')
                    circuit = 'H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[0] -> ctrl[0]\nCX ctrl[0] 1'
                    if selected:
                        circuit += '\nPOSTSELECT det[0]'
                    recovery = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[0] -> ctrl[0]\nCX ctrl[0] 0',
                                             controllers=registry)
                    compiled.append(compile_circuit(parse_circuit(circuit, controllers=registry),
                        recovery=recovery, ideal_state=parse_circuit('I 0'),
                        task={'outputs': [1], 'sample_records': [0], 'observables': [[0, 1], [1, 0]]},
                        options={'history_sampler': sampler}))
                rngs = [Random(19), Random(19)]
                rejected = 0
                for _ in range(32):
                    reference, draw = [c.sample(rng) for c, rng in zip(compiled, rngs)]
                    bit, = draw['sampled_records']
                    self.assertEqual(draw['sampled_records'], reference['sampled_records'])
                    self.assertEqual(draw['exact']['history_mass'], reference['exact']['history_mass'])
                    self.assertEqual((draw['decoder_calls'], draw['recovery_decoder_calls']), (1, 1))
                    self.assertEqual((draw['acceptance'], draw['error'], draw['moments']),
                                     (1-bit, 0, [1-bit, 0]))
                    self.assertEqual(rngs[0].getstate(), rngs[1].getstate())
                    rejected += bit
                self.assertEqual(traces[0], traces[1])
                self.assertGreater(rejected, 0)
                self.assertLess(rejected, 32)
                state = compiled[1]._execution.state()
                self.assertEqual(state['reference_network_evaluations']['acceptance'], 32)
                self.assertEqual(state['reference_network_evaluations']['fidelity'], 32-rejected)

    def test_small_nonzero_acceptance_is_not_rounded_to_zero(self):
        qubits = ' '.join(map(str, range(1, 65)))
        circuit = f'I 0\nH {qubits}\nM {qubits}\n'
        circuit += ''.join(f'DETECTOR rec[{-j}]\nPOSTSELECT det[{j-1}]\n' for j in range(1, 65))
        compiled = compile_circuit(parse_circuit(circuit), ideal_state=parse_circuit('H 0'),
            task={'outputs': [0], 'observables': [[0, 1]]})
        draw = compiled.sample(Random(0))
        self.assertEqual(ExactScalar(**draw['exact']['acceptance']), ExactScalar((1, 0, 0, 0), -64))
        self.assertEqual(ExactScalar(**draw['exact']['error']), ExactScalar((1, 0, 0, 0), -65))
        self.assertEqual(draw['moments'], [2**-64])
        self.assertEqual(compiled._execution.state()['reference_network_evaluations']['fidelity'], 1)

    def test_interference_zero_histories_keep_zero_weight_draws(self):
        physical = parse_circuit('H 0 1 2 3\nCCZ 0 1 2\nCCZ 0 1 3\nCCZ 0 2 3\n'
                                 'H 0 1 2 3\nM 0 1 2 3\nI 4')
        compiled = compile_circuit(physical, ideal_state=parse_circuit('I 0'),
            task={'outputs': [4], 'sample_records': [0, 1, 2, 3], 'observables': [[0, 1]]})
        zeros = 0
        for seed in range(64):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw['error'], 0)
            self.assertEqual(draw['moments'], [draw['acceptance']])
            if ExactScalar(**draw['exact']['history_mass']) == ZERO:
                zeros += 1
                self.assertEqual(draw['acceptance'], 0)
        self.assertGreater(zeros, 0)
        state = compiled._execution.state()
        self.assertEqual(state['reference_network_evaluations']['acceptance'], 64)
        self.assertEqual(state['reference_network_evaluations']['fidelity'], 64-zeros)

    def test_operation_boundary_omits_zero_moment_queries(self):
        from magna.operations import BoundarySampler, BoundaryTask, InstrumentIR, StabilizerKet
        ir = InstrumentIR((0, 1), (0,),
            (dict(id='m', kind='MZ_DESTRUCT', qubits=(1,), record='a'),), StabilizerKet.positive(2))
        sampler = BoundarySampler(ir, BoundaryTask((('a',),), (1,), ((0, 1), (1, 0))))
        draw = sampler.sample(Random(4))
        self.assertEqual(draw.acceptance, ZERO)
        self.assertEqual(draw.moments, (ZERO, ZERO))
        self.assertIsNone(draw.error)
        self.assertEqual(len(draw.diagnostics['final_resources']), 1)


if __name__ == '__main__':
    unittest.main()
