"""Three-file output metrics, with independent finite state/channel oracles."""
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
from random import Random
import tempfile
import unittest

import numpy as np

from magna import ControllerRegistry, compile_circuit, parse_circuit, run
from magna.file_api import main


class TerminalEvaluationTests(unittest.TestCase):
    def evaluate(self, physical, target, recovery=None, **kwargs):
        result = run(parse_circuit(physical), ideal_state=parse_circuit(target),
                     recovery=parse_circuit(recovery) if recovery is not None else None, **kwargs)
        self.assertEqual(result['status'], 'complete', result)
        return result

    def test_t_targets_and_empty_recovery(self):
        for suffix, expected in (('', 0), ('\nZ 0', 1), ('\nS 0', .5)):
            for recovery in (None, '', 'I 0'):
                with self.subTest(suffix=suffix, recovery=recovery):
                    result = self.evaluate('H 0\nT 0'+suffix, 'H 0\nT 0', recovery)
                    self.assertAlmostEqual(result['estimates']['conditional_error'], expected)
                    self.assertEqual(result['estimates']['acceptance'], 1)

    def test_target_inverse_against_independent_dense_amplitudes(self):
        h = np.array([[1, 1], [1, -1]], dtype=complex)/math.sqrt(2)
        t = np.diag([1, np.exp(1j*math.pi/4)])
        y = np.array([[0, -1j], [1j, 0]])
        # Qubit 0 is the low bit. Use entangling magic targets, not only U/U† cancellation.
        cx = np.eye(4, dtype=complex); cx[[1, 3]] = cx[[3, 1]]
        ch = np.eye(4, dtype=complex); ch[np.ix_([1, 3], [1, 3])] = h
        cs = np.diag([1, 1, 1, 1j])
        unitary = ch @ cs @ cx @ np.kron(t @ h, y @ t @ h)
        target = 'H 0 1\nT 0 1\nY 0\nCX 0 1\nCS 0 1\nCH 0 1'
        for word in range(4):
            physical = 'I 0 1\n'+'\n'.join('X '+str(q) for q in range(2) if word >> q & 1)
            result = self.evaluate(physical, target)
            self.assertAlmostEqual(result['estimates']['error'], 1-abs(unitary[word, 0])**2)

    def test_wide_clifford_encoded_t_target_uses_compact_native_boundary(self):
        for width in (3, 65):
            target = 'H 0\nT 0\n'+'\n'.join(f'CX 0 {q}' for q in range(1, width))
            result = self.evaluate(target+'\nX_ERROR(1) 1', target,
                                   'MPP Z0*Z1\nCX rec[-1] 1', options={'max_quotient_rank': 2})
            self.assertEqual(result['estimates']['error'], 0)
            self.assertEqual(result['evaluation']['ideal_state']['width'], width)

    def test_output_mapping_and_unobserved_environment(self):
        result = self.evaluate('X 2', 'X 0\nI 1', task={'outputs': [2, 0]})
        self.assertEqual(result['estimates']['error'], 0)
        result = self.evaluate('X 2', 'X 0\nI 1', task={'outputs': [0, 2]})
        self.assertEqual(result['estimates']['error'], 1)
        result = self.evaluate('H 0\nCX 0 1', 'I 0', task={'outputs': [0]})
        self.assertEqual(result['estimates']['error'], .5)
        # Extra recovery ancilla is traced, not projected or coherently summed.
        result = self.evaluate('H 2', 'H 0', 'CX 0 1', task={'outputs': [2]})
        self.assertEqual(result['estimates']['error'], .5)
        self.assertEqual(result['evaluation']['recovery']['ancillas'], 1)

    def test_repetition_recovery_preserves_encoded_magic_and_original_faults(self):
        registry = ControllerRegistry()
        def decode(bits, config):
            return {(0, 0): (0, 0, 0), (1, 0): (1, 0, 0),
                    (1, 1): (0, 1, 0), (0, 1): (0, 0, 1)}[bits]
        registry.register('decode', decode, version='repetition-v1')
        recovery = parse_circuit('MPP Z0*Z1 Z1*Z2\n'
            'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
            'CONTROL[decode] det[-2] det[-1] -> ctrl[0] ctrl[1] ctrl[2]\n'
            'CX ctrl[0] 0\nCX ctrl[1] 1\nCX ctrl[2] 2', controllers=registry)
        target = parse_circuit('H 0\nT 0\nCX 0 1\nCX 0 2')
        for faults, expected in (('', 0), ('X_ERROR(1) 0', 0), ('X_ERROR(1) 1', 0),
                                 ('X_ERROR(1) 2', 0), ('X_ERROR(1) 0 1', .5)):
            physical = parse_circuit(target.text+'\n'+faults)
            result = run(physical, ideal_state=target, recovery=recovery, shots=2)
            self.assertEqual(result['status'], 'complete', result)
            self.assertAlmostEqual(result['estimates']['conditional_error'], expected)
            if faults:
                raw = run(physical, ideal_state=target)
                self.assertEqual(raw['estimates']['error'], 1)

    def test_terminal_namespace_report_errors_and_external_decoders(self):
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        physical = parse_circuit('H 0\nM(1) 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
                                 'CX ctrl[0] 1', controllers=registry)
        recovery = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0',
                                 controllers=registry)
        compiled = compile_circuit(physical, task={'outputs': [1]},
            ideal_state=parse_circuit('I 0'), recovery=recovery)
        for seed in range(8):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw['decoder_calls'], 1)
            self.assertEqual(draw['recovery_decoder_calls'], 1)
            self.assertEqual(draw['error'], 0)

    def test_recovery_samples_joint_syndromes_without_acceptance_or_target_bias(self):
        registry = ControllerRegistry(); observed = []
        def echo(bits, config):
            observed.append(bits[0]); return bits
        registry.register('echo', echo, version='1')
        physical = parse_circuit('H 0\nCX 0 1\nM 1\nDETECTOR rec[-1]\nPOSTSELECT !det[0]')
        recovery = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0',
                                 controllers=registry)
        a = compile_circuit(physical, task={'outputs': [0]},
                            recovery=recovery, ideal_state=parse_circuit('I 0'))
        b = compile_circuit(physical, task={'outputs': [0]},
                            recovery=recovery, ideal_state=parse_circuit('X 0'))
        for seed in range(12):
            da = a.sample(Random(seed)); sa = observed[-1]
            db = b.sample(Random(seed)); sb = observed[-1]
            self.assertEqual(sa, sb)
            self.assertEqual(da['acceptance'], sa)
            self.assertEqual(db['acceptance'], sa)
            self.assertEqual(da['error'], 0)
            self.assertEqual(db['error'], sa)
        self.assertEqual(set(observed), {0, 1})

    def test_affine_recovery_integrates_records_and_moments_before_target(self):
        compiled = compile_circuit(parse_circuit('H 0'), ideal_state=parse_circuit('I 0'),
            recovery=parse_circuit('M 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n'
                                   'CONTROL[XOR] rec[-1] -> ctrl[0]\nCX ctrl[0] 0'),
            task={'observables': [[0, 1]]})
        draw = compiled.sample(Random(1))
        self.assertEqual(draw['sampled_parities'], 0)
        self.assertEqual(draw['error'], 0)
        self.assertEqual(draw['moments'], [1])
        result = self.evaluate('H 0', 'X 0', 'R 0')
        self.assertEqual(result['estimates']['error'], 1)

    def test_nonclifford_terminal_feedback_and_multiple_calls(self):
        registry = ControllerRegistry(); calls = []
        def echo(bits, config):
            calls.append(bits[0]); return bits
        registry.register('echo', echo, version='1')
        recovery = parse_circuit('H 1\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CS ctrl[0] 0\nCONTROL[echo] ctrl[0] -> ctrl[1]\nCH ctrl[1] 0', controllers=registry)
        compiled = compile_circuit(parse_circuit('H 0\nT 0'), ideal_state=parse_circuit('H 0\nT 0'), recovery=recovery)
        tstate = np.array([1, np.exp(1j*math.pi/4)])/math.sqrt(2)
        h = np.array([[1, 1], [1, -1]])/math.sqrt(2)
        changed = 1-abs(np.vdot(tstate, h @ np.diag([1, 1j]) @ tstate))**2
        for seed in range(8):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw['sampled_parities'], 1)
            self.assertEqual(draw['recovery_decoder_calls'], 2)
            self.assertEqual(calls[-1], calls[-2])
            self.assertAlmostEqual(draw['error'], changed if calls[-1] else 0)

    def test_module_validation_and_mutually_exclusive_metrics(self):
        physical = parse_circuit('I 0')
        for text in ('M 0', 'R 0', 'X_ERROR(0) 0', 'H 1', 'CONTROL[XOR](1) -> ctrl[0]'):
            with self.subTest(target=text), self.assertRaises(ValueError):
                compile_circuit(physical, ideal_state=parse_circuit(text))
        for text in ('X_ERROR(0) 0', 'M(1/2) 0'):
            with self.subTest(recovery=text), self.assertRaises(ValueError):
                compile_circuit(physical, ideal_state=physical, recovery=parse_circuit(text))
        with self.assertRaises(ValueError):
            compile_circuit(physical, recovery=parse_circuit(''))
        for task, message in (({'error_checks': []}, "task field 'error_checks' is no longer supported"),
                              ({'error_effect': {}}, 'unknown task fields.*error_effect'),
                              ({'error_observables': []}, 'error_observables asks a classical question')):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, message):
                compile_circuit(physical, ideal_state=physical, task=task)
        failed = run(physical, ideal_state=physical, options={'max_queries': 1})
        self.assertEqual(failed['status'], 'failed')
        self.assertIsNone(failed['estimates'])

    def test_three_file_cli_task_relative_paths_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'physical.magna').write_text('X 0')
            (root/'target.magna').write_text('I 0')
            (root/'recovery.magna').write_text('M 0\nCX rec[-1] 0')
            task = root/'task.json'
            task.write_text(json.dumps({'ideal_state': 'target.magna', 'recovery': 'recovery.magna'}))
            for command in ('validate', 'inspect', 'run'):
                output = StringIO()
                with redirect_stdout(output):
                    code = main([command, str(root/'physical.magna'), '--task', str(task), '--json'])
                self.assertEqual(code, 0, output.getvalue())
                data = json.loads(output.getvalue())
                self.assertIsNotNone(data['evaluation']['ideal_state'])
                if command == 'run': self.assertEqual(data['estimates']['error'], 0)
            before = run(root/'physical.magna', task=task)
            (root/'target.magna').write_text('X 0')
            after = run(root/'physical.magna', task=task)
            self.assertNotEqual(before['fingerprints']['ideal_state'], after['fingerprints']['ideal_state'])
            self.assertEqual(after['estimates']['error'], 1)
            with self.assertRaises(ValueError):
                compile_circuit(root/'physical.magna', task=task, ideal_state=root/'target.magna')

    def test_fingerprints_do_not_depend_on_file_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for name in ('first', 'second'):
                root = Path(directory)/name
                root.mkdir()
                (root/'physical.magna').write_text('X 0')
                (root/'target.magna').write_text('I 0')
                (root/'recovery.magna').write_text('M 0\nCX rec[-1] 0')
                (root/'task.json').write_text(json.dumps({'ideal_state': 'target.magna', 'recovery': 'recovery.magna'}))
                results.append(run(root/'physical.magna', task=root/'task.json'))
            first, second = results
            self.assertNotEqual(first['evaluation']['ideal_state']['source'],
                                second['evaluation']['ideal_state']['source'])
            self.assertEqual(first['fingerprints'], second['fingerprints'])


if __name__ == '__main__':
    unittest.main()
