"""Native CUDA noise plans agree with original exact non-Clifford execution."""
import math
from random import Random
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy is required for the CUDA experiment')

from magna import _native, compile_circuit, parse_circuit
from magna._cuda_experiment import CudaExperiment
from magna.scalar import ExactScalar
from tests.support.cuda_reference import (cpu_frames, draw, evaluate_frames, evaluate_noise,
                                          explicit, validate_csr)


def real(raw):
    return float(ExactScalar(*raw).real_decimal())


class CsrValidationTests(unittest.TestCase):
    def test_empty_rows_and_repeated_actions_are_valid_for_the_xor_kernel(self):
        offsets, indices = validate_csr([0, 0, 2, 2], [3, 3], 4)
        self.assertEqual(offsets.dtype, np.dtype('uint64'))
        self.assertEqual(indices.dtype, np.dtype('uint32'))
        empty_offsets, empty_indices = validate_csr([0], [], 0)
        self.assertEqual((empty_offsets.size, empty_indices.size), (1, 0))

    def test_rejects_malformed_and_out_of_range_csr_before_launch(self):
        cases = [([], [], 1), ([1], [], 1), ([0, 2, 1], [0], 1),
                 ([0, 2], [0], 1), ([0, 1], [1], 1), ([0, 1], [-1], 1),
                 ([0, -1], [], 1), ([[0]], [], 1), ([0, 1], [[0]], 1),
                 ([0, 1], [2**32], 2**33)]
        for offsets, indices, actions in cases:
            with self.subTest(offsets=offsets, indices=indices):
                with self.assertRaises(ValueError):
                    validate_csr(offsets, indices, actions)
        for offsets, indices in [([0.0], []), ([0, 1], [True]), ([0, 1], [0.5])]:
            with self.assertRaises(TypeError):
                validate_csr(offsets, indices, 1)


class NativeCudaBridgeTests(unittest.TestCase):
    @staticmethod
    def magic_plan():
        return CudaExperiment(compile_circuit(
            parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
            ideal_state=parse_circuit('H 0\nT 0')))

    def test_exhaustive_paulis_preserve_true_t_target_infidelity(self):
        plan = self.magic_plan()
        self.assertEqual(plan.actions, 3)
        batch = explicit(plan, [[], [0], [1], [2]])
        original = evaluate_noise(plan, batch)
        frames = cpu_frames(plan, batch)
        self.assertEqual(evaluate_frames(plan, frames), original)
        self.assertEqual(sorted(real(row[2]) for row in original), [0.0, 0.5, 0.5, 1.0])
        self.assertEqual([real(row[1]) for row in original], [1.0] * 4)

    def test_irrational_acceptance_and_measurement_report_flips(self):
        plan = CudaExperiment(compile_circuit(parse_circuit(
            'H 0\nT 0\nZ_ERROR(1/3) 0\nH 0\nM(1/4) 0\n'
            'DETECTOR rec[-1]\nPOSTSELECT det[0]')))
        batch = explicit(plan, [[], [0], [1], [0, 1]])
        original = evaluate_noise(plan, batch)
        self.assertEqual(evaluate_frames(plan, cpu_frames(plan, batch)), original)
        expected = [0.5 - math.sqrt(2) / 4] * 2 + [0.5 + math.sqrt(2) / 4] * 2
        np.testing.assert_allclose(sorted(real(row[1]) for row in original), expected, atol=1e-15)

    def test_inverted_affine_feedback_preserves_output_infidelity(self):
        plan = CudaExperiment(compile_circuit(parse_circuit(
            'M 0\nCX !rec[-1] 1\nH 1\nT 1\nZ_ERROR(1/3) 1'),
            task={'outputs': [1]}, ideal_state=parse_circuit('H 0\nT 0')))
        batch = explicit(plan, [[], [0]])
        original = evaluate_noise(plan, batch)
        self.assertEqual(evaluate_frames(plan, cpu_frames(plan, batch)), original)
        self.assertEqual([real(row[2]) for row in original], [1.0, 0.0])

    def test_native_draws_partition_reproducibly_and_match_public_sampler(self):
        plan = self.magic_plan()
        rng = Random(981)
        seeds = [rng.getrandbits(64) for _ in range(79)]
        whole = draw(plan, seeds)
        left, right = draw(plan, seeds[:23]), draw(plan, seeds[23:])
        np.testing.assert_array_equal(whole.indices, np.concatenate([left.indices, right.indices]))
        np.testing.assert_array_equal(whole.offsets,
                                      np.concatenate([left.offsets, right.offsets[1:] + left.offsets[-1]]))
        self.assertEqual(evaluate_frames(plan, cpu_frames(plan, whole)), evaluate_noise(plan, whole))
        # With no history sampling, public sample consumes exactly one 64-bit
        # seed from this RNG per draw.  Check the actual public API, too.
        rng = Random(981)
        public = [plan.compiled.sample(rng)['exact'] for _ in seeds]
        exact = evaluate_noise(plan, whole)
        for row, sample in zip(exact, public):
            for value, name in zip(row, ('history_mass', 'acceptance', 'error')):
                self.assertEqual(value, (tuple(sample[name]['coefficients']), sample[name]['power']))

    def test_public_draws_match_named_noise_for_every_channel_kind(self):
        # Public draws transport each drawn action through images resolved once
        # per noise site; the reference names every action and packs it.
        # Explicit correlated actions exercise the named path inside a draw.
        pair = ', '.join(['1/40'] * 15)
        plan = CudaExperiment(compile_circuit(parse_circuit(
            'H 0\nT 0\nCX 0 1\nX_ERROR(1/5) 1\nY_ERROR(1/7) 0\nZ_ERROR(1/6) 2\n'
            'DEPOLARIZE1(1/3) 1\nPAULI_CHANNEL_1(1/10, 1/9, 1/8) 2\nH 2\nT 1\nCX 1 2\n'
            f'DEPOLARIZE2(1/2) 0 1 1 2\nPAULI_CHANNEL_2({pair}) 0 2\n'
            'CORRELATED_ERROR(1/4) X0 Z1\nELSE_CORRELATED_ERROR(1/3) Y2\nT 2\n'
            'M(1/4) 0\nCX rec[-1] 1'),
            task={'outputs': [1, 2]}, ideal_state=parse_circuit('H 0\nT 0\nCX 0 1\nH 1\nT 1')))
        rng = Random(20261028)
        seeds = [rng.getrandbits(64) for _ in range(160)]
        batch = draw(plan, seeds)
        self.assertGreater(batch.indices.size, 3 * len(seeds))
        rng = Random(20261028)
        public = [plan.compiled.sample(rng)['exact'] for _ in seeds]
        for row, sample in zip(evaluate_noise(plan, batch), public):
            for value, name in zip(row, ('history_mass', 'acceptance', 'error')):
                self.assertEqual(value, (tuple(sample[name]['coefficients']), sample[name]['power']))

    def test_zero_shots_and_ownership(self):
        plan = self.magic_plan()
        batch = draw(plan, [])
        frames = cpu_frames(plan, batch)
        self.assertEqual(frames.shape, (0, plan.width))
        self.assertEqual(evaluate_frames(plan, frames), ())
        self.assertEqual(evaluate_noise(plan, batch), ())
        other = self.magic_plan()
        with self.assertRaisesRegex(ValueError, 'different experiment'):
            cpu_frames(other, batch)
        with self.assertRaises(ValueError):
            evaluate_frames(plan, np.zeros((1, plan.width + 1), dtype=np.uint64))
        for seeds in ([-1], [True], [0.1]):
            with self.assertRaises((TypeError, ValueError)):
                draw(plan, seeds)

    def test_experiment_rejects_multiple_outcomes_at_one_site(self):
        plan = self.magic_plan()
        for rows in ([[0, 1]], [[0, 0]], [[1, 0]]):
            batch = explicit(plan, rows)
            for method in (cpu_frames, evaluate_noise):
                with self.assertRaisesRegex(ValueError, 'one outcome per site'):
                    method(plan, batch)
        # Row boundaries, including empty shots, do not constrain ordering.
        batch = explicit(plan, [[2], [], [0], [], [1]])
        self.assertEqual(evaluate_frames(plan, cpu_frames(plan, batch)), evaluate_noise(plan, batch))

    def test_recovery_and_postselected_target_normalization(self):
        plan = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
            recovery=parse_circuit('X 0'),
            ideal_state={'circuit': parse_circuit('H 0 1\nT 0'),
                         'outputs': [0], 'postselection': {1: 0}}))
        normalization = real(plan.metadata['target_normalization'])
        self.assertEqual(normalization, 0.5)
        batch = explicit(plan, [[], [0], [1], [2]])
        original = evaluate_noise(plan, batch)
        self.assertEqual(evaluate_frames(plan, cpu_frames(plan, batch)), original)
        self.assertEqual(sorted(real(row[2]) / normalization for row in original), [0.0, 0.5, 0.5, 1.0])

    def test_padding_and_stale_native_plan_are_rejected(self):
        plan = self.magic_plan()
        batch = explicit(plan, [[]])
        frames = cpu_frames(plan, batch).copy()
        bit = int(plan.metadata['frame_bits'])
        self.assertNotEqual(bit % 64, 0)
        frames[0, bit // 64] |= np.uint64(1) << np.uint64(bit % 64)
        with self.assertRaises(ValueError):
            evaluate_frames(plan, frames)
        options = dict(plan.compiled.options, max_quotient_rank=0)
        plan.compiled._execution.configure(options)
        with self.assertRaisesRegex(ValueError, 'reconfigured'):
            cpu_frames(plan, batch)

    def test_option_dictionary_edits_invalidate_the_fixed_experiment(self):
        plan = self.magic_plan()
        batch = explicit(plan, [[]])
        frames = cpu_frames(plan, batch)
        plan.compiled.options['max_quotient_rank'] = 0
        operations = [lambda: plan.validate(), lambda: draw(plan, []),
                      lambda: cpu_frames(plan, batch),
                      lambda: evaluate_frames(plan, frames), lambda: evaluate_noise(plan, batch)]
        for operation in operations:
            with self.assertRaisesRegex(ValueError, 'options changed'):
                operation()
        with self.assertRaisesRegex(ValueError, 'options changed'):
            CudaExperiment(plan.compiled)

    def test_rejects_history_conditional_and_importance_paths(self):
        cases = [
            (parse_circuit('H 0\nM 0'), {'task': {'sample_records': [0]}}),
            (parse_circuit('M 0\nX_ERROR(1/4) rec[-1] 1'), {}),
            (parse_circuit('X_ERROR(1/4) 0'), {'sampling': {'method': 'importance', 'proposal_scale': '2'}}),
        ]
        for circuit, kwargs in cases:
            with self.subTest(circuit=circuit):
                compiled = compile_circuit(circuit, **kwargs)
                with self.assertRaises(ValueError):
                    CudaExperiment(compiled)

    def test_native_buffer_validation(self):
        plan = self.magic_plan()
        bad = [([0, 2], [0]), ([0, 1], [plan.actions]), ([1], []), ([0, 2, 1], [0])]
        for offsets, indices in bad:
            with self.assertRaises(ValueError):
                _native._cuda_map_frames(plan.handle, np.array(offsets, dtype='<u8').tobytes(),
                                        np.array(indices, dtype='<u4').tobytes())
        with self.assertRaises(ValueError):
            _native._cuda_map_frames(plan.handle, b'x', b'')
        with self.assertRaises(ValueError):
            _native._cuda_evaluate_frames(plan.handle, b'x')


if __name__ == '__main__':
    unittest.main()
