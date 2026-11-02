"""Fused GPU noise laws, RNG stream addressing and exact frame semantics."""
import math
import unittest

from tests.support.cuda import require_cuda

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna import compile_circuit, parse_circuit
from magna._cuda_experiment import CudaExperiment
from magna._cuda_noise import FusedCudaNoise, MASK64
from tests.support.cuda_reference import (MASK32, PhiloxStream, cpu_frames, cpu_sample, evaluate_frames,
                                          evaluate_noise, noise_to_actions, oracle_sample, philox4x32)


def sampler(text, **kwargs):
    return FusedCudaNoise(CudaExperiment(compile_circuit(parse_circuit(text), **kwargs)))


class PhiloxOracleTests(unittest.TestCase):
    def test_official_random123_known_answer_vectors(self):
        # https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
        self.assertEqual(philox4x32((0, 0, 0, 0), (0, 0)),
                         (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8))
        self.assertEqual(philox4x32((MASK32,) * 4, (MASK32,) * 2),
                         (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD))
        rng = PhiloxStream(0, 0)
        self.assertEqual([rng(), rng()], [0xE169C58D6627E8D5, 0x9B00DBD8BC57AC4C])


class FusedNoiseCpuTests(unittest.TestCase):
    def test_exact_uint64_occurrence_boundary_and_extreme_metadata(self):
        # Philox(seed=0, shot=0)'s first uint64 from the official KAT.
        word = 0xE169C58D6627E8D5
        for threshold, expected in ((word, 0), (word + 1, 1)):
            sample = sampler(f'X_ERROR({threshold}/{1 << 64}) 0')
            self.assertEqual(int(sample.tables['thresholds'][0]), threshold)
            _, noise = cpu_sample(sample, 1, 0, return_noise=True)
            self.assertEqual(noise.indices.size, expected)
        for threshold in (1, MASK64):
            sample = sampler(f'X_ERROR({threshold}/{1 << 64}) 0')
            self.assertEqual(int(sample.tables['thresholds'][0]), threshold)
            self.assertEqual(int(sample.tables['certain'][0]), 0)

    def test_endpoints_nonuniform_correlated_and_report_actions(self):
        text = ('H 0\nT 0\nX_ERROR(0) 0\nZ_ERROR(1) 0\n'
                'PAULI_CHANNEL_1(1/8,1/4,1/16) 0\n'
                'CORRELATED_ERROR(1/4) X0 Y1\nELSE_CORRELATED_ERROR(1/3) Z0 X1\n'
                'M(1/4) 1\nCX rec[-1] 0')
        sample = sampler(text, task={'outputs': [0]}, ideal_state=parse_circuit('H 0\nT 0'))
        py_frames, py_noise = oracle_sample(sample, 37, MASK64, offset=2**32 + 11)
        reference, cpu_noise = cpu_sample(sample, 37, MASK64, offset=2**32 + 11, return_noise=True)
        np.testing.assert_array_equal(py_frames, reference)
        np.testing.assert_array_equal(py_noise.offsets, cpu_noise.offsets)
        np.testing.assert_array_equal(py_noise.indices, cpu_noise.indices)
        plan = sample.experiment
        np.testing.assert_array_equal(reference, cpu_frames(plan, cpu_noise))
        self.assertEqual(evaluate_frames(plan, reference), evaluate_noise(plan, cpu_noise))

    def test_occurrence_bitplanes_and_partial_blocks(self):
        text = 'H 0\nT 0\n' + '\n'.join(f'X_ERROR(1/4) {q}' for q in range(137))
        sample = sampler(text, task={'outputs': [0]}, ideal_state=parse_circuit('H 0\nT 0'))
        py_frames, py_noise = oracle_sample(sample, 31, 92016)
        native_frames, native_noise = cpu_sample(sample, 31, 92016, return_noise=True)
        np.testing.assert_array_equal(py_frames, native_frames)
        np.testing.assert_array_equal(py_noise.indices, native_noise.indices)
        np.testing.assert_array_equal(py_noise.offsets, native_noise.offsets)

    def test_no_noise_empty_batch_counter_limits_and_stale_plan(self):
        sample = sampler('H 0\nT 0', ideal_state=parse_circuit('H 0\nT 0'))
        frames, noise = cpu_sample(sample, 7, 0, return_noise=True)
        self.assertEqual(noise.indices.size, 0)
        self.assertTrue(np.array_equal(frames, np.tile(sample.experiment.base_frame, (7, 1))))
        self.assertEqual(cpu_sample(sample, 0, 0).shape, (0, sample.width))
        self.assertEqual(cpu_sample(sample, 1, MASK64, offset=MASK64).shape[0], 1)
        for shots, seed, offset in [(2, 0, MASK64), (1, -1, 0), (1, 0, -1),
                                     (1, 2**64, 0), (True, 0, 0), (1, 0.5, 0)]:
            with self.assertRaises((ValueError, TypeError)):
                cpu_sample(sample, shots, seed, offset=offset)
        sample.experiment.compiled._execution.configure(dict(sample.experiment.compiled.options,
                                                            max_quotient_rank=0))
        with self.assertRaisesRegex(ValueError, 'reconfigured'):
            cpu_sample(sample, 1, 0)

class FusedNoiseGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def test_chunk_independence_above_32bits(self):
        sample = sampler('X_ERROR(1/4) 0')
        seed, offset = MASK64 - 9, 2**32 - 3
        all_frames = sample.sample(271, seed, offset=offset)
        left, right = sample.sample(17, seed, offset=offset), sample.sample(254, seed, offset=offset + 17)
        np.testing.assert_array_equal(all_frames.get(), sample.cp.concatenate([left, right]).get())
        np.testing.assert_array_equal(all_frames.get(), cpu_sample(sample, 271, seed, offset=offset))

    def test_exact_uint64_occurrence_boundary_on_device(self):
        # The first Philox word of (seed 0, shot 0) is the Random123 known answer.
        word = 0xE169C58D6627E8D5
        for threshold, expected in ((word, 0), (word + 1, 1)):
            sample = sampler(f'X_ERROR({threshold}/{1 << 64}) 0')
            frames = sample.sample(1, 0).get()
            self.assertEqual(not np.array_equal(frames[0], sample.experiment.base_frame), bool(expected))
            np.testing.assert_array_equal(frames, cpu_sample(sample, 1, 0))

    def test_gpu_frame_and_original_exact_true_t_probabilities(self):
        text = ('H 0\nT 0\nX_ERROR(0) 0\nZ_ERROR(1) 0\n'
                'PAULI_CHANNEL_1(1/8,1/4,1/16) 0\n'
                'CORRELATED_ERROR(1/4) X0 Y1\nELSE_CORRELATED_ERROR(1/3) Z0 X1\n'
                'M(1/4) 1\nCX rec[-1] 0\n')
        text += '\n'.join(f'Z_ERROR(1/4) {q}' for q in range(137))
        sample = sampler(text, task={'outputs': [0]}, ideal_state=parse_circuit('H 0\nT 0'))
        frames = sample.sample(265, 84612)
        cpu, noise = cpu_sample(sample, 265, 84612, return_noise=True)
        np.testing.assert_array_equal(frames.get(), cpu)
        plan = sample.experiment
        self.assertEqual(evaluate_frames(plan, frames.get()), evaluate_noise(plan, noise))

    def test_marginals_exclusivity_cross_block_and_cross_shot_independence(self):
        sample = sampler('\n'.join(f'X_ERROR(1/4) {q}' for q in range(131)) +
                         '\nPAULI_CHANNEL_1(1/8,1/4,1/16) 0\n'
                         'CORRELATED_ERROR(1/4) X0 Y1\nELSE_CORRELATED_ERROR(1/3) Z0 X1\n'
                         'X_ERROR(0) 0\nZ_ERROR(1) 0')
        shots = 65536
        frames = sample.sample(shots, 268104)
        cpu, noise = cpu_sample(sample, shots, 268104, return_noise=True)
        np.testing.assert_array_equal(frames.get(), cpu)
        actions = noise_to_actions(sample, noise)
        present = actions != MASK32
        def check(count, probability, n=shots):
            self.assertLess(abs(count - n * probability), 7 * math.sqrt(n * probability * (1 - probability)) + 1)
        for site in (0, 1, 63, 64, 127, 130):
            check(int(present[:, site].sum()), 0.25)
        for first, second in ((0, 1), (0, 64), (63, 64), (64, 130)):
            check(int((present[:, first] & present[:, second]).sum()), 0.25**2)
        check(int((present[:-1, 0] & present[1:, 0]).sum()), 0.25**2, shots - 1)
        # Each site stores one action ID, so alternatives are mutually exclusive.
        start = int(sample.experiment.metadata['offsets'][131])
        for outcome, probability in enumerate((1/8, 1/4, 1/16)):
            check(int((actions[:, 131] == start + outcome).sum()), probability)
        start = int(sample.experiment.metadata['offsets'][132])
        check(int((actions[:, 132] == start).sum()), 0.25)
        check(int((actions[:, 132] == start + 1).sum()), 0.25)
        self.assertFalse(present[:, 133].any())
        self.assertTrue(present[:, 134].all())

    def test_zero_shots_and_no_noise_on_gpu(self):
        sample = sampler('H 0\nT 0', ideal_state=parse_circuit('H 0\nT 0'))
        self.assertEqual(sample.sample(0, 0).shape, (0, sample.width))
        frames = sample.sample(5, 0)
        np.testing.assert_array_equal(frames.get(), cpu_sample(sample, 5, 0))


if __name__ == '__main__':
    unittest.main()
