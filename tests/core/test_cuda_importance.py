"""Native homogeneous importance tables and fused device weight generation."""
from fractions import Fraction
from random import Random
import unittest

from tests.support.cuda import require_cuda

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna import compile_circuit, parse_circuit
from magna._cuda_experiment import CudaExperiment
from magna._cuda_noise import FusedCudaNoise
from tests.support.cuda_reference import (cpu_sample, cpu_sample_weighted, draw, evaluate_frames,
                                          evaluate_noise, oracle_sample, weights_for_noise)


def resource_noise(count=14, *, scale=None, endpoints=False):
    text = 'H 0\nT 0\n'
    if endpoints:
        text += 'Z_ERROR(0) 0\nZ_ERROR(1) 0\n'
    text += '\n'.join('Z_ERROR(1/1000) 0' for _ in range(count))
    if scale is None:
        scale = Fraction(2000, count)
    compiled = compile_circuit(parse_circuit(text), ideal_state=parse_circuit('H 0\nT 0'),
                               sampling={'method': 'importance', 'proposal_scale': str(scale)})
    return FusedCudaNoise(CudaExperiment(compiled))


def expected_weight(count, faults, p, q):
    return ((1-p)/(1-q))**(count-faults) * (p/q)**faults


class ImportanceNoiseCpuTests(unittest.TestCase):
    def test_all_paper_bh_tables_match_independent_exact_fraction_formula(self):
        p = Fraction(1, 1000)
        for count in (14, 20, 26, 38, 68, 128, 188, 248, 308):
            with self.subTest(resource_count=count):
                noise = resource_noise(count)
                self.assertTrue(noise.has_importance_weights)
                self.assertEqual(int(noise.weighted_sites.sum()), count)
                self.assertEqual(len(noise.likelihood_values), count+1)
                q = Fraction(2, count)
                threshold = int(q*(1 << 64))
                self.assertTrue(np.all(noise.tables['thresholds'] == threshold))
                for faults, value in enumerate(noise.likelihood_values):
                    self.assertEqual(value, float(expected_weight(count, faults, p, q)))

    def test_matches_public_native_sampler_exact_likelihoods(self):
        noise = resource_noise(14)
        rng = Random(20261028)
        seeds = [rng.getrandbits(64) for _ in range(128)]
        batch = draw(noise.experiment, seeds)
        weights = weights_for_noise(noise, batch)
        rng = Random(20261028)
        for shot in range(128):
            public = noise.experiment.compiled.sample(rng)
            raw = public['exact']['noise_likelihood_ratio_hex']
            native = Fraction(int(raw[0], 16), int(raw[1], 16))
            faults = int(batch.offsets[shot+1]-batch.offsets[shot])
            independent = expected_weight(14, faults, Fraction(1, 1000), Fraction(1, 7))
            self.assertEqual(native, independent)
            self.assertEqual(weights[shot], float(native))
            self.assertAlmostEqual(public['acceptance'], weights[shot], places=14)

    def test_unweighted_interfaces_cannot_silently_discard_likelihood(self):
        noise = resource_noise()
        with self.assertRaisesRegex(ValueError, 'sample_weighted'):
            noise.sample(1, 0)
        with self.assertRaisesRegex(ValueError, 'cpu_sample_weighted'):
            cpu_sample(noise, 1, 0)

    def test_endpoints_and_unity_proposals_do_not_change_weight_count(self):
        noise = resource_noise(14, endpoints=True)
        self.assertEqual(noise.weighted_sites.tolist(), [0, 0]+[1]*14)
        frames, weights, batch = cpu_sample_weighted(noise, 257, 71, return_noise=True)
        np.testing.assert_array_equal(weights, weights_for_noise(noise, batch))
        oracle, oracle_batch = oracle_sample(noise, 3, 71)
        np.testing.assert_array_equal(frames[:3], oracle)
        np.testing.assert_array_equal(batch.indices[:int(batch.offsets[3])], oracle_batch.indices)
        # Exactly one certain unweighted action occurs in every shot.
        for shot in range(257):
            self.assertIn(1, batch.indices[batch.offsets[shot]:batch.offsets[shot+1]])
        for text, scale in [('X_ERROR(0) 0\nX_ERROR(1) 0', '3'),
                            ('X_ERROR(1/4) 0\nZ_ERROR(1/4) 0', '1')]:
            experiment = CudaExperiment(compile_circuit(parse_circuit(text),
                sampling={'method': 'importance', 'proposal_scale': scale}))
            unit = FusedCudaNoise(experiment)
            self.assertFalse(unit.has_importance_weights)
            a = cpu_sample(unit, 17, 123)
            b, weights = cpu_sample_weighted(unit, 17, 123)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(weights, np.ones(17))

    def test_heterogeneous_missing_tables_and_unresolved_proposals_are_rejected(self):
        circuits = [
            ('X_ERROR(1/4) 0', '2'),
            ('X_ERROR(1/1000) 0 1\nZ_ERROR(2/1000) 0 1', '2'),
            ('\n'.join('Z_ERROR(1/1000) 0' for _ in range(513)), '2'),
            ('Z_ERROR(1/4) 0 1', '1/' + str(1 << 100)),
        ]
        for text, scale in circuits:
            with self.subTest(scale=scale, locations=text.count('ERROR')):
                compiled = compile_circuit(parse_circuit(text),
                    sampling={'method':'importance', 'proposal_scale':scale})
                with self.assertRaisesRegex(ValueError, 'importance'):
                    CudaExperiment(compiled)

    def test_zero_shots_and_chunk_partition(self):
        noise = resource_noise(68)
        frames, weights, empty = cpu_sample_weighted(noise, 0, 1, return_noise=True)
        self.assertEqual(frames.shape, (0, noise.width))
        self.assertEqual(weights.shape, (0,))
        self.assertEqual(empty.shots, 0)
        full, full_w = cpu_sample_weighted(noise, 257, 912, offset=2**32-37)
        a, wa = cpu_sample_weighted(noise, 93, 912, offset=2**32-37)
        b, wb = cpu_sample_weighted(noise, 164, 912, offset=2**32-37+93)
        np.testing.assert_array_equal(full, np.concatenate([a, b]))
        np.testing.assert_array_equal(full_w, np.concatenate([wa, wb]))


class ImportanceNoiseGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def test_fused_weighted_frames_and_exact_native_oracle(self):
        # A nonuniform categorical law, report noise, zero and certain sites.
        text = ('H 0\nT 0\nX_ERROR(0) 0\nZ_ERROR(1) 0\n'
                'PAULI_CHANNEL_1(1/16,3/16,0) 0\n'
                'PAULI_CHANNEL_1(1/16,3/16,0) 0\nM(1/4) 0')
        noise = FusedCudaNoise(CudaExperiment(compile_circuit(parse_circuit(text),
            ideal_state=parse_circuit('H 0\nT 0'),
            sampling={'method':'importance','proposal_scale':'2'})))
        frames, weights = noise.sample_weighted(257, 2971, offset=2**32-129)
        cpu_frames, cpu_weights, batch = cpu_sample_weighted(noise, 257, 2971,
            offset=2**32-129, return_noise=True)
        np.testing.assert_array_equal(frames.get(), cpu_frames)
        np.testing.assert_array_equal(weights.get(), cpu_weights)
        np.testing.assert_array_equal(cpu_weights, weights_for_noise(noise, batch))
        self.assertGreater(float(cpu_weights.max()), 1.)
        self.assertEqual(evaluate_frames(noise.experiment, cpu_frames), evaluate_noise(noise.experiment, batch))
        oracle, oracle_batch = oracle_sample(noise, 3, 2971, offset=2**32-129)
        np.testing.assert_array_equal(cpu_frames[:3], oracle)
        np.testing.assert_array_equal(batch.indices[:int(batch.offsets[3])], oracle_batch.indices)

    def test_paper_size_device_weight_distribution_and_chunking(self):
        noise = resource_noise(308)
        frames, weights = noise.sample_weighted(2048, 2026100800)
        cpu_frames, cpu_weights, batch = cpu_sample_weighted(noise, 2048, 2026100800, return_noise=True)
        np.testing.assert_array_equal(frames.get(), cpu_frames)
        np.testing.assert_array_equal(weights.get(), cpu_weights)
        a, wa = noise.sample_weighted(997, 2026100800)
        b, wb = noise.sample_weighted(1051, 2026100800, offset=997)
        np.testing.assert_array_equal(frames.get(), self.cp.concatenate([a, b]).get())
        np.testing.assert_array_equal(weights.get(), self.cp.concatenate([wa, wb]).get())
        # Independent expected fault count is n*q=2; this is a statistical
        # sanity check only, supplementing the deterministic checks above.
        counts = np.diff(batch.offsets).astype(float)
        self.assertLess(abs(counts.mean()-2), 6*np.sqrt(2*(1-2/308)/2048))
        empty_f, empty_w = noise.sample_weighted(0, 2)
        self.assertEqual(empty_f.shape, (0, noise.width))
        self.assertEqual(empty_w.shape, (0,))

    def test_physical_weighted_interface_is_identical_with_unit_weights(self):
        noise = FusedCudaNoise(CudaExperiment(compile_circuit(parse_circuit('DEPOLARIZE1(1/4) 0'))))
        frames = noise.sample(257, 26)
        weighted_frames, weights = noise.sample_weighted(257, 26)
        np.testing.assert_array_equal(frames.get(), weighted_frames.get())
        np.testing.assert_array_equal(weights.get(), np.ones(257))


if __name__ == '__main__':
    unittest.main()
