"""Independent checks for exact-error subtraction and GPU batch statistics."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from types import SimpleNamespace
import unittest

from tests.support.cuda import require_cuda

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna.cuda_sampler import CudaSampler, _merge_moments as merge_moments, _summarize_moments as summarize
from tests.support.cuda_reference import cpu_moments, cpu_run


def dyadic(value):
    value = Fraction(value)
    if value.denominator & (value.denominator - 1):
        raise ValueError('test oracle requires a dyadic rational')
    return [value.numerator, 0, 0, 0], -(value.denominator.bit_length() - 1)


def exact_arrays(acceptances, errors=None, norm_power=0):
    coefficients, powers = [], []
    for j, acceptance in enumerate(acceptances):
        ca, pa = dyadic(acceptance)
        cs, ps = [ca], [pa]
        if errors is not None:
            ct, pt = dyadic(Fraction(acceptance) - Fraction(errors[j]))
            cs.append(ct)
            ps.append(pt + norm_power)
        coefficients.append(cs)
        powers.append(ps)
    queries = 1 if errors is None else 2
    return np.asarray(coefficients, dtype=np.int64).reshape(-1, queries, 4), np.asarray(powers, dtype=np.int32).reshape(-1, queries)


def independent_means(a, e):
    a, e = list(map(float, a)), list(map(float, e))
    return np.array([math.fsum(a)/len(a), math.fsum(e)/len(e)])


def assert_moments_close(actual, expected, rtol):
    assert actual.count == expected.count, (actual.count, expected.count)
    np.testing.assert_allclose(actual.mean, expected.mean, rtol=rtol, atol=1e-16)
    # An exactly zero residual entry is compared at the scale of the whole factor.
    scale = float(np.max(np.abs(expected.root)))
    np.testing.assert_allclose(actual.root, expected.root, rtol=rtol, atol=rtol*scale)


def exact_standard_errors(a, e=None):
    """Fraction oracle, independent of raw moment subtraction and QR updates."""
    a = list(map(Fraction, a))
    n = len(a)
    mean_a = sum(a)/n
    acceptance = math.sqrt(float(sum((x-mean_a)**2 for x in a)/(n*(n-1))))
    if e is None:
        return acceptance, None, None
    e = list(map(Fraction, e))
    mean_e = sum(e)/n
    error = math.sqrt(float(sum((x-mean_e)**2 for x in e)/(n*(n-1))))
    ratio = mean_e/mean_a
    residual = [y-ratio*x for x,y in zip(a,e)]
    conditional = math.sqrt(float(sum(x*x for x in residual)/(n*(n-1))))/float(mean_a)
    return acceptance, error, conditional


def invalid_cases():
    def item(c, p):
        return np.array(c, dtype=np.int64), np.array(p, dtype=np.int32)
    return [
        item([[[-1, 0, 0, 0]]], [[0]]),
        item([[[1, 0, 1, 0]]], [[-2]]),
        item([[[0, 1, 0, 0]]], [[-2]]),
        item([[[1, 0, 0, 0]]], [[1024]]),
        item([[[1, 0, 0, 0]]], [[-1075]]),
        # Probability survives float64 but its squared moment does not.
        item([[[1, 0, 0, 0]]], [[-600]]),
        item([[[1, 0, 0, 0], [1, 0, 0, 0]]], [[0, -62]]),
        item([[[1, 0, 0, 0], [(1 << 63)-1, 0, 0, 0]]], [[0, 0]]),
        item([[[1, 0, 0, 0], [1, 0, 0, 0]]], [[-1, 0]]),
        item([[[1, 0, 0, 0], [-1, 0, 0, 0]]], [[-30, -60]]),
    ]


def pell(bits):
    """x, y with x*x - 2*y*y = 1 and x above 2**bits."""
    x, y = 3, 2
    while x < 1 << bits:
        x, y = 3*x + 4*y, 2*x + 3*y
    return x, y


def large_gap_cases():
    """Exact values whose aligned coefficients exceed 2**31."""
    x, y = pell(50)
    with localcontext() as context:
        context.prec = 90
        tiny = Decimal(1) / (Decimal(x) + Decimal(y)*Decimal(2).sqrt())
        cases = [
            # Acceptance 1/2 with fidelity mass 2**-36.
            ([[1, 0, 0, 0], [1, 0, 0, 0]], [-1, -36], [0.5, 0.5 - 2.**-36]),
            # Acceptance x - y*sqrt(2) = 1/(x + y*sqrt(2)).
            ([[x, -y, 0, y]], [0], [float(tiny)]),
            # Acceptance 1/2 with fidelity mass x - y*sqrt(2).
            ([[1, 0, 0, 0], [x, -y, 0, y]], [-1, 0], [0.5, float(Decimal(1)/2 - tiny)]),
        ]
    return [(np.array([c], dtype=np.int64), np.array([p], dtype=np.int32), np.array(v)) for c, p, v in cases]


class CpuMomentTests(unittest.TestCase):
    def test_constant_irrational_has_zero_standard_error(self):
        for n in (3, 301, 2049):
            c = np.tile(np.array([[[2, 1, 0, -1]]], dtype=np.int64), (n, 1, 1))
            p = np.full((n, 1), -2, dtype=np.int32)
            chunks = [cpu_moments(c[i:i+17], p[i:i+17], 0) for i in range(0, n, 17)]
            for moments in (cpu_moments(c,p,0), merge_moments(chunks)):
                self.assertEqual(summarize(moments, False)['standard_errors']['acceptance'], 0.)

    def test_close_probabilities_and_chunking_keep_small_variance(self):
        a = [Fraction(1, 2)+Fraction(i%3, 2**28) for i in range(301)]
        c,p = exact_arrays(a)
        expected = exact_standard_errors(a)[0]
        chunks = [cpu_moments(c[i:i+7],p[i:i+7],0) for i in range(0,len(a),7)]
        for moments in (cpu_moments(c,p,0),merge_moments(chunks)):
            actual = summarize(moments, False)['standard_errors']['acceptance']
            self.assertGreater(actual,0)
            np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=0)
        c,p = exact_arrays([Fraction(1,2), Fraction(1,2)+Fraction(1,2**28)])
        self.assertEqual(summarize(cpu_moments(c,p,0), False)['standard_errors']['acceptance'],2.**-29)

    def test_conditional_error_uses_stable_residual_factor(self):
        a = [Fraction(1+(i%4),4) for i in range(301)]
        e = [x/2+Fraction((i%3)-1,2**28) for i,x in enumerate(a)]
        c,p = exact_arrays(a,e)
        expected = exact_standard_errors(a,e)
        chunks=[cpu_moments(c[i:i+13],p[i:i+13],0) for i in range(0,len(a),13)]
        for moments in (cpu_moments(c,p,0),merge_moments(chunks)):
            errors=summarize(moments)['standard_errors']
            np.testing.assert_allclose([errors['acceptance'],errors['error'],errors['conditional_error']],
                                       expected,rtol=1e-7,atol=0)
            self.assertGreater(errors['conditional_error'],0)

    def test_small_centered_variance_does_not_underflow_before_sqrt(self):
        a = [Fraction(1,2**500),Fraction(1,2**500)+Fraction(1,2**528)]
        c,p = exact_arrays(a)
        error=summarize(cpu_moments(c,p,0), False)['standard_errors']['acceptance']
        self.assertEqual(error,2.**-529)

    def test_exact_error_survives_float_cancellation(self):
        # Both A and T round to 1 if converted first; their exact difference
        # still has a nonzero binary64 representation.
        c = np.array([[[1, 0, 0, 0], [(1 << 55)-1, 0, 0, 0]]], dtype=np.int64)
        p = np.array([[0, -55]], dtype=np.int32)
        self.assertEqual(float((1 << 55)-1) * 2.**-55, 1.)
        np.testing.assert_array_equal(cpu_moments(c, p, 0).mean, [1, 2.**-55])
        p[0, 1] -= 1
        np.testing.assert_array_equal(cpu_moments(c, p, -1).mean, [1, 2.**-55])

    def test_stable_irrational_conversion_with_opposite_large_coefficients(self):
        pairs = [(131836323, -93222358), (-1855077841, 1311738121)]
        c = np.array([[[a, b, 0, -b]] for a, b in pairs], dtype=np.int64)
        p = np.zeros((2, 1), dtype=np.int32)
        with localcontext() as context:
            context.prec = 90
            values = [float(Decimal(a) + Decimal(b) * Decimal(2).sqrt()) for a, b in pairs]
        self.assertTrue(all(0 < v < 1e-8 for v in values))
        np.testing.assert_allclose(cpu_moments(c, p, 0).mean, independent_means(values, [0, 0]),
                                   rtol=3e-15, atol=0)

    def test_large_exponent_gap_converts_exactly(self):
        for i, (c, p, expected) in enumerate(large_gap_cases()):
            with self.subTest(case=i):
                np.testing.assert_allclose(cpu_moments(c, p, 0).mean[:len(expected)], expected, rtol=3e-15, atol=0)

    def test_nonunit_target_and_paired_delta_method(self):
        a = [Fraction(1, 2), Fraction(1), Fraction(1, 4), Fraction(0)]
        e = [Fraction(1, 8), Fraction(1, 4), Fraction(0), Fraction(0)]
        c, p = exact_arrays(a, e, norm_power=-2)
        moments = cpu_moments(c, p, -2)
        np.testing.assert_array_equal(moments.mean, independent_means(a, e))
        result = summarize(moments)
        aa, ee = np.array(a, dtype=float), np.array(e, dtype=float)
        ratio = ee.mean()/aa.mean()
        residual = ee - ratio*aa
        expected_ratio_se = np.std(residual, ddof=1) / math.sqrt(4) / aa.mean()
        self.assertEqual(result['estimates']['acceptance'], float(sum(a)/4))
        self.assertEqual(result['estimates']['error'], float(sum(e)/4))
        self.assertAlmostEqual(result['estimates']['conditional_error'], 3/14)
        self.assertAlmostEqual(result['standard_errors']['acceptance'], np.std(aa, ddof=1)/2)
        self.assertAlmostEqual(result['standard_errors']['error'], np.std(ee, ddof=1)/2)
        self.assertAlmostEqual(result['standard_errors']['conditional_error'], expected_ratio_se)

    def test_empty_single_and_acceptance_only_statistics(self):
        c, p = exact_arrays([])
        self.assertEqual(cpu_moments(c, p, 0).count, 0)
        empty = summarize(cpu_moments(c, p, 0))
        self.assertTrue(all(v is None for section in empty.values() for v in section.values()))
        c, p = exact_arrays([Fraction(1, 4)], [Fraction(1, 8)])
        result = summarize(cpu_moments(c, p, 0))
        self.assertEqual(result['estimates']['conditional_error'], 0.5)
        self.assertTrue(all(v is None for v in result['standard_errors'].values()))
        c, p = exact_arrays([1, 0])
        result = summarize(cpu_moments(c, p, 0), has_error=False)
        self.assertIsNone(result['estimates']['error'])
        self.assertIsNone(result['estimates']['conditional_error'])
        self.assertIsNone(result['standard_errors']['error'])
        self.assertEqual(result['standard_errors']['acceptance'], 0.5)
        c, p = exact_arrays([0] * 5, [0] * 5)
        result = summarize(cpu_moments(c, p, 0))
        self.assertIsNone(result['estimates']['conditional_error'])

    def test_rejects_invalid_scalar_ranges_and_probabilities(self):
        for i, (c, p) in enumerate(invalid_cases()):
            with self.subTest(case=i), self.assertRaises(ValueError):
                cpu_moments(c, p, 0)

    def test_rejects_noninteger_inputs(self):
        c, p = exact_arrays([Fraction(1, 2)])
        with self.assertRaises((ValueError, TypeError)):
            cpu_moments(c.astype(float), p, 0)
        with self.assertRaises((ValueError, TypeError)):
            cpu_moments(c, p.astype(float), 0)


class GpuMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def fake_sampler(self, norm_power=0):
        obj = object.__new__(CudaSampler)
        obj._noise = SimpleNamespace(cp=self.cp, has_importance_weights=False)
        obj._norm_power = norm_power
        obj._statistics_module = None
        obj._unit_weights = None
        obj._history = None
        obj._has_error = True
        obj._experiment = SimpleNamespace(metadata={'max_shots': 1_000_000}, validate=lambda: None)
        return obj

    def gpu(self, c, p, norm_power=0):
        exact = SimpleNamespace(coefficients=self.cp.asarray(c), powers=self.cp.asarray(p),
                                status=self.cp.zeros(p.shape, dtype=self.cp.int32))
        return self.fake_sampler(norm_power)._gpu_moments(exact)

    def test_gpu_constant_irrational_and_near_constant_variance(self):
        for n in (3,301,2049):
            c=np.tile(np.array([[[2,1,0,-1]]],dtype=np.int64),(n,1,1))
            p=np.full((n,1),-2,dtype=np.int32)
            actual=summarize(self.gpu(c,p), False)
            self.assertEqual(actual['standard_errors']['acceptance'],0.)
        a=[Fraction(1,2)+Fraction(i%3,2**28) for i in range(301)]
        c,p=exact_arrays(a)
        expected=exact_standard_errors(a)[0]
        chunks=[self.gpu(c[i:i+17],p[i:i+17]) for i in range(0,len(a),17)]
        for result in (self.gpu(c,p),merge_moments(chunks)):
            error=summarize(result, False)['standard_errors']['acceptance']
            np.testing.assert_allclose(error,expected,rtol=1e-7,atol=0)
        c,p=exact_arrays([Fraction(1,2**500),Fraction(1,2**500)+Fraction(1,2**528)])
        error=summarize(self.gpu(c,p), False)['standard_errors']['acceptance']
        np.testing.assert_allclose(error,2.**-529,rtol=2e-15,atol=0)

    def test_gpu_conditional_residual_stays_nonzero(self):
        a=[Fraction(1+(i%4),4) for i in range(301)]
        e=[x/2+Fraction((i%3)-1,2**28) for i,x in enumerate(a)]
        c,p=exact_arrays(a,e)
        expected=exact_standard_errors(a,e)
        chunks=[self.gpu(c[i:i+17],p[i:i+17]) for i in range(0,len(a),17)]
        for result in (self.gpu(c,p),merge_moments(chunks)):
            errors=summarize(result)['standard_errors']
            np.testing.assert_allclose([errors['acceptance'],errors['error'],errors['conditional_error']],
                                       expected,rtol=1e-7,atol=0)

    def test_gpu_cpu_moments_with_tail_multiple_blocks_and_normalization(self):
        a = [Fraction(1, 2), Fraction(1), Fraction(1, 4), Fraction(0)] * 75 + [Fraction(1)]
        e = [Fraction(1, 8), Fraction(1, 4), Fraction(0), Fraction(0)] * 75 + [Fraction(1, 2)]
        c, p = exact_arrays(a, e, norm_power=-2)
        np.testing.assert_allclose(self.gpu(c, p, -2).mean, independent_means(a, e), rtol=2e-15, atol=0)
        assert_moments_close(self.gpu(c, p, -2), cpu_moments(c, p, -2), rtol=2e-15)

    def test_gpu_keeps_rare_error_and_stable_irrational_values(self):
        c = np.array([[[1, 0, 0, 0], [(1 << 55)-1, 0, 0, 0]]], dtype=np.int64)
        p = np.array([[0, -55]], dtype=np.int32)
        np.testing.assert_array_equal(self.gpu(c, p).mean, [1, 2.**-55])
        pairs = [(131836323, -93222358), (-1855077841, 1311738121)]
        c = np.array([[[a, b, 0, -b]] for a, b in pairs], dtype=np.int64)
        p = np.zeros((2, 1), dtype=np.int32)
        with localcontext() as context:
            context.prec = 90
            values = [float(Decimal(a) + Decimal(b)*Decimal(2).sqrt()) for a, b in pairs]
        np.testing.assert_allclose(self.gpu(c, p).mean, independent_means(values, [0, 0]), rtol=3e-15, atol=0)

    def test_gpu_large_exponent_gap_converts_exactly(self):
        for i, (c, p, expected) in enumerate(large_gap_cases()):
            with self.subTest(case=i):
                np.testing.assert_allclose(self.gpu(c, p).mean[:len(expected)], expected, rtol=3e-15, atol=0)

    def test_gpu_zero_shots(self):
        c, p = exact_arrays([])
        self.assertEqual(self.gpu(c, p).count, 0)
        run = self.fake_sampler().run(0, seed=123, batch_size=7)
        self.assertEqual(run['completed'], 0)
        self.assertIsNone(run['estimates']['acceptance'])

    def test_gpu_rejects_invalid_scalar_ranges_and_probabilities(self):
        for i, (c, p) in enumerate(invalid_cases()):
            with self.subTest(case=i), self.assertRaises(ValueError):
                self.gpu(c, p)

    def test_query_contract_failure_is_not_silently_aggregated(self):
        c, p = exact_arrays([Fraction(1, 2)], [Fraction(1, 4)])
        exact = SimpleNamespace(coefficients=self.cp.asarray(c), powers=self.cp.asarray(p),
                                status=self.cp.asarray([[0, 1]], dtype=self.cp.int32))
        with self.assertRaisesRegex(ValueError, 'flags=16'):
            self.fake_sampler()._gpu_moments(exact)

    def test_complete_true_t_sampler_cpu_gpu_and_chunking(self):
        from magna import compile_circuit, parse_circuit
        sampler = CudaSampler(compile_circuit(parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
                                              ideal_state=parse_circuit('H 0\nT 0')))
        gpu = sampler.run(257, seed=20261026, offset=2**32-19, batch_size=128)
        cpu = cpu_run(sampler, 257, seed=20261026, offset=2**32-19, batch_size=17)
        self.assertEqual(gpu['completed'], 257)
        for key in gpu['estimates']:
            np.testing.assert_allclose(gpu['estimates'][key],cpu['estimates'][key],rtol=2e-14,atol=1e-16)
            np.testing.assert_allclose(gpu['standard_errors'][key],cpu['standard_errors'][key],rtol=2e-14,atol=1e-16)
        self.assertEqual(gpu['estimates']['acceptance'], 1.)


if __name__ == '__main__':
    unittest.main()
