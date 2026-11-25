"""Native exact completed-marginal draws and Monte Carlo summaries."""
import math
import statistics
import unittest
from decimal import Decimal, localcontext

from magna import _native
from tests.support import native_testing


def ket(width, base=0, columns=(), linear=(), power=0):
    return (width, base, columns, linear, (0,) * len(columns), power, 0)


class NativeFileRuntimeTests(unittest.TestCase):
    def test_scaled_scalar_ratios_preserve_underflow_and_sign(self):
        one = ((1, 0, 0, 0), 0)
        pairs = [(((9, 0, 0, 0), -1160), one),
                 (((-9, 0, 0, 0), -1160), one),
                 (((1, 0, 0, 0), -1000000), ((1, 0, 0, 0), 1000000)),
                 (((0, 0, 0, 0), 0), one),
                 (one, ((1, 0, 0, 0), -1000000))]
        self.assertEqual(native_testing.load().file_scalar_ratios(pairs),
                         ((.5625, -1156), (-.5625, -1156), (.5, -1999999), (0., 0), (.5, 1000001)))

    def test_scaled_scalar_ratios_exact_cancellation(self):
        # Pell units have a^2-2b^2=1. a-b sqrt(2) is exponentially tiny,
        # although both integer coefficients grow exponentially.
        a, b = 1, 0
        for _ in range(120):
            a, b = 3*a + 4*b, 2*a + 3*b
        small = ((a, -b, 0, b), -500000)
        large = ((a, b, 0, -b), 500000)
        result = native_testing.load().file_scalar_ratios(((small, large), (large, small)))
        with localcontext() as context:
            context.prec = 1000
            s = Decimal(a) - Decimal(b)*Decimal(2).sqrt()
            l = Decimal(a) + Decimal(b)*Decimal(2).sqrt()
            for (mantissa, exponent), ratio, offset in zip(result, (s/l, l/s), (-1000000, 1000000)):
                self.assertTrue(.5 <= mantissa < 1)
                expected = float(ratio * Decimal(2)**(offset-exponent))
                self.assertEqual(mantissa, expected)

    def test_scaled_scalar_ratios_round_half_even(self):
        one = ((1, 0, 0, 0), 0)
        pairs = [((((1 << 53)+1, 0, 0, 0), -54), one),
                 ((((1 << 53)+3, 0, 0, 0), -54), one),
                 ((((1 << 54)-1, 0, 0, 0), -54), one)]
        self.assertEqual(native_testing.load().file_scalar_ratios(pairs),
                         ((.5, 0), (.5 + 2**-52, 0), (.5, 1)))

    def test_scaled_scalar_ratios_reject_nonreal_and_nonpositive_denominator(self):
        one = ((1, 0, 0, 0), 0)
        for denominator in (((0, 0, 0, 0), 0), ((-1, 0, 0, 0), 0)):
            with self.assertRaisesRegex(ValueError, "positive"):
                native_testing.load().file_scalar_ratios(((one, denominator),))
        with self.assertRaisesRegex(RuntimeError, "real"):
            native_testing.load().file_scalar_ratios(((((0, 0, 1, 0), 0), one),))

    def test_affine_mixture_big_boundary_and_reproducibility(self):
        width = 130
        all_bits = (1 << width) - 1
        raw = ((((1, 0, 0, 0), -1), ket(width, columns=(all_bits,), linear=(0,))),)
        draws = [native_testing.load().sample_boundary_terms(raw, width, seed) for seed in range(256)]
        self.assertEqual(set(draws), {0, all_bits})
        self.assertLess(abs(sum(x != 0 for x in draws) / 256 - .5), .12)
        self.assertEqual(draws, [native_testing.load().sample_boundary_terms(raw, width, seed) for seed in range(256)])

    def test_exact_irrational_signed_global_marginal(self):
        # P(x)=1/2 + (-1)^x sqrt(2)/4. The second term has varying
        # signs; a positive affine-mixture shortcut cannot be used.
        raw = (
            (((1, 0, 0, 0), -1), ket(1, columns=(1,), linear=(0,))),
            (((0, 1, 0, -1), -2), ket(1, columns=(1,), linear=(2,))),
        )
        draws = [native_testing.load().sample_boundary_terms(raw, 1, seed) for seed in range(1024)]
        expected = .5 - math.sqrt(2) / 4
        self.assertLess(abs(sum(draws) / len(draws) - expected), .035)

    def test_complex_terms_cancel_before_sampling(self):
        # Individual terms need not be real. Their global sum is deterministic.
        raw = (
            (((1, 0, 1, 0), 0), ket(1, base=1)),
            (((0, 0, -1, 0), 0), ket(1, base=1)),
        )
        for seed in range(10):
            self.assertEqual(native_testing.load().sample_boundary_terms(raw, 1, seed), 1)

    def test_boundary_rejects_invalid_distribution_and_shape(self):
        with self.assertRaisesRegex(ValueError, "zero or negative"):
            native_testing.load().sample_boundary_terms((), 1, 0)
        with self.assertRaisesRegex(ValueError, "width"):
            native_testing.load().sample_boundary_terms(((((1, 0, 0, 0), 0), ket(2)),), 1, 0)
        signed = (
            (((2, 0, 0, 0), 0), ket(1)),
            (((-1, 0, 0, 0), 0), ket(1, base=1)),
        )
        with self.assertRaisesRegex(ValueError, "Bernoulli"):
            native_testing.load().sample_boundary_terms(signed, 1, 0)
        with self.assertRaises((ValueError, OverflowError)):
            native_testing.load().sample_boundary_terms((), 1, -1)

    def test_categories_zero_one_residual_and_large_rational_inputs(self):
        denominator = 10**500
        categories = ((), ((0, 1),), ((1, 1),), ((0, 1), (denominator, denominator)))
        self.assertEqual(native_testing.load().sample_file_categories(categories, 10), (-1, -1, 0, 1))
        counts = [0, 0, 0]
        for seed in range(1024):
            selected, = native_testing.load().sample_file_categories((((1, 10), (1, 5)),), seed)
            counts[selected + 1] += 1
        for actual, expected in zip(counts, (.7, .1, .2)):
            self.assertLess(abs(actual / 1024 - expected), .045)

    def test_category_rejects_invalid_fraction(self):
        for category in (((1, 0),), ((-1, 2),), ((1, -2),), ((3, 4), (1, 2))):
            with self.subTest(category=category), self.assertRaises(ValueError):
                native_testing.load().sample_file_categories((category,), 0)

    def test_native_summary_matches_paired_delta_method(self):
        rows = [(0.2, .1, (.3, -.1)), (.8, .2, (-.2, .5)), (.5, 0., (.1, .1)), (1., .4, (.7, -.3))]
        result = native_testing.load().summarize_file_draws(rows)
        a = [row[0] for row in rows]
        e = [row[1] for row in rows]
        ratio = statistics.mean(e) / statistics.mean(a)
        se = lambda values: statistics.stdev(values) / math.sqrt(len(values))
        self.assertAlmostEqual(result["estimates"]["acceptance"], statistics.mean(a))
        self.assertAlmostEqual(result["estimates"]["error"], statistics.mean(e))
        self.assertAlmostEqual(result["estimates"]["conditional_error"], ratio)
        self.assertAlmostEqual(result["standard_errors"]["acceptance"], se(a))
        self.assertAlmostEqual(result["standard_errors"]["error"], se(e))
        self.assertAlmostEqual(result["standard_errors"]["conditional_error"],
                               se([x - ratio*y for x, y in zip(e, a)]) / statistics.mean(a))
        for j in range(2):
            ms = [row[2][j] for row in rows]
            self.assertAlmostEqual(result["estimates"]["moments"][j], statistics.mean(ms))
            self.assertAlmostEqual(result["standard_errors"]["moments"][j], se(ms))

    def test_scientific_summary_matches_independent_statistics_across_scale_changes(self):
        # Offsets near +/-500 cross the accumulator's scale boundary; the wider
        # offsets exercise summaries whose ordinary representation cannot exist.
        exponents = (-2, -1, 0, 1, 2, 3, 3, 1)
        rows = [(math.ldexp(.5, exponent), math.ldexp((j % 3) / 8, exponent),
                 math.ldexp((-1)**j * .25, exponent)) for j, exponent in enumerate(exponents)]
        columns = list(zip(*rows))
        ratio = statistics.mean(columns[1]) / statistics.mean(columns[0])
        se = lambda values: statistics.stdev(values) / math.sqrt(len(values))
        estimates = [statistics.mean(columns[0]), statistics.mean(columns[1]), ratio,
                     statistics.mean(columns[2])]
        errors = [se(columns[0]), se(columns[1]),
                  se([e - ratio*a for a, e, _ in rows]) / statistics.mean(columns[0]), se(columns[2])]
        for offset in (-1100, -501, 0, 499, 1100):
            with self.subTest(offset=offset):
                shifted = [[(mantissa, exponent + offset) for mantissa, exponent in map(math.frexp, row)]
                           for row in rows]
                result = native_testing.load().summarize_scientific_draws(shifted)
                for field, expected in (('estimates_scientific', estimates),
                                        ('standard_errors_scientific', errors)):
                    actual = result[field]
                    values = [actual['acceptance'], actual['error'], actual['conditional_error'],
                              actual['moments'][0]]
                    for j, (value, reference) in enumerate(zip(values, expected)):
                        normalized = math.ldexp(value['mantissa'], value['exponent2'] - (0 if j == 2 else offset))
                        self.assertAlmostEqual(normalized, reference, places=14, msg=(offset, field, j))

    def test_summary_missing_metrics_single_draw_and_zero_acceptance(self):
        self.assertEqual(native_testing.load().summarize_file_draws([]), dict(estimates=None, standard_errors=None))
        result = native_testing.load().summarize_file_draws([(0., None, ())])
        self.assertIsNone(result["estimates"]["conditional_error"])
        self.assertIsNone(result["estimates"]["error"])
        self.assertIsNone(result["standard_errors"]["acceptance"])
        result = native_testing.load().summarize_file_draws([(0., 0., ()), (0., 0., ())])
        self.assertIsNone(result["estimates"]["conditional_error"])
        self.assertIsNone(result["standard_errors"]["conditional_error"])
        self.assertEqual(result["standard_errors"]["acceptance"], 0.)

    def test_summary_rejects_nonfinite_and_inconsistent_metrics(self):
        for rows in ([(float('nan'), None, ())], [(1., None, ()), (1., 0., ())],
                     [(1., None, ()), (1., None, (0.,))], [(1., None, (float('inf'),))]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                native_testing.load().summarize_file_draws(rows)


if __name__ == "__main__":
    unittest.main()
