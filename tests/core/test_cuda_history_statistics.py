"""Range-safe history contributions and safe-subset statistics."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from types import SimpleNamespace
import unittest

import numpy as np

from magna._cuda_history_statistics import (
    _HistoryStatistics, _merge_scaled, _scientific_moments, _summarize_scaled)
from tests.support.cuda import require_cuda
from tests.support.cuda_reference import history_scientific_rows


def arrays(rows):
    return (np.asarray([[item[0] for item in row] for row in rows], dtype=np.int64),
            np.asarray([[item[1] for item in row] for row in rows], dtype=np.int32))


def scalar(value=1, power=0):
    return ([value, 0, 0, 0], power)


def scientific(value):
    if not value:
        return 0., 0
    return math.frexp(value)


def fraction_summary(rows):
    with localcontext() as context:
        context.prec = 100
        n = len(rows)
        a, e = [[Fraction(row[j]) for row in rows] for j in range(2)]
        ma, me = sum(a)/n, sum(e)/n
        ratio = me/ma
        def number(value):
            return Decimal(value.numerator)/Decimal(value.denominator)
        sa = (number(sum((x-ma)**2 for x in a)/(n*(n-1)))).sqrt()
        se = (number(sum((x-me)**2 for x in e)/(n*(n-1)))).sqrt()
        sr = (number(sum((y-ratio*x)**2 for x, y in zip(a, e))/(n*(n-1)))).sqrt()/number(ma)
        return dict(estimates=dict(acceptance=float(ma), error=float(me), conditional_error=float(ratio)),
                    standard_errors=dict(acceptance=float(sa), error=float(se), conditional_error=float(sr)))


class HistoryStatisticsCpuTests(unittest.TestCase):
    def test_independent_scales_and_chunking(self):
        rows = [(Fraction(i % 4, 4), Fraction(i % 4, 2**1000)+Fraction((i % 3)*(i % 4), 2**1005))
                for i in range(301)]
        expected = fraction_summary(rows)
        pairs = [[scientific(float(value)) for value in row] for row in rows]
        chunks = [_scientific_moments(pairs[i:i+7]) for i in range(0, len(pairs), 7)]
        for actual in (_scientific_moments(pairs), _merge_scaled(chunks)):
            self.assertEqual(actual.count, len(rows))
            result = _summarize_scaled(actual)
            for section in expected:
                np.testing.assert_allclose(list(result[section].values()), list(expected[section].values()),
                                           rtol=2e-13, atol=0)

    def test_empty_zero_and_acceptance_only(self):
        empty = _summarize_scaled(_merge_scaled([]))
        self.assertTrue(all(x is None for section in empty.values() for x in section.values()))
        result = _summarize_scaled(_scientific_moments([((0., 0), None), ((.5, -999), None)]), False)
        self.assertEqual(result['estimates']['acceptance'], 2.**-1001)
        self.assertEqual(result['standard_errors']['acceptance'], 2.**-1001)
        self.assertIsNone(result['estimates']['error'])

    def test_subnormal_mean_and_standard_error_round_at_the_end(self):
        rows = [((.5, -1073), (.5, -1073)), ((0., 0), (0., 0))]
        result = _summarize_scaled(_scientific_moments(rows))
        self.assertEqual(result['estimates'], dict(acceptance=0., error=0., conditional_error=1.))
        self.assertEqual(result['standard_errors'], dict(acceptance=0., error=0., conditional_error=0.))
        rows[0] = ((.5, -1072), (.5, -1072))
        result = _summarize_scaled(_scientific_moments(rows))
        self.assertEqual(result['estimates']['acceptance'], 2.**-1074)
        self.assertEqual(result['standard_errors']['acceptance'], 2.**-1074)


class HistoryStatisticsGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def reduce(self, rows, ranks=None, status=None):
        cp = self.cp
        c, p = arrays(rows)
        ranks = [0]*len(c) if ranks is None else ranks
        exact = SimpleNamespace(coefficients=cp.asarray(c), powers=cp.asarray(p),
            status=cp.zeros(p.shape, dtype=cp.int32) if status is None else cp.asarray(status))
        actual, continuation = _HistoryStatistics(cp).reduce(exact, proposal_rank=cp.asarray(ranks, dtype=cp.int64))
        return actual, continuation, history_scientific_rows(c, p, ranks)

    def assert_summary(self, actual, expected):
        a, e = _summarize_scaled(actual), _summarize_scaled(expected)
        for section in a:
            np.testing.assert_allclose(list(a[section].values()), list(e[section].values()), rtol=3e-13, atol=0)

    def test_affine_weights_and_zero_draws(self):
        rows = [[scalar(i % 3, -5), scalar(i % 2 if i % 3 else 0, -7)] for i in range(301)]
        actual, flags, reference = self.reduce(rows, [4]*len(rows))
        np.testing.assert_array_equal(flags, 0)
        self.assertEqual(actual.count, len(rows))
        self.assert_summary(actual, _scientific_moments(reference))

    def test_tiny_masses_and_large_rank_do_not_underflow(self):
        actual, flags, reference = self.reduce([[scalar(1, -4001), scalar(1, -4002)]]*129, [4000]*129)
        np.testing.assert_array_equal(flags, 0)
        self.assert_summary(actual, _scientific_moments(reference))
        self.assertEqual(_summarize_scaled(actual)['estimates']['acceptance'], .5)

    def test_signed_likelihood_scale_preserves_subnormal_weight(self):
        from magna._cuda_history import _CudaHistory
        history = _CudaHistory.__new__(_CudaHistory)
        history._cp = self.cp
        history.proposal = True
        history.normalization = (1.25, -1)
        for rank, expected in ((0, 2.**-1074), (1074, 1.)):
            ranks, weights = history._statistics_inputs(self.cp.asarray([rank], dtype=self.cp.int64),
                self.cp.asarray([2.**-1074]))
            c, p = arrays([[scalar(5, -3)]])
            exact = SimpleNamespace(coefficients=self.cp.asarray(c), powers=self.cp.asarray(p),
                                    status=self.cp.zeros(p.shape, dtype=self.cp.int32))
            actual, flags = _HistoryStatistics(self.cp).reduce(exact, proposal_rank=ranks, likelihood=weights)
            np.testing.assert_array_equal(flags, 0)
            self.assertEqual(_summarize_scaled(actual, False)['estimates']['acceptance'], expected)

    def test_exact_error_and_pell_cancellation(self):
        x, y = 3, 2
        while x < 1 << 50:
            x, y = 3*x+4*y, 2*x+3*y
        actual, flags, reference = self.reduce([[scalar(), scalar((1 << 55)-1, -55)],
                                               [([x, -y, 0, y], 0), scalar(0)]])
        np.testing.assert_array_equal(flags, 0)
        self.assert_summary(actual, _scientific_moments(reference))
        self.assertGreater(_summarize_scaled(actual)['estimates']['error'], 0)

    def test_unsafe_draw_continues_without_losing_safe_zero_draws(self):
        actual, flags, reference = self.reduce([[scalar(), scalar(1, -2000)],
                                               [scalar(0), scalar(0)],
                                               [scalar(1, -1), scalar(1, -2)]])
        np.testing.assert_array_equal(flags, [1, 0, 0])
        self.assertEqual(actual.count, 2)
        completed = _merge_scaled([actual, _scientific_moments([reference[0]])])
        self.assertEqual(completed.count, 3)
        self.assert_summary(completed, _scientific_moments(reference))

    def test_only_guard_status_replays(self):
        status = np.zeros((130, 1), dtype=np.int32)
        status[128, 0] = 1
        actual, flags, _ = self.reduce([[scalar(1, -1)]]*130, status=status)
        self.assertEqual(int(flags.sum()), 1)
        self.assertEqual(actual.count, 129)
        for code, flag in ((2, 16), (3, 1)):
            status[128, 0] = code
            with self.assertRaisesRegex(ValueError, f'flags={flag}'):
                self.reduce([[scalar(1, -1)]]*130, status=status)


if __name__ == '__main__':
    unittest.main()
