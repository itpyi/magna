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
    def test_tiny_likelihood_is_scaled_before_target_normalization(self):
        from magna._cuda_history import _CudaHistory
        history = _CudaHistory.__new__(_CudaHistory)
        history._cp = np
        history.proposal = True
        history.normalization = (1.25, -1)
        ranks, weights = history._statistics_inputs(np.array([0, 1074], dtype=np.int64),
            np.array([2.**-1074, 2.**-1074]))
        np.testing.assert_array_equal(ranks, [-1072, 2])
        np.testing.assert_array_equal(weights, [.4, .4])
        c, p = arrays([[scalar(5, -3), scalar(5, -3)]]*2)
        rows = history_scientific_rows(c, p, proposal_rank=ranks, likelihood=weights)
        for row, expected in zip(rows, (2.**-1074, 1.)):
            self.assertEqual(_summarize_scaled(_scientific_moments([row]), False)['estimates']['acceptance'], expected)

    def test_proposal_and_conditional_normalization(self):
        c, p = arrays([[scalar(1, -3), scalar(1, -4), scalar(1, -6)]])
        exact = history_scientific_rows(c, p, -1)
        affine = history_scientific_rows(c, p, -1, [4], [2.])
        self.assertEqual(_summarize_scaled(_scientific_moments(exact))['estimates'],
                         dict(acceptance=.5, error=.25, conditional_error=.5))
        self.assertEqual(_summarize_scaled(_scientific_moments(affine))['estimates'],
                         dict(acceptance=2., error=1., conditional_error=.5))

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

    def test_tiny_history_and_large_proposal_cancel_before_conversion(self):
        c, p = arrays([[scalar(1, -4000), scalar(1, -4001), scalar(1, -6002)]])
        for ranks in (None, [4000]):
            rows = history_scientific_rows(c, p, -2000, ranks)
            self.assertEqual(_summarize_scaled(_scientific_moments(rows))['estimates'],
                             dict(acceptance=.5, error=.25, conditional_error=.5))

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

    def reduce(self, c, p, norm_power=0, ranks=None, likelihood=None, status=None):
        cp = self.cp
        exact = SimpleNamespace(coefficients=cp.asarray(c), powers=cp.asarray(p),
                                status=cp.zeros(p.shape, dtype=cp.int32) if status is None else cp.asarray(status))
        return _HistoryStatistics(cp).reduce(exact, norm_power=norm_power,
                    proposal_rank=None if ranks is None else cp.asarray(ranks, dtype=cp.int64),
                    likelihood=None if likelihood is None else cp.asarray(likelihood, dtype=cp.float64))

    def test_both_history_weights_and_target_normalization(self):
        c, p = arrays([[scalar(1, -3), scalar(i % 3, -5), scalar(i % 2, -7)]
                       if i % 3 else [scalar(1, -3), scalar(0), scalar(0)] for i in range(301)])
        for ranks in (None, [4]*len(c)):
            weights = np.array([0., .5, 2.]*101)[:len(c)]
            actual, flags = self.reduce(c, p, -1, ranks, weights)
            np.testing.assert_array_equal(flags, 0)
            expected = _scientific_moments(history_scientific_rows(c, p, -1, ranks, weights))
            self.assertEqual(actual.count, len(c))
            a, e = _summarize_scaled(actual), _summarize_scaled(expected)
            for section in a:
                np.testing.assert_allclose(list(a[section].values()), list(e[section].values()), rtol=3e-14, atol=0)

    def test_tiny_masses_and_large_rank_do_not_underflow(self):
        c, p = arrays([[scalar(1, -4000), scalar(1, -4001), scalar(1, -6002)]]*129)
        for ranks in (None, [4000]*len(c)):
            actual, flags = self.reduce(c, p, -2000, ranks)
            np.testing.assert_array_equal(flags, 0)
            result = _summarize_scaled(actual)
            self.assertEqual(result['estimates'], dict(acceptance=.5, error=.25, conditional_error=.5))
            self.assertEqual(result['standard_errors'], dict(acceptance=0., error=0., conditional_error=0.))

    def test_signed_likelihood_scale_preserves_subnormal_weight(self):
        from magna._cuda_history import _CudaHistory
        history = _CudaHistory.__new__(_CudaHistory)
        history._cp = self.cp
        history.proposal = True
        history.normalization = (1.25, -1)
        for rank, expected in ((0, 2.**-1074), (1074, 1.)):
            ranks, weights = history._statistics_inputs(self.cp.asarray([rank], dtype=self.cp.int64),
                self.cp.asarray([2.**-1074]))
            c, p = arrays([[scalar(5, -3), scalar(5, -3)]])
            actual, flags = self.reduce(c, p, ranks=ranks, likelihood=weights)
            np.testing.assert_array_equal(flags, 0)
            self.assertEqual(_summarize_scaled(actual, False)['estimates']['acceptance'], expected)

    def test_exact_error_and_pell_cancellation(self):
        x, y = 3, 2
        while x < 1 << 50:
            x, y = 3*x+4*y, 2*x+3*y
        c, p = arrays([[scalar(), scalar(), scalar((1 << 55)-1, -55)],
                       [scalar(), ([x, -y, 0, y], 0), scalar(0)]])
        actual, flags = self.reduce(c, p)
        np.testing.assert_array_equal(flags, 0)
        expected = _scientific_moments(history_scientific_rows(c, p))
        a, e = _summarize_scaled(actual), _summarize_scaled(expected)
        for section in a:
            np.testing.assert_allclose(list(a[section].values()), list(e[section].values()), rtol=4e-15, atol=0)
        self.assertGreater(a['estimates']['error'], 0)

    def test_unsafe_draw_continues_without_losing_safe_zero_draws(self):
        c, p = arrays([[scalar(), scalar(), scalar(1, -2000)],
                       [scalar(), scalar(0), scalar(0)],
                       [scalar(), scalar(1, -1), scalar(1, -2)]])
        actual, flags = self.reduce(c, p)
        np.testing.assert_array_equal(flags, [1, 0, 0])
        self.assertEqual(actual.count, 2)
        rows = history_scientific_rows(c, p)
        completed = _merge_scaled([actual, _scientific_moments([rows[0]])])
        self.assertEqual(completed.count, 3)
        expected = _summarize_scaled(_scientific_moments(rows))
        for section, values in _summarize_scaled(completed).items():
            np.testing.assert_allclose(list(values.values()), list(expected[section].values()), rtol=3e-15, atol=0)

    def test_query_status_excludes_only_the_affected_draw(self):
        c, p = arrays([[scalar(), scalar(1, -1)]]*130)
        status = np.zeros(p.shape, dtype=np.int32)
        status[128, 1] = 2
        actual, flags = self.reduce(c, p, status=status)
        self.assertEqual(int(flags.sum()), 1)
        self.assertEqual(actual.count, 129)
        self.assertEqual(_summarize_scaled(actual, False)['estimates']['acceptance'], .5)
