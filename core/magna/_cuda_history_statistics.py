"""Scaled statistics for completed CUDA histories."""
from dataclasses import dataclass
import math

import numpy as np

from ._cuda_source import _read_cuda_source


@dataclass(frozen=True)
class _ScaledMoments:
    count: int
    mean: np.ndarray
    root: np.ndarray
    scales: np.ndarray


def _empty():
    return _ScaledMoments(0, np.zeros(2), np.zeros(3), np.zeros(2, dtype=np.int64))


def _merge_scaled(chunks):
    from .cuda_sampler import _Moments, _merge_moments
    total = _empty()
    for chunk in chunks:
        if not chunk.count:
            continue
        if not total.count:
            total = chunk
            continue
        scales = np.array([int(chunk.scales[j]) if total.mean[j] == 0 else
                           int(total.scales[j]) if chunk.mean[j] == 0 else
                           max(int(total.scales[j]), int(chunk.scales[j])) for j in range(2)],
                          dtype=np.int64)
        def rescale(item):
            shifts = [int(item.scales[j])-int(scales[j]) for j in range(2)]
            def scaled(value, j):
                return 0. if value == 0 or shifts[j] < -(1 << 31) else math.ldexp(float(value), shifts[j])
            return _Moments(item.count, np.array([scaled(item.mean[j], j) for j in range(2)]),
                            np.array([scaled(item.root[0], 0), scaled(item.root[1], 1),
                                      scaled(item.root[2], 1)]))
        merged = _merge_moments((rescale(total), rescale(chunk)))
        total = _ScaledMoments(merged.count, merged.mean, merged.root, scales)
    return total


def _scientific_moments(rows):
    return _merge_scaled(_ScaledMoments(1, np.array([row[0][0], (row[1] or (0., 0))[0]], dtype=float),
                                       np.zeros(3), np.array([row[0][1], (row[1] or (0., 0))[1]], dtype=np.int64))
                         for row in rows)


def _summarize_scaled(moments, has_error=True):
    from .cuda_sampler import _Moments, _summarize_moments
    result = _summarize_moments(_Moments(moments.count, moments.mean, moments.root), has_error)
    powers = (int(moments.scales[0]), int(moments.scales[1]),
              int(moments.scales[1])-int(moments.scales[0]))
    for section in result.values():
        for name, power in zip(('acceptance', 'error', 'conditional_error'), powers):
            value = section[name]
            if value is None or value == 0:
                continue
            try:
                restored = 0. if power < -(1 << 31) else math.ldexp(value, power)
            except OverflowError:
                raise ValueError('CUDA history summary exceeds float64 range') from None
            if not math.isfinite(restored):
                raise ValueError('CUDA history summary exceeds float64 range')
            section[name] = restored
    return result


class _HistoryStatistics:
    def __init__(self, cp):
        self.cp = cp
        self.module = None

    def reduce(self, exact, *, norm_power=0, proposal_rank=None, likelihood=None):
        cp = self.cp
        count, queries = exact.powers.shape
        if not count:
            return _empty(), np.empty(0, dtype=np.uint8)
        if self.module is None:
            self.module = cp.RawModule(code=_read_cuda_source('history_statistics.cu'),
                                       options=('--std=c++11',))
        ranks = proposal_rank if proposal_rank is not None else cp.empty(1, dtype=cp.int64)
        weights = likelihood if likelihood is not None else cp.empty(1, dtype=cp.float64)
        blocks = (count+127)//128
        partial = cp.empty((blocks, 8), dtype=cp.float64)
        continuation = cp.empty(count, dtype=cp.uint8)
        output = cp.empty(8, dtype=cp.float64)
        self.module.get_function('history_moments')((blocks,), (128,),
            (exact.coefficients, exact.powers, exact.status, ranks, weights,
             np.int32(proposal_rank is not None), np.int32(likelihood is not None),
             np.uint64(count), np.int32(queries), np.int64(norm_power), partial, continuation))
        self.module.get_function('history_finish')((1,), (128,),
            (partial, np.uint64(blocks), output))
        host = output.get()
        return (_ScaledMoments(int(host[5]), host[:2], host[2:5], host[6:8].astype(np.int64)),
                continuation.get())
