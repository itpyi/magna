"""Scaled statistics for completed CUDA histories."""
from dataclasses import dataclass

import numpy as np

from . import _native
from ._cuda_source import _read_cuda_source


@dataclass(frozen=True)
class _ScaledMoments:
    count: int
    mean: np.ndarray
    root: np.ndarray
    scales: np.ndarray


def _empty():
    return _ScaledMoments(0, np.zeros(2), np.zeros(3), np.zeros(2, dtype=np.int64))


def _packed(moments):
    return moments.count, moments.mean.tobytes(), moments.root.tobytes(), moments.scales.tobytes()


def _merge_scaled(chunks):
    total = _empty()
    for chunk in chunks:
        count, mean, root, scales = _native._cuda_history_merge(_packed(total), _packed(chunk))
        total = _ScaledMoments(count, np.frombuffer(mean, dtype='<f8'), np.frombuffer(root, dtype='<f8'),
                               np.frombuffer(scales, dtype='<i8'))
    return total


def _scientific_moments(rows):
    return _merge_scaled(_ScaledMoments(1, np.array([row[0][0], (row[1] or (0., 0))[0]], dtype=float),
                                       np.zeros(3), np.array([row[0][1], (row[1] or (0., 0))[1]], dtype=np.int64))
                         for row in rows)


def _summarize_scaled(moments, has_error=True):
    return _native._cuda_history_summary(_packed(moments), has_error)


class _HistoryStatistics:
    def __init__(self, cp):
        self.cp = cp
        self.module = None

    def reduce(self, exact, *, proposal_rank, likelihood=None):
        cp = self.cp
        count, queries = exact.powers.shape
        if not count:
            return _empty(), np.empty(0, dtype=np.uint8)
        if self.module is None:
            self.module = cp.RawModule(code=_read_cuda_source('history_statistics.cu'),
                                       options=('--std=c++11',))
        weights = likelihood if likelihood is not None else cp.empty(1, dtype=cp.float64)
        blocks = (count+127)//128
        partial = cp.empty((blocks, 8), dtype=cp.float64)
        continuation = cp.empty(count, dtype=cp.uint8)
        output = cp.empty(8, dtype=cp.float64)
        flags = cp.zeros(1, dtype=cp.uint32)
        self.module.get_function('history_moments')((blocks,), (128,),
            (exact.coefficients, exact.powers, exact.status, proposal_rank, weights,
             np.int32(likelihood is not None), np.uint64(count), np.int32(queries), partial, continuation, flags))
        self.module.get_function('history_finish')((1,), (128,),
            (partial, np.uint64(blocks), output))
        flag = int(flags.get()[0])
        if flag:
            raise ValueError(f'GPU probability summary rejected values (flags={flag}: '
                             '1=nonreal, 16=query contract)')
        host = output.get()
        return (_ScaledMoments(int(host[5]), host[:2], host[2:5], host[6:8].astype(np.int64)),
                continuation.get())
