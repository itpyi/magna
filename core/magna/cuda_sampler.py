"""Physical and importance sampling of supported tasks on an NVIDIA GPU.

Compilation stays in magna. Philox noise, Pauli-frame accumulation, exact
conditional probabilities and batch statistics run on the device; each chunk
copies six numbers back to the host. Each chunk's statistics are a mean and
a centered covariance factor, merged on the host without subtracting raw
moments, so a small variance is not lost to cancellation.

Likelihoods multiply joint contributions; means are divided by the number of
attempted draws, never by the sum of likelihoods. Individual probabilities are
exact; statistics are float64 estimates. Values outside the supported
numerical range raise an error rather than being clipped.
"""
from dataclasses import dataclass
import math
from operator import index
from time import perf_counter

import numpy as np

from ._cuda_source import _read_cuda_source
from ._cuda_experiment import CudaExperiment
from ._cuda_noise import FusedCudaNoise, MASK64, _uint64
from ._cuda_queries import GpuQueries

_MAX_LIKELIHOOD = math.ldexp(1., 480)

_STATISTICS_SOURCE = _read_cuda_source("statistics.cu")


@dataclass(frozen=True)
class _Moments:
    """Means of ``(acceptance, error)`` and an upper-triangular factor.

    The factor ``[r00, r01, r11]`` satisfies
    ``R.T @ R = sum((x-mean)(x-mean).T)`` for ``x=(acceptance,error)``.
    It is kept as a square root because squaring it could underflow even
    when the standard error is representable.
    """
    count: int
    mean: np.ndarray
    root: np.ndarray


_EMPTY = _Moments(0, np.zeros(2), np.zeros(3))


def _run_arguments(shots, seed, offset, batch_size, max_batch_size=1_000_000):
    """Shared bounds for single-device and distributed draw intervals."""
    shots, seed, offset = _uint64(shots, 'shots'), _uint64(seed, 'seed'), _uint64(offset, 'offset')
    if shots and offset > MASK64-(shots-1):
        raise ValueError('global shot ID overflows uint64')
    if isinstance(batch_size, (bool, np.bool_)):
        raise TypeError('batch_size must be an integer')
    batch_size = index(batch_size)
    if not 1 <= batch_size <= max_batch_size:
        raise ValueError('batch_size outside supported range')
    if shots > (1 << 53):
        raise ValueError('summary shot count exceeds exact float64 count range')
    return shots, seed, offset, batch_size


def _fma(a, b, c):
    if hasattr(math, 'fma'):
        return math.fma(a, b, c)
    # Python 3.11-3.12 lack math.fma. This exact host fallback is used only
    # for a few covariance-factor operations per batch, never per shot.
    from fractions import Fraction
    return float(Fraction(a)*Fraction(b)+Fraction(c))


def _add_covariance_row(factor, x, y):
    """Scalar QR rank update; called per chunk, never once per shot on CPU."""
    r00, r01, r11 = map(float, factor)
    if x == 0:
        return np.array([r00, r01, math.hypot(r11, y)])
    radius = math.hypot(r00, x)
    cosine, sine = r00/radius, x/radius
    xy = sine*r01
    residual = _fma(cosine,y,-xy)-_fma(sine,r01,-xy)
    yy = sine*y
    middle = _fma(cosine,r01,yy)+_fma(sine,y,-yy)
    return np.array([radius, middle, math.hypot(r11, residual)])


def _merge_moments(chunks):
    """Chan mean/covariance merge, with stable square-root rank updates."""
    count, means, factor = 0, np.zeros(2), np.zeros(3)
    for chunk in chunks:
        other_n = chunk.count
        if not other_n:
            continue
        if not count:
            count, means, factor = other_n, chunk.mean.copy(), chunk.root.copy()
            continue
        delta = chunk.mean-means
        total = count+other_n
        weight = math.sqrt(count*other_n/total)
        other = chunk.root
        factor = _add_covariance_row(factor, float(other[0]), float(other[1]))
        factor = _add_covariance_row(factor, 0., float(other[2]))
        factor = _add_covariance_row(factor, weight*float(delta[0]), weight*float(delta[1]))
        means += delta*(other_n/total)
        count = total
    return _Moments(count, means, factor)


def _summarize_moments(moments, has_error=True):
    """Estimates and IID standard errors from merged centered statistics."""
    count = moments.count
    if not count:
        return dict(estimates=dict(acceptance=None, error=None, conditional_error=None),
                    standard_errors=dict(acceptance=None, error=None, conditional_error=None))
    a, e = map(float, moments.mean)
    ratio = e/a if has_error and a else None
    sa_error = se_error = ratio_error = None
    if count > 1:
        scale = math.sqrt(count*(count-1))
        r00, r01, r11 = map(float, moments.root)
        sa_error = r00/scale
        if has_error:
            se_error = math.hypot(r01, r11)/scale
            if ratio is not None:
                ratio_error = math.hypot(_fma(-ratio, r00, r01), r11)/scale/abs(a)
    return dict(estimates=dict(acceptance=a, error=e if has_error else None, conditional_error=ratio),
                standard_errors=dict(acceptance=sa_error, error=se_error, conditional_error=ratio_error))


class CudaSampler:
    """Sample a compiled task on an NVIDIA GPU.

    ``CudaSampler(compiled)`` raises ``ValueError`` when the task is not
    supported. Do not change the compiled task's options afterwards.
    """
    def __init__(self, compiled):
        self._experiment = CudaExperiment(compiled)
        self._noise = FusedCudaNoise(self._experiment)
        if np.any(self._noise.likelihood_values > _MAX_LIKELIHOOD):
            raise ValueError('importance likelihood table exceeds the 2^480 statistics bound')
        self._queries = GpuQueries(self._experiment)
        coefficients, power = self._experiment.metadata['target_normalization']
        if tuple(coefficients) != (1, 0, 0, 0):
            raise ValueError('CUDA sampler requires the ideal state postselection probability to be a power of two')
        self._norm_power = int(power)
        if abs(self._norm_power) > 1_000_000:
            raise ValueError('target normalization exponent exceeds summary bound')
        self._has_error = compiled.ideal_state is not None
        self._statistics_module = None
        self._unit_weights = None

    def _gpu_moments(self, exact, weights=None):
        cp = self._noise.cp
        c, p = exact.coefficients, exact.powers
        if weights is None:
            if self._unit_weights is None:
                self._unit_weights = cp.empty(1, dtype=cp.float64)
            weights = self._unit_weights
        if not c.shape[0]:
            return _EMPTY
        if self._statistics_module is None:
            self._statistics_module = cp.RawModule(code=_STATISTICS_SOURCE, options=('--std=c++11',))
        first = self._statistics_module.get_function('scalar_moments')
        finish = self._statistics_module.get_function('finish_moments')
        blocks = (c.shape[0]+127)//128
        partial = cp.empty((blocks, 5), dtype=cp.float64)
        flags = cp.zeros(1, dtype=cp.uint32)
        output = cp.empty(6, dtype=cp.float64)
        first((blocks,), (128,), (c, p, exact.status, weights, np.int32(weights is not self._unit_weights),
                                  np.uint64(c.shape[0]), np.int32(c.shape[1]),
                                  np.int32(self._norm_power), partial, flags))
        finish((1,), (128,), (partial, np.uint64(blocks), np.uint64(c.shape[0]), flags, output))
        # output: two means, three factor entries, flags.
        host = output.get()
        if host[5]:
            raise ValueError(f'GPU probability summary rejected values (flags={int(host[5])}: '
                             '1=nonreal, 2=integer bound, 4=probability bound, 8=float range, 16=query contract, 32=likelihood)')
        return _Moments(c.shape[0], host[0:2], host[2:5])

    def _chunks(self, shots, seed, offset, batch_size):
        """Statistics of each chunk, produced as it finishes so memory stays constant."""
        for first in range(0, shots, batch_size):
            size = min(batch_size, shots-first)
            weights = None
            if self._noise.has_importance_weights:
                frames, weights = self._noise.sample_weighted(size, seed, offset=offset+first)
            else:
                frames = self._noise.sample(size, seed, offset=offset+first)
            yield self._gpu_moments(self._queries.evaluate(frames), weights)

    def run(self, shots, *, seed=0, offset=0, batch_size=65536):
        """Attempt ``shots`` draws in chunks of at most ``batch_size``.

        Draw ``j`` uses Philox random numbers addressed by ``(seed, offset + j)``,
        so the chunk size does not change which faults are drawn.
        """
        self._experiment.validate()
        shots, seed, offset, batch_size = _run_arguments(
            shots, seed, offset, batch_size, self._experiment.metadata['max_shots'])
        start = perf_counter()
        totals = _merge_moments(self._chunks(shots, seed, offset, batch_size))
        return dict(status='complete', requested=shots, completed=shots,
                    seed=seed, offset=offset, batch_size=batch_size,
                    timings=dict(sampling_seconds=perf_counter()-start),
                    **_summarize_moments(totals, self._has_error))
