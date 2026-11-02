"""Fused noise sampling and Pauli-frame accumulation on CUDA.

Noise probabilities use magna's compiled uint64 thresholds, including
sequential conditional coins for nonuniform categorical channels.
The GPU uses Philox4x32-10, not the mt19937_64 stream of the CPU sampler.
Every random block is addressed by (global shot ID, per-shot block counter).

Importance proposals are supported only when the native compiler builds a
single complete count table. Use ``sample_weighted`` for those plans: raw
conditional probabilities must be multiplied by the returned likelihoods.
"""
from operator import index as integer_index

import numpy as np

from ._cuda_source import _read_cuda_source

from . import _native


MASK64 = (1 << 64) - 1


def _uint64(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    try:
        value = integer_index(value)
    except TypeError:
        raise TypeError(f'{name} must be an integer') from None
    if value < 0 or value > MASK64:
        raise ValueError(f'{name} must fit uint64')
    return value


_SOURCE = _read_cuda_source("noise_sampling.cu")


class FusedCudaNoise:
    """GPU noise RNG and complete-frame XOR in one kernel.

    ``sample(shots, seed, offset=0)`` is asynchronous and returns a CuPy uint64
    array shaped ``(shots, experiment.width)``. It can feed the GPU query kernel
    directly.
    """
    _types = dict(thresholds='<u8', certain='u1', block_offsets='<u4', block_sites='<u4',
                  choice_offsets='<u4', choices='<u4', uniform='u1',
                  conditional_thresholds='<u8', conditional_certain='u1')

    def __init__(self, experiment):
        experiment.validate()
        self.experiment = experiment
        self.metadata = _native.cuda_noise_metadata(experiment.handle)
        self.tables = {name: np.frombuffer(self.metadata[name], dtype=dtype)
                       for name, dtype in self._types.items()}
        self.width = experiment.width
        self.sites, self.blocks = int(self.metadata['sites']), int(self.metadata['blocks'])
        self.has_importance_weights = bool(self.metadata['has_importance_weights'])
        self.weighted_sites = np.frombuffer(self.metadata['weighted_sites'], dtype='u1')
        self.likelihood_values = np.frombuffer(self.metadata['likelihood_values'], dtype='<f8')
        # Keep GPU initialization lazy: metadata requires no CUDA.
        self._cp = None

    @property
    def cp(self):
        if self._cp is None:
            import cupy as cp
            self._cp = cp
            self._columns = cp.asarray(self.experiment.columns)
            self._base = cp.asarray(self.experiment.base_frame)
            self._tables = {name: cp.asarray(value) for name, value in self.tables.items()}
            self._weighted_sites = cp.asarray(self.weighted_sites)
            self._likelihood_values = cp.asarray(self.likelihood_values)
            module = cp.RawModule(code=_SOURCE, options=('--std=c++11',))
            self._kernel = module.get_function('fused_noise')
            self._dummy_weights = cp.empty(1, dtype=cp.float64)
        return self._cp

    def _arguments(self, shots, seed, offset):
        shots, seed, offset = (_uint64(shots, 'shots'), _uint64(seed, 'seed'),
                               _uint64(offset, 'offset'))
        if shots > self.experiment.metadata['max_shots']:
            raise ValueError('noise batch exceeds the native shot limit')
        if shots and offset > MASK64 - (shots - 1):
            raise ValueError('global shot ID overflows uint64')
        return shots, seed, offset

    def sample(self, shots, seed, *, offset=0):
        """Enqueue device noise/frames; does not synchronize or copy to host."""
        if self.has_importance_weights:
            raise ValueError('importance draws require sample_weighted so likelihoods are not discarded')
        return self._sample(shots, seed, offset, weighted=False)

    def sample_weighted(self, shots, seed, *, offset=0):
        """Return device ``(frames, likelihoods)`` without a host transfer.

        The float64 likelihood is the native count table's rounded exact
        target/proposal ratio. It is not a probability and can exceed one.
        Physical/no-op proposals return unit weights.
        """
        return self._sample(shots, seed, offset, weighted=True)

    def _sample(self, shots, seed, offset, *, weighted):
        shots, seed, offset = self._arguments(shots, seed, offset)
        cp = self.cp
        frames = cp.empty((shots, self.width), dtype=cp.uint64)
        likelihoods = cp.empty(shots, dtype=cp.float64) if weighted else self._dummy_weights
        if shots:
            args = (self._columns, self._base, *(self._tables[k] for k in self._types),
                    self._weighted_sites, self._likelihood_values, frames, likelihoods,
                    np.uint64(shots), np.uint64(seed), np.uint64(offset),
                    np.uint32(self.width), np.uint32(self.blocks), np.int32(weighted))
            self._kernel(((shots + 127) // 128,), (128,), args)
        return (frames, likelihoods) if weighted else frames
