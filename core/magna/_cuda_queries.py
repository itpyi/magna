"""Exact device-side scalar queries for fixed magna probability networks.

The compiler exports the boundary support and frame-dependent Clifford
updates. Small quadratic forms use an exact binary Gauss sum. Large forms
use a prepared elimination only after an algebraic proof that every noise
column preserves its geometry; device guards reject other geometries.
Residual non-Clifford tensors of at most 12 variables are summed into eight
integer phase bins. No sampled truth table or floating contraction is used.
"""
from dataclasses import dataclass
import numpy as np

from ._cuda_source import _read_cuda_source
from . import _native


_CUDA = _read_cuda_source("probability_queries.cu")


@dataclass
class ExactQueryResult:
    """Device arrays of exact query values; a nonzero ``status`` rejects a value."""
    coefficients: object
    powers: object
    status: object


class GpuQueries:
    """Compile scalar acceptance/target-success queries for GPU batches.

    Query 0 is acceptance; query 1, when present, is target_success including
    the target trace multiplier. Error is acceptance * target_normalization
    minus target_success. Nonzero device ``status`` must be checked before
    accepting results; the sampler's statistics kernel checks it.
    """
    def __init__(self, experiment):
        experiment.validate()
        self.experiment = experiment
        self.handle, self.metadata = _native.make_cuda_queries(experiment.handle)
        self.query_count = len(self.metadata)
        self._host_plans = []
        self._cp = None
        self._plans = None
        for raw in self.metadata:
            host = {}
            for name in ('feature_offsets', 'feature_words', 'support', 'guards', 'linear_shift',
                         'step_offsets', 'step_indices'):
                host[name] = np.frombuffer(raw[name], dtype='<u4')
            for name in ('feature_masks', 'magic_masks', 'upper'):
                host[name] = np.frombuffer(raw[name], dtype='<u8')
            for name in ('feature_constants', 'linear', 'phase_table', 'step_tags'):
                host[name] = np.frombuffer(raw[name], dtype='u1')
            for name in ('constant_linear', 'constant_pairs', 'magic', 'steps'):
                host[name] = np.frombuffer(raw[name], dtype='<i4')
            # The CUDA elimination works with symmetric adjacency rows.
            if raw['mode'] in (1, 3):
                adj = host['upper'].copy()
                for j, row in enumerate(list(map(int, adj))):
                    while row:
                        k = (row & -row).bit_length() - 1
                        adj[k] |= np.uint64(1) << np.uint64(j)
                        row &= row - 1
                host['upper'] = adj
            self._host_plans.append((raw, host))

    @property
    def cp(self):
        """Upload and compile device plans on first use."""
        if self._cp is None:
            import cupy as cp
            plans = []
            for raw, host in self._host_plans:
                defines = dict(N=raw['n'], MODE=raw['mode'], STRIDE=raw['frame_words'],
                               FRAME_BITS=self.experiment.metadata['frame_bits'], FEATURES=raw['feature_count'],
                               SUPPORT=host['support'].size, GUARDS=host['guards'].size,
                               CONSTANT_LINEAR=host['constant_linear'].size // 2,
                               CONSTANT_PAIRS=host['constant_pairs'].size // 2, MAGIC=host['magic'].size // 8,
                               STEPS=host['steps'].size // 3, CONSTANT=raw['constant'],
                               POWER=raw['power_sqrt2'], OUTPUT_POWER=raw['output_power'])
                source = _CUDA
                for key, value in defines.items():
                    source = source.replace('@' + key + '@', str(int(value)))
                module = cp.RawModule(code=source, options=('--std=c++11',),
                                      name_expressions=('query_features', 'exact_query'))
                plans.append((raw, {k: cp.asarray(v) for k, v in host.items()},
                              module.get_function('query_features'), module.get_function('exact_query')))
            self._plans = plans
            self._cp = cp
        return self._cp

    def evaluate(self, device_frames):
        """Evaluate every query on device frames from ``FusedCudaNoise``."""
        cp = self.cp
        shots = device_frames.shape[0]
        coefficients = cp.empty((shots, self.query_count, 4), dtype=cp.int64)
        powers = cp.empty((shots, self.query_count), dtype=cp.int32)
        status = cp.empty((shots, self.query_count), dtype=cp.int32)
        if shots:
            for query, (raw, arrays, feature_kernel, kernel) in enumerate(self._plans):
                count = shots * raw['feature_count']
                features = cp.empty(count, dtype=cp.uint8)
                if count:
                    feature_kernel(((count + 255) // 256,), (256,),
                                   (device_frames, arrays['feature_offsets'], arrays['feature_words'],
                                    arrays['feature_masks'], arrays['feature_constants'], features, np.uint64(shots)))
                kernel(((shots + 127) // 128,), (128,),
                       (device_frames, features, arrays['support'], arrays['guards'],
                        arrays['linear'], arrays['upper'], arrays['linear_shift'],
                        arrays['constant_linear'], arrays['constant_pairs'], arrays['magic'], arrays['magic_masks'],
                        arrays['phase_table'], arrays['steps'], arrays['step_offsets'], arrays['step_indices'],
                        arrays['step_tags'], coefficients, powers, status,
                        np.uint64(shots), np.int32(self.query_count), np.int32(query)))
        return ExactQueryResult(coefficients, powers, status)
