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


def _symmetric_upper(values, n):
    words = (n+63)//64
    rows = values.reshape(n, words).copy()
    for j, row in enumerate(rows.copy()):
        for w, value in enumerate(row):
            bits = int(value)
            while bits:
                k = 64*w + (bits & -bits).bit_length()-1
                rows[k, j//64] |= np.uint64(1) << np.uint64(j % 64)
                bits &= bits-1
    return rows.ravel()


def _query_arrays(raw):
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
    if raw['mode'] in (1, 3, 4):
        host['upper'] = _symmetric_upper(host['upper'], raw['n'])
    return host


class GpuQueries:
    """Compile scalar acceptance/target-success queries for GPU batches.

    Query 0 is acceptance; query 1, when present, is target_success including
    the target trace multiplier. Error is acceptance * target_normalization
    minus target_success. Nonzero device ``status`` must be checked before
    accepting results; the sampler's statistics kernel checks it.
    """
    def __init__(self, experiment, plans=None, *, real_output=True, shared=()):
        self.experiment = experiment
        self._shared = shared
        if plans is None:
            experiment.validate()
            self.handle, self.metadata = _native.make_cuda_queries(experiment.handle)
        else:
            self.handle, self.metadata = None, plans
        self.query_count = len(self.metadata)
        self._host_plans = []
        self._cp = None
        self._plans = None
        self._real_output = real_output
        self._sum_kernel = None
        for raw in self.metadata:
            if raw['mode'] == 6:
                host = {}
            elif raw['mode'] == 5:
                weights, powers = zip(*raw['weights'])
                fits = all(-(1 << 63) < x < (1 << 63) for row in weights for x in row)
                host = dict(components=GpuQueries(experiment, raw['components'], real_output=False, shared=shared),
                    weights=np.asarray(weights, dtype=np.int64) if fits else None,
                    weight_powers=np.asarray(powers, dtype=np.int64))
            else:
                host = _query_arrays(raw)
            self._host_plans.append((raw, host))

    @property
    def cp(self):
        """Upload and compile device plans on first use."""
        if self._cp is None:
            import cupy as cp
            plans = []
            for raw, host in self._host_plans:
                if raw['mode'] == 6:
                    plans.append((raw, {}, None, None))
                    continue
                if raw['mode'] == 5:
                    arrays = dict(components=host['components'],
                        weights=None if host['weights'] is None else cp.asarray(host['weights']),
                        weight_powers=cp.asarray(host['weight_powers']))
                    plans.append((raw, arrays, None, None))
                    if self._sum_kernel is None:
                        module = cp.RawModule(code=_read_cuda_source('query_sum.cu'), options=('--std=c++11',))
                        self._sum_kernel = module.get_function('sum_queries')
                    continue
                defines = dict(N=raw['n'], MODE=raw['mode'], STRIDE=raw['frame_words'],
                               FRAME_BITS=self.experiment.metadata['frame_bits'],
                               PAULI_WORDS=2*self.experiment.words, REAL_OUTPUT=int(self._real_output),
                               FEATURES=raw['feature_count'],
                               SUPPORT=host['support'].size, GUARDS=host['guards'].size,
                               CONSTANT_LINEAR=host['constant_linear'].size // 2,
                               CONSTANT_PAIRS=host['constant_pairs'].size // 2, MAGIC=host['magic'].size // 8,
                               STEPS=host['steps'].size // 3, CONSTANT=raw['constant'],
                               POWER=raw['power_sqrt2'], OUTPUT_POWER=raw['output_power'])
                source = _CUDA
                for key, value in defines.items():
                    source = source.replace('@' + key + '@', str(int(value)))
                name = 'exact_query_large' if raw['mode'] == 4 else 'exact_query'
                module = cp.RawModule(code=source, options=('--std=c++11',),
                                      name_expressions=('query_features', name))
                plans.append((raw, {k: cp.asarray(v) for k, v in host.items()},
                              module.get_function('query_features'), module.get_function(name)))
            self._plans = plans
            self._cp = cp
        return self._cp

    def evaluate(self, device_frames, *, inactive_caps=0):
        """Evaluate every query on device frames from ``FusedCudaNoise``."""
        cp = self.cp
        shots = device_frames.shape[0]
        coefficients = cp.empty((shots, self.query_count, 4), dtype=cp.int64)
        powers = cp.empty((shots, self.query_count), dtype=cp.int32)
        status = cp.empty((shots, self.query_count), dtype=cp.int32)
        if shots:
            for query, (raw, arrays, feature_kernel, kernel) in enumerate(self._plans):
                if raw['mode'] == 6:
                    values = self._shared[raw['base_index']].evaluate(device_frames,
                        inactive_caps=inactive_caps+raw['disabled_caps'])
                    coefficients[:, query] = values.coefficients[:, 0]
                    powers[:, query] = values.powers[:, 0]
                    status[:, query] = values.status[:, 0]
                    continue
                if raw['mode'] == 5:
                    if arrays['weights'] is None:
                        coefficients[:, query].fill(0)
                        powers[:, query].fill(0)
                        status[:, query].fill(1)
                        continue
                    values = arrays['components'].evaluate(device_frames, inactive_caps=inactive_caps)
                    self._sum_kernel(((shots+127)//128,), (128,),
                        (values.coefficients, values.powers, values.status,
                         arrays['weights'], arrays['weight_powers'], coefficients, powers, status,
                         np.uint64(shots), np.int32(values.powers.shape[1]),
                         np.int32(self.query_count), np.int32(query), np.int32(self._real_output)))
                    continue
                count = shots * raw['feature_count']
                features = cp.empty(count, dtype=cp.uint8)
                if count:
                    feature_kernel(((count + 255) // 256,), (256,),
                                   (device_frames, arrays['feature_offsets'], arrays['feature_words'],
                                    arrays['feature_masks'], arrays['feature_constants'], features, np.uint64(shots)))
                arguments = (device_frames, features, arrays['support'], arrays['guards'],
                    arrays['linear'], arrays['upper'], arrays['linear_shift'],
                    arrays['constant_linear'], arrays['constant_pairs'], arrays['magic'], arrays['magic_masks'],
                    arrays['phase_table'], arrays['steps'], arrays['step_offsets'], arrays['step_indices'],
                    arrays['step_tags'], coefficients, powers, status,
                    np.uint64(shots), np.int32(self.query_count), np.int32(query))
                if raw['mode'] == 4:
                    blocks = min(shots, 128)
                    scratch = cp.empty((blocks, raw['n'], (raw['n']+63)//64), dtype=cp.uint64)
                    kernel((blocks,), (128,), (device_frames, features, arrays['support'],
                        arrays['linear'], arrays['upper'], arrays['linear_shift'],
                        arrays['constant_linear'], arrays['constant_pairs'], arrays['magic'], arrays['magic_masks'],
                        coefficients, powers, status, np.uint64(shots), np.int32(self.query_count),
                        np.int32(query), np.int32(raw['n']-inactive_caps), scratch))
                else:
                    kernel(((shots + 127) // 128,), (128,), arguments)
        return ExactQueryResult(coefficients, powers, status)
