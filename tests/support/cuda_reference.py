"""CPU and Python references for the CUDA sampler's tests.

The native references are private bindings of ``magna._native``; the Python
ones here are independent oracles. None of them is part of the package.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from magna import _native
from magna._cuda_noise import MASK64
from magna._cuda_queries import ExactQueryResult
from magna.cuda_sampler import _MAX_LIKELIHOOD, _Moments, _merge_moments, _summarize_moments

MASK32 = (1 << 32) - 1


# Sparse noise batches and the original native evaluator.

def _unsigned(values, dtype, ndim, name):
    """Convert integer input without silently wrapping negatives or floats."""
    if isinstance(values, (list, tuple)):
        # NumPy can infer float64 for a Python sequence mixing small integers
        # and values above INT64_MAX.  Inspect objects before any conversion,
        # otherwise those uint64 RNG seeds lose their low bits.
        array = np.asarray(values, dtype=object)
        if array.ndim != ndim:
            raise ValueError(f'{name} must have {ndim} dimensions')
        maximum = np.iinfo(dtype).max
        for value in array.flat:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f'{name} must contain unsigned integers')
            if int(value) < 0 or int(value) > maximum:
                raise ValueError(f'{name} integer outside {np.dtype(dtype).name}')
        return np.ascontiguousarray(array, dtype=dtype)
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions')
    if array.dtype.kind not in 'iu':
        raise TypeError(f'{name} must contain unsigned integers')
    if array.size:
        if array.dtype.kind == 'i' and np.any(array < 0):
            raise ValueError(f'{name} contains a negative integer')
        if int(array.max()) > np.iinfo(dtype).max:
            raise ValueError(f'{name} integer exceeds {np.dtype(dtype).name}')
    return np.ascontiguousarray(array, dtype=dtype)


def validate_csr(offsets, indices, actions):
    """Validate and canonicalize sparse shot rows."""
    offsets = _unsigned(offsets, np.uint64, 1, 'offsets')
    indices = _unsigned(indices, np.uint32, 1, 'indices')
    if not offsets.size or offsets[0] != 0:
        raise ValueError('offsets must start with zero')
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('offsets must be nondecreasing')
    if int(offsets[-1]) != indices.size:
        raise ValueError('final offset must equal the number of indices')
    if indices.size and int(indices.max()) >= actions:
        raise ValueError('action index outside the prepared map')
    return offsets, indices


@dataclass(frozen=True)
class NoiseBatch:
    """Sparse native noise draws tied to one experiment."""
    owner: object = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    indices: np.ndarray = field(repr=False)

    @property
    def shots(self):
        return self.offsets.size - 1


def explicit(experiment, rows):
    offsets, indices = [0], []
    for row in rows:
        indices.extend(row)
        offsets.append(len(indices))
    return NoiseBatch(experiment, np.array(offsets, dtype=np.uint64), np.array(indices, dtype=np.uint32))


def action_sites(experiment):
    site_offsets = np.asarray(experiment.metadata['offsets'], dtype=np.uint64)
    return np.searchsorted(site_offsets, np.arange(experiment.actions, dtype=np.uint64), side='right') - 1


def _csr(experiment, batch):
    experiment.validate()
    if not isinstance(batch, NoiseBatch) or batch.owner is not experiment:
        raise ValueError('noise batch belongs to a different experiment')
    offsets, indices = validate_csr(batch.offsets, batch.indices, experiment.actions)
    if offsets.size - 1 > experiment.metadata['max_shots']:
        raise ValueError('noise batch exceeds the native shot limit')
    if indices.size > 1:
        sites = action_sites(experiment)[indices]
        invalid = sites[1:] <= sites[:-1]
        # Comparisons straddling separate shots are unconstrained.  Empty
        # shots may repeat an offset, which is harmless here.
        starts = offsets[1:-1]
        starts = starts[(starts > 0) & (starts < indices.size)]
        invalid[(starts - 1).astype(np.intp)] = False
        if np.any(invalid):
            raise ValueError('actions must be site-ordered with at most one outcome per site')
    return offsets, indices


def draw(experiment, seeds):
    """Draw native noise with an explicit independent uint64 seed per shot."""
    experiment.validate()
    seeds = _unsigned(seeds, np.uint64, 1, 'seeds')
    offsets, indices = _native._cuda_draw_noise(experiment.handle, seeds.tolist())
    return NoiseBatch(experiment, np.frombuffer(offsets, dtype='<u8'), np.frombuffer(indices, dtype='<u4'))


def cpu_frames(experiment, batch):
    """Map a noise batch to Pauli frames with the native C++ XOR kernel."""
    offsets, indices = _csr(experiment, batch)
    frames = _native._cuda_map_frames(experiment.handle, offsets.tobytes(), indices.tobytes())
    return np.frombuffer(frames, dtype='<u8').reshape(batch.shots, experiment.width)


def evaluate_frames(experiment, frames):
    """Evaluate complete frames with magna's native exact kernel.

    Rows are ``(history, acceptance, error_or_None)`` in magna's
    ``((c0,c1,c2,c3), power)`` scalar representation.
    """
    experiment.validate()
    frames = _unsigned(frames, np.uint64, 2, 'frames')
    if frames.shape[1] != experiment.width:
        raise ValueError('frame width differs from the compiled plan')
    return _native._cuda_evaluate_frames(experiment.handle, frames.tobytes())


def evaluate_noise(experiment, batch):
    """Original native execution from the same noise, for exact comparison."""
    offsets, indices = _csr(experiment, batch)
    return _native._cuda_evaluate_noise(experiment.handle, offsets.tobytes(), indices.tobytes())


# Philox and the fused noise sampler.

def philox4x32(counter, key):
    """Independent Python arithmetic oracle for the standard 10-round PRNG.

    Known-answer vectors: DEShawResearch/random123 tests/kat_vectors.
    https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    """
    a, b, c, d = map(int, counter)
    k0, k1 = map(int, key)
    if any(v < 0 or v > MASK32 for v in (a, b, c, d, k0, k1)):
        raise ValueError('Philox counter/key words must be uint32')
    for _ in range(10):
        p0, p1 = 0xD2511F53 * a, 0xCD9E8D57 * c
        a, b, c, d = ((p1 >> 32) ^ b ^ k0, p1 & MASK32,
                      (p0 >> 32) ^ d ^ k1, p0 & MASK32)
        k0, k1 = (k0 + 0x9E3779B9) & MASK32, (k1 + 0xBB67AE85) & MASK32
    return a, b, c, d


class PhiloxStream:
    def __init__(self, seed, shot):
        self.seed, self.shot, self.block, self.saved = seed, shot, 0, None

    def __call__(self):
        if self.saved is not None:
            value, self.saved = self.saved, None
            return value
        a, b, c, d = philox4x32((self.shot & MASK32, self.shot >> 32,
                                self.block & MASK32, self.block >> 32),
                               (self.seed & MASK32, self.seed >> 32))
        self.block += 1
        self.saved = c | (d << 32)
        return a | (b << 32)


def _cpu_sample(noise, shots, seed, offset, return_noise, weighted):
    shots, seed, offset = noise._arguments(shots, seed, offset)
    raw = _native._cuda_noise_cpu_sample(noise.experiment.handle, shots, seed, offset, return_noise, weighted)
    if weighted:
        if return_noise:
            frames, weights, starts, indices = raw
        else:
            frames, weights = raw
    elif return_noise:
        frames, starts, indices = raw
    else:
        frames = raw
    output = (np.frombuffer(frames, dtype='<u8').reshape(shots, noise.width),)
    if weighted:
        output += (np.frombuffer(weights, dtype='<f8'),)
    if return_noise:
        output += (NoiseBatch(noise.experiment, np.frombuffer(starts, dtype='<u8'),
                              np.frombuffer(indices, dtype='<u4')),)
    return output if len(output) > 1 else output[0]


def cpu_sample(noise, shots, seed, *, offset=0, return_noise=False):
    """Compiled C++ scalar implementation of the same Philox/fusion algorithm."""
    if noise.has_importance_weights:
        raise ValueError('importance draws require cpu_sample_weighted so likelihoods are not discarded')
    return _cpu_sample(noise, shots, seed, offset, return_noise, weighted=False)


def cpu_sample_weighted(noise, shots, seed, *, offset=0, return_noise=False):
    """Return host ``(frames, likelihoods[, NoiseBatch])`` from the C++ reference."""
    return _cpu_sample(noise, shots, seed, offset, return_noise, weighted=True)


def weights_for_noise(noise, batch):
    """Count selected weighted sites and apply the native likelihood table."""
    offsets, indices = _csr(noise.experiment, batch)
    flags = noise.weighted_sites[action_sites(noise.experiment)[indices]]
    prefix = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(flags, dtype=np.uint64)])
    counts = prefix[offsets[1:].astype(np.intp)] - prefix[offsets[:-1].astype(np.intp)]
    return noise.likelihood_values[counts.astype(np.intp)]


def noise_to_actions(noise, batch):
    """Dense ``(shots, sites)`` action IDs of a noise batch; MASK32 means no error."""
    offsets, indices = _csr(noise.experiment, batch)
    actions = np.full((batch.shots, noise.sites), MASK32, dtype=np.uint32)
    shots = np.repeat(np.arange(batch.shots), np.diff(offsets).astype(np.intp))
    actions[shots, action_sites(noise.experiment)[indices]] = indices
    return actions


def oracle_sample(noise, shots, seed, *, offset=0):
    """Slow independent Python Philox/coin oracle, for small batches."""
    shots, seed, offset = noise._arguments(shots, seed, offset)
    if shots > 4096:
        raise ValueError('Python oracle is limited to 4096 shots; use the C++ reference')
    t = noise.tables
    experiment = noise.experiment
    frames = np.tile(experiment.base_frame, (shots, 1)).copy()
    offsets, indices = [0], []
    for shot in range(shots):
        rng = PhiloxStream(seed, offset + shot)
        chosen = []
        def coin(threshold, certain):
            return bool(certain) or (bool(threshold) and rng() < int(threshold))
        def select(site):
            first, end = map(int, t['choice_offsets'][site:site+2])
            if t['uniform'][site]:
                n = end - first
                at = 0
                if n != 1:
                    reject = ((1 << 64) - n) % n
                    value = rng()
                    while value < reject:
                        value = rng()
                    at = value % n
                action = int(t['choices'][first + at])
            else:
                at = end - 1
                for j in range(first, end - 1):
                    if coin(t['conditional_thresholds'][j], t['conditional_certain'][j]):
                        at = j
                        break
                action = int(t['choices'][at])
            chosen.append(action)
            frames[shot] ^= experiment.columns[action]
        for block in range(noise.blocks):
            first, end = map(int, t['block_offsets'][block:block+2])
            threshold, certain = int(t['thresholds'][block]), bool(t['certain'][block])
            if end - first < 8:
                for j in range(first, end):
                    if coin(threshold, certain):
                        select(int(t['block_sites'][j]))
                continue
            pending = (1 << (end - first)) - 1
            selected = pending if certain else 0
            if not certain:
                remaining = threshold
                while pending and remaining:
                    bits = rng()
                    if remaining & (1 << 63):
                        selected |= pending & ~bits
                        pending &= bits
                    else:
                        pending &= ~bits
                    remaining = (remaining << 1) & MASK64
            while selected:
                bit = (selected & -selected).bit_length() - 1
                selected &= selected - 1
                select(int(t['block_sites'][first + bit]))
        indices.extend(sorted(chosen))
        offsets.append(len(indices))
    return frames, NoiseBatch(experiment, np.asarray(offsets, dtype=np.uint64),
                              np.asarray(indices, dtype=np.uint32))


# Exact queries.

def check_status(result):
    """Reject invalid query results, synchronizing a device status check."""
    if bool(result.status.any()):
        flags = result.status if isinstance(result.status, np.ndarray) else result.status.get()
        codes = sorted(set(map(int, flags[flags != 0])))
        raise ValueError(f'CUDA exact query rejected frames (1=geometry, 2=padding, 3=nonreal): {codes}')


def to_host(result):
    if not isinstance(result.coefficients, np.ndarray):
        result = ExactQueryResult(result.coefficients.get(), result.powers.get(), result.status.get())
    check_status(result)
    return result


def tuples(result):
    host = to_host(result)
    return tuple(tuple((tuple(map(int, c)), int(p)) for c, p in zip(cs, ps))
                 for cs, ps in zip(host.coefficients, host.powers))


def evaluate_host(queries, frames):
    queries.experiment.validate()
    values = np.asarray(frames)
    if values.dtype != np.uint64 or values.ndim != 2 or values.shape[1] != queries.experiment.width:
        raise ValueError('host frames must match the compiled uint64 frame shape')
    return to_host(queries.evaluate(queries.cp.asarray(np.ascontiguousarray(values))))


def cpu_evaluate(queries, frames):
    queries.experiment.validate()
    values = np.asarray(frames)
    if values.dtype != np.uint64 or values.ndim != 2 or values.shape[1] != queries.experiment.width:
        raise ValueError('host frames must match the compiled uint64 frame shape')
    c, p, s = _native._cuda_query_cpu_evaluate(queries.handle, np.ascontiguousarray(values).tobytes())
    shape = (values.shape[0], queries.query_count)
    return to_host(ExactQueryResult(np.frombuffer(c, '<i8').reshape(*shape, 4),
                                    np.frombuffer(p, '<i4').reshape(shape),
                                    np.frombuffer(s, '<i4').reshape(shape)))


# Statistics.

def _norm_difference(a, b):
    """Exact ``a*a - 2*b*b`` as Python integers."""
    return np.array([int(x)*int(x) - 2*int(y)*int(y) for x, y in zip(a, b)], dtype=object)


def _real_values(coefficients, powers):
    """Vectorized CPU equivalent of the stable real conversion."""
    c = np.asarray(coefficients, dtype=np.int64)
    if np.any(c[..., 2] != 0) or np.any(c[..., 1] != -c[..., 3]):
        raise ValueError('statistics require real cyclotomic probabilities')
    a, b = c[..., 0], c[..., 1]
    value = a.astype(np.float64) + b.astype(np.float64) * math.sqrt(2)
    opposite = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
    if np.any(opposite):
        aa, bb = a[opposite], b[opposite]
        numerator = np.array([float(x) for x in _norm_difference(aa, bb)])
        value[opposite] = numerator / (aa.astype(np.float64) - bb.astype(np.float64)*math.sqrt(2))
    with np.errstate(over='ignore', under='ignore'):
        result = np.ldexp(value, powers)
    if np.any(~np.isfinite(result)) or np.any((value != 0) & (result == 0)):
        raise ValueError('statistics probability overflows or underflows float64')
    return result


def contributions(coefficients, powers, norm_power, weights=None):
    """Checked float64 ``(acceptance, error)`` contribution of every draw."""
    c, p = np.asarray(coefficients), np.asarray(powers)
    if c.dtype != np.int64 or p.dtype not in (np.dtype('int32'), np.dtype('int64')):
        raise TypeError('exact-query results require int64 coefficients and signed integer powers')
    if np.any(p < -(1 << 31)) or np.any(p > (1 << 31)-1):
        raise ValueError('query powers exceed int32 range')
    p = p.astype(np.int64, copy=False)
    if c.ndim != 3 or c.shape[-1] != 4 or c.shape[1] not in (1, 2) or p.shape != c.shape[:2]:
        raise ValueError('invalid exact-query result shape')
    a = _real_values(c[:, 0], p[:, 0])
    e = np.zeros_like(a)
    if c.shape[1] == 2:
        ca, ct = c[:, 0], c[:, 1]
        safe = (1 << 61)-1
        if np.any(ct < -safe) or np.any(ct > safe):
            raise ValueError('statistics target coefficient exceeds int64 sign bound')
        if np.any(ct[:, 2] != 0) or np.any(ct[:, 1] != -ct[:, 3]):
            raise ValueError('statistics require a real target probability')
        ta, tb = ct[:, 0], ct[:, 1]
        if np.any((ta <= 0) & (tb <= 0) & ((ta != 0) | (tb != 0))):
            raise ValueError('query result has a negative target probability')
        opposite = ((ta < 0) & (tb > 0)) | ((ta > 0) & (tb < 0))
        if np.any(opposite):
            x = ta[opposite]
            delta = _norm_difference(x, tb[opposite])
            if np.any(((x > 0) & (delta < 0)) | ((x < 0) & (delta > 0))):
                raise ValueError('query result has a negative target probability')
        pa, pt = p[:, 0].copy(), p[:, 1] - norm_power
        pa[np.all(ca == 0, axis=1)] = pt[np.all(ca == 0, axis=1)]
        pt[np.all(ct == 0, axis=1)] = pa[np.all(ct == 0, axis=1)]
        common = np.minimum(pa, pt)
        da, dt = pa-common, pt-common
        if np.any(da > 61) or np.any(dt > 61):
            raise ValueError('statistics exact error alignment exceeds int64 bound')
        bound_a, bound_t = ((1 << 61)-1) >> da, ((1 << 61)-1) >> dt
        if (np.any(ca > bound_a[:, None]) or np.any(ca < -bound_a[:, None]) or
                np.any(ct > bound_t[:, None]) or np.any(ct < -bound_t[:, None])):
            raise ValueError('statistics exact error alignment exceeds int64 bound')
        error = (ca << da[:, None]) - (ct << dt[:, None])
        e = _real_values(error, common)
    if np.any(a < 0) or np.any(e < 0) or np.any(a > 1+1e-13) or np.any(e > a+1e-13):
        raise ValueError('query result violates physical probability bounds')
    if weights is not None:
        weights = np.asarray(weights)
        if weights.dtype != np.float64 or weights.shape != a.shape:
            raise ValueError('likelihoods must be float64 with one value per shot')
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > _MAX_LIKELIHOOD):
            raise ValueError('likelihoods must be finite, nonnegative and at most 2^480')
        old_a, old_e = a, e
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            a, e = a*weights, e*weights
            invalid = np.any(~np.isfinite(a)) or np.any(~np.isfinite(e)) or np.any(~np.isfinite(a*a)) or np.any(~np.isfinite(e*e))
        if invalid or np.any((weights != 0) & (((old_a != 0) & (a == 0)) | ((old_e != 0) & (e == 0)))):
            raise ValueError('weighted statistics overflow or underflow float64')
    if np.any((a != 0) & (a*a == 0)) or np.any((e != 0) & (e*e == 0)):
        raise ValueError('statistics second moment underflows float64')
    return a, e


def cpu_moments(coefficients, powers, norm_power, weights=None):
    """NumPy reference of the device statistics kernel."""
    a, e = contributions(coefficients, powers, norm_power, weights)
    if not a.size:
        return _Moments(0, np.zeros(2), np.zeros(3))
    # Anchoring before averaging makes a constant irrational input exactly
    # constant even when raw sum/n would move by a rounding bit.
    da, de = a-a[0], e-e[0]
    ma, me = da.mean(), de.mean()
    da, de = da-ma, de-me
    scale_a, scale_e = float(np.max(np.abs(da))), float(np.max(np.abs(de)))
    def norm(values):
        scale = float(np.max(np.abs(values)))
        return scale*math.sqrt(float(((values/scale)**2).sum())) if scale else 0.
    if scale_a:
        scaled_a = da/scale_a
        variance_scaled = float((scaled_a*scaled_a).sum())
        slope = ((scale_e/scale_a)*float((scaled_a*(de/scale_e)).sum())/variance_scaled) if scale_e else 0.
        r00 = scale_a*math.sqrt(variance_scaled)
        residual = de-slope*da
        factor = [r00, slope*r00, norm(residual)]
    else:
        factor = [0., 0., norm(de)]
    return _Moments(a.size, np.array([float(a[0])+ma, float(e[0])+me]), np.array(factor))


def cpu_run(sampler, shots, *, seed=0, offset=0, batch_size=65536):
    """``CudaSampler.run`` with every step replaced by its CPU reference."""
    noise, queries = sampler._noise, sampler._queries
    chunks = []
    for first in range(0, shots, batch_size):
        size = min(batch_size, shots-first)
        weights = None
        if noise.has_importance_weights:
            frames, weights = cpu_sample_weighted(noise, size, seed, offset=offset+first)
        else:
            frames = cpu_sample(noise, size, seed, offset=offset+first)
        exact = cpu_evaluate(queries, frames)
        chunks.append(cpu_moments(exact.coefficients, exact.powers, sampler._norm_power, weights))
    return _summarize_moments(_merge_moments(chunks), sampler._has_error)


def history_scientific_rows(coefficients, powers, norm_power=0, proposal_rank=None, likelihood=None):
    """Arbitrary-precision oracle for history-normalized contributions."""
    from decimal import Decimal, localcontext
    c, p = np.asarray(coefficients), np.asarray(powers)
    exponent_span = max([abs(int(norm_power)), *(abs(int(x)) for x in p.flat)], default=0)
    rows = []
    with localcontext() as context:
        context.prec = max(100, int(exponent_span*.61)+100)
        two, sqrt2 = Decimal(2), Decimal(2).sqrt()
        def scalar(cs, power):
            a, b, imaginary, opposite = map(int, cs)
            if imaginary or b != -opposite:
                raise ValueError('nonreal history mass')
            if a*b < 0:
                value = Decimal(a*a-2*b*b)/(Decimal(a)-Decimal(b)*sqrt2)
            else:
                value = Decimal(a)+Decimal(b)*sqrt2
            return value*two**int(power)
        def scientific(value):
            if not value:
                return (0., 0)
            exponent = int(value.adjusted()*math.log2(10))+1
            normalized = value/two**exponent
            while normalized < Decimal('.5'):
                normalized *= 2
                exponent -= 1
            while normalized >= 1:
                normalized /= 2
                exponent += 1
            mantissa, shift = math.frexp(float(normalized))
            return mantissa, exponent+shift
        for shot in range(len(c)):
            history = scalar(c[shot, 0], p[shot, 0])
            acceptance = scalar(c[shot, 1], p[shot, 1])
            target = scalar(c[shot, 2], int(p[shot, 2])-norm_power) if c.shape[1] == 3 else Decimal(0)
            if not 0 <= target <= acceptance <= history:
                raise ValueError('invalid history masses')
            divisor = history if proposal_rank is None else two**(-int(proposal_rank[shot]))
            if divisor == 0:
                raise ValueError('zero conditional history mass')
            weight = Decimal.from_float(float(likelihood[shot])) if likelihood is not None else Decimal(1)
            rows.append((scientific(acceptance*weight/divisor),
                         scientific((acceptance-target)*weight/divisor) if c.shape[1] == 3 else (0., 0)))
    return rows


def history_draw(experiment, handle, metadata, seed, shot):
    """Exact addressed history with the original ControllerBinding.invoke calls."""
    inputs, outputs = [], []
    compiled = experiment.compiled
    for site in metadata['sites']:
        row = _native.cuda_history_replay(handle, seed, shot, tuple(inputs), tuple(outputs))
        controls = (compiled._control_metadata if site['stage'] == 'protocol'
                    else compiled._recovery_control_metadata)
        instruction, binding, _, _ = controls.callbacks[site['index']]
        result = binding.invoke(tuple(row), site['output_count'], source=instruction.source)
        inputs.append(row)
        outputs.append(bytes(result))
    return inputs, outputs, _native._cuda_history_cpu_reference(handle, seed, shot, tuple(outputs))


def history_affine_value(raw, packed):
    """Independent integer-parity reference for schedule and cap affine rows."""
    values = np.frombuffer(packed, dtype='<u8')
    rows = np.frombuffer(raw['rows'], dtype='<u8').reshape(raw['count'], len(values))
    constants = np.frombuffer(raw['constants'], dtype='u1')
    return bytes(int(constant) ^ (sum((int(a)&int(b)).bit_count() for a, b in zip(row, values)) & 1)
                 for row, constant in zip(rows, constants))
