"""Device history steps joined by the existing host decoder calls."""
import numpy as np

from . import _native
from ._cuda_queries import GpuQueries, _query_arrays, _symmetric_upper
from ._cuda_source import _read_cuda_source
from ._cuda_history_statistics import _HistoryStatistics, _merge_scaled, _scientific_moments


class _CudaHistory:
    def __init__(self, experiment):
        self.experiment = experiment
        self.handle, self.metadata = _native.make_cuda_history(experiment.handle)
        self.proposal = experiment.compiled.options['history_sampler'] == 'affine_support'
        self.stride = self.metadata['frame_words']
        self.pauli_words = 2*experiment.words
        self.sites = self.metadata['sites']
        self.bindings = []
        for site in self.sites:
            controls = (experiment.compiled._control_metadata if site['stage'] == 'protocol'
                        else experiment.compiled._recovery_control_metadata)
            self.bindings.append(controls.callbacks[site['index']][:2])
        self.query_bases = [GpuQueries(experiment, [raw]) for raw in self.metadata.get('query_bases', ())]
        self.terminal = GpuQueries(experiment, self.metadata['terminal'], shared=self.query_bases)
        self.prefixes = []
        self.closed = []
        words = (self.metadata['open_bits']+63)//64
        for site in self.sites:
            keep = np.frombuffer(site['keep'], dtype='<u8').reshape(site['pending_count'], words)
            closed = ~np.any(keep, axis=1)
            self.closed.append(closed)
            self.prefixes.append([GpuQueries(experiment, [raw], shared=self.query_bases)
                if j == 0 or not closed[j-1] else None
                for j, raw in enumerate(site['prefixes'])] if not np.all(closed) else [])
        mantissa, exponent = self.metadata['normalization']
        self.normalization = (2*mantissa, exponent-1)
        self._cp = None

    def _array(self, raw, dtype):
        return self._cp.asarray(np.frombuffer(raw, dtype=dtype))

    def _affine(self, raw):
        rows = np.frombuffer(raw['rows'], dtype='<u8').reshape(raw['count'], self.stride)
        indexes, words = np.nonzero(rows)
        counts = np.bincount(indexes, minlength=raw['count'])
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.uint32)
        return (self._cp.asarray(offsets), self._cp.asarray(words.astype(np.uint32)),
                self._cp.asarray(rows[indexes, words]), self._array(raw['constants'], 'u1'))

    def prepare(self, cp):
        if self._cp is not None:
            return
        self._cp = cp
        self.control = cp.RawModule(code=_read_cuda_source('history_control.cu'), options=('--std=c++11',))
        self.statistics = _HistoryStatistics(cp)
        self.sector_base = self._array(self.metadata['sector_base'], '<u8')
        self.sector_columns = self._array(self.metadata['sector_columns'], '<u8')
        self.device_sites = []
        for site, closed in zip(self.sites, self.closed):
            arrays = {name: self._array(site[name], dtype) for name, dtype in
                (('outputs', '<u4'), ('input_for_pending', '<u4'), ('columns', '<u8'))}
            arrays.update(inputs=self._affine(site['inputs']), offsets=self._affine(site['offsets']),
                closed=cp.asarray(closed, dtype=cp.uint8))
            if self.proposal and not np.all(closed):
                for name in ('keep', 'events'):
                    arrays[name] = self._array(site[name], '<u8')
                for name in ('offsets', 'outcomes'):
                    arrays[name+'_rows'] = self._array(site[name]['rows'], '<u8')
                    arrays[name+'_constants'] = self._array(site[name]['constants'], 'u1')
            self.device_sites.append(arrays)
        if self.proposal and any(not np.all(closed) for closed in self.closed):
            raw = self.metadata['mother']
            host = _query_arrays(raw)
            if raw['mode'] not in (1, 3, 4):
                host['upper'] = _symmetric_upper(host['upper'], raw['n'])
            names = ('support', 'linear', 'upper', 'linear_shift', 'magic', 'magic_masks',
                     'feature_offsets', 'feature_words', 'feature_masks', 'feature_constants')
            self.mother = {name: cp.asarray(host[name]) for name in names}
            self.open_rows = self._array(self.metadata['open_base']['rows'], '<u8')
            self.open_constants = self._array(self.metadata['open_base']['constants'], 'u1')
            self.entries = []
            for entry in self.metadata['entries']:
                arrays = {name: self._array(entry[name], dtype) for name, dtype in
                    (('shift', '<u8'), ('change', '<u8'), ('columns', '<u8'), ('base', '<u8'), ('linear', 'u1'))}
                arrays['upper'] = cp.asarray(_symmetric_upper(np.frombuffer(entry['upper'], '<u8'), entry['n']))
                self.entries.append(arrays)
            maximum = max([raw['n'], *(entry['n'] for entry in self.metadata['entries'])])
            source = _read_cuda_source('history_sampling.cu')
            for key, value in dict(MAX_N=maximum, MAX_OPEN=self.metadata['open_bits'],
                MAX_PENDING=max((site['pending_count'] for site in self.sites), default=0)).items():
                source = source.replace('@'+key+'@', str(value))
            self.support = cp.RawModule(code=source, options=('--std=c++11',))

    def _inputs(self, frames, arrays, count, column_major=False):
        cp = self._cp
        shots = frames.shape[0]
        output = cp.empty((count, shots) if column_major else (shots, count), dtype=cp.uint8)
        if shots*count:
            self.control.get_function('history_inputs')(((shots*count+255)//256,), (256,),
                (frames, *arrays, output, np.uint64(shots), np.int32(self.stride),
                 np.int32(count), np.int32(column_major)))
        return output

    def _affine_site(self, frames, ranks, index, seed, offset):
        cp = self._cp
        site, arrays = self.sites[index], self.device_sites[index]
        shots, pending = frames.shape[0], site['pending_count']
        pwords, owords = (pending+63)//64, (self.metadata['open_bits']+63)//64
        base = cp.empty((shots, pwords), dtype=cp.uint64)
        basis = cp.empty((shots, pending, pwords), dtype=cp.uint64)
        valid, rank = cp.empty(shots, dtype=cp.uint8), cp.empty(shots, dtype=cp.int64)
        mother = self.metadata['mother']
        feature_names = ('feature_offsets', 'feature_words', 'feature_masks', 'feature_constants')
        features = self._inputs(frames, tuple(self.mother[k] for k in feature_names),
                                mother['feature_count'], True)
        blocks = min(shots, 128)
        for term, (entry, values) in enumerate(zip(self.metadata['entries'], self.entries)):
            n, m = mother['n'], entry['n']
            words = n*((n+63)//64)+m*((m+63)//64)+m*owords+m*pwords
            scratch = cp.empty((blocks, words), dtype=cp.uint64)
            parameters = [self.stride, n, m, self.metadata['open_bits'], pending, site['event_count'],
                          self.mother['support'].size, self.mother['magic'].size//8,
                          m == n and not bool(entry['change']), term == 0]
            self.support.get_function('history_affine_entry')((blocks,), (128,),
                (frames, features, self.mother['support'], self.mother['linear'], self.mother['upper'],
                 self.mother['linear_shift'], self.mother['magic'], self.mother['magic_masks'],
                 self.open_rows, self.open_constants, values['shift'], values['change'], values['columns'],
                 values['base'], values['linear'], values['upper'], arrays['keep'], arrays['offsets_rows'],
                 arrays['offsets_constants'], arrays['events'], arrays['outcomes_rows'],
                 arrays['outcomes_constants'], scratch, base, basis, valid, rank,
                 np.uint64(shots), *map(np.int32, parameters)))
        self.support.get_function('history_affine_draw')(((shots+127)//128,), (128,),
            (frames, base, basis, valid, rank, ranks, np.uint64(shots), np.uint64(seed), np.uint64(offset),
             *map(np.int32, (index, self.stride, self.pauli_words, self.metadata['sector_bits'],
                            site['observation_first'], pending))))

    def _exact_site(self, frames, index, seed, offset):
        cp = self._cp
        site, arrays = self.sites[index], self.device_sites[index]
        shots = frames.shape[0]
        continuation = cp.zeros(shots, dtype=cp.uint8)
        random = cp.zeros((shots, 3), dtype=cp.uint64)
        current = self.prefixes[index][0].evaluate(frames)
        for j, closed in enumerate(self.closed[index]):
            if closed:
                continue
            zero = self.prefixes[index][j+1].evaluate(frames)
            self.control.get_function('history_choose')(((shots+127)//128,), (128,),
                (frames, current.coefficients, current.powers, current.status,
                 zero.coefficients, zero.powers, zero.status, random, continuation,
                 np.uint64(shots), np.uint64(seed), np.uint64(offset),
                 *map(np.int32, (index, self.stride, self.pauli_words,
                                self.metadata['sector_bits']+site['observation_first']+j))))
        return continuation

    def _statistics_inputs(self, ranks, weights):
        if not self.proposal:
            return None, weights
        mantissa, exponent = self.normalization
        if weights is not None:
            weights, powers = self._cp.frexp(weights)
            return ranks+powers-exponent, weights/mantissa
        adjusted = ranks-exponent if exponent else ranks
        weights = self._cp.full(ranks.shape[0], 1/mantissa, dtype=self._cp.float64) if mantissa != 1 else None
        return adjusted, weights

    def chunk(self, noise, shots, seed, offset):
        cp = noise.cp
        self.prepare(cp)
        weights = None
        if noise.has_importance_weights:
            original, weights = noise.sample_weighted(shots, seed, offset=offset)
        else:
            original = noise.sample(shots, seed, offset=offset)
        frames = cp.empty((shots, self.stride), dtype=cp.uint64)
        ranks = cp.empty(shots if self.proposal else 1, dtype=cp.int64)
        self.control.get_function('initialize_history')(((shots+127)//128,), (128,),
            (original, self.sector_base, self.sector_columns, frames, ranks,
             np.uint64(shots), np.uint64(seed), np.uint64(offset),
             *map(np.int32, (self.pauli_words, self.stride, self.metadata['sector_bits'],
                            self.metadata['sector_rank'], self.proposal))))
        previous_inputs, previous_outputs = [], []
        for index, (site, arrays, binding) in enumerate(zip(self.sites, self.device_sites, self.bindings)):
            continuation = None
            pending = site['pending_count']
            if pending:
                self.control.get_function('history_closed')(((shots+127)//128,), (128,),
                    (frames, *arrays['offsets'], arrays['closed'], np.uint64(shots),
                     *map(np.int32, (self.stride, self.pauli_words, self.metadata['sector_bits'],
                                    site['observation_first'], pending))))
                if not np.all(self.closed[index]):
                    if self.proposal:
                        self._affine_site(frames, ranks, index, seed, offset)
                    else:
                        continuation = self._exact_site(frames, index, seed, offset)
            device_inputs = self._inputs(frames, arrays['inputs'], site['input_count'])
            if continuation is not None:
                packet = cp.concatenate((device_inputs, continuation[:, None]), axis=1).get()
                host_inputs = packet[:, :-1].copy()
                for row in np.flatnonzero(packet[:, -1]):
                    completed = _native.cuda_history_replay(self.handle, seed, offset+int(row),
                        tuple(x[row].tobytes() for x in previous_inputs),
                        tuple(x[row].tobytes() for x in previous_outputs))
                    host_inputs[row] = np.frombuffer(completed, dtype=np.uint8)
            else:
                host_inputs = device_inputs.get()
            inst, controller = binding
            outputs = np.empty((shots, site['output_count']), dtype=np.uint8)
            for row, bits in enumerate(host_inputs.tolist()):
                outputs[row] = controller.invoke(bits, site['output_count'], source=inst.source)
            self.control.get_function('history_feedback')(((shots+127)//128,), (128,),
                (frames, *arrays['inputs'], arrays['input_for_pending'], arrays['outputs'], arrays['columns'],
                 cp.asarray(host_inputs), cp.asarray(outputs), np.uint64(shots),
                 *map(np.int32, (self.stride, self.pauli_words, self.metadata['sector_bits'],
                                site['observation_first'], pending, site['input_count'], site['output_count']))))
            previous_inputs.append(host_inputs)
            previous_outputs.append(outputs)
        exact = self.terminal.evaluate(frames)
        adjusted_ranks, weights = self._statistics_inputs(ranks, weights)
        moments, continuation = self.statistics.reduce(exact, proposal_rank=adjusted_ranks, likelihood=weights)
        completed = []
        for row in np.flatnonzero(continuation):
            completed.append(_native.cuda_history_replay(self.handle, seed, offset+int(row),
                tuple(x[row].tobytes() for x in previous_inputs),
                tuple(x[row].tobytes() for x in previous_outputs)))
        return _merge_scaled((moments, _scientific_moments(completed))) if completed else moments
