"""Experimental single-host CUDA sampling with one persistent process per GPU."""
from copy import deepcopy
from dataclasses import dataclass
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from threading import Lock
from time import monotonic, perf_counter
import traceback

import numpy as np

from .cuda_sampler import CudaSampler, _merge_moments, _run_arguments, _summarize_moments
from ._cuda_noise import _uint64


def _snapshot(compiled):
    """Only source and configuration cross processes, never native handles.

    Resolve targets from the compiled object so relative paths, later edits to
    files and the worker's working directory cannot change the experiment.
    """
    task = deepcopy({k: v for k, v in compiled.task['raw'].items()
                     if k not in ('ideal_state', 'recovery')})
    target = None
    # A classical error-observable target is generated again by the compiler.
    if compiled.ideal_state is not None and compiled._error_observables is None:
        ideal = compiled.ideal_state
        target = (ideal.circuit.to_text(), ideal.outputs, dict(ideal.postselection))
    recovery = None if compiled.recovery is None else compiled.recovery.circuit.to_text()
    return dict(circuit=compiled._source_circuit.to_text(), task=task,
                options=deepcopy(compiled.options), sampling=deepcopy(compiled.noise_schedule.sampling),
                target=target, recovery=recovery)


def _compile_snapshot(snapshot):
    from . import compile_circuit, parse_circuit
    task = deepcopy(snapshot['task'])
    target = snapshot['target']
    ideal = None
    if target is not None:
        text, outputs, postselection = target
        ideal = parse_circuit(text)
        ideal = (ideal if outputs is None and not postselection else
                 dict(circuit=ideal, outputs=outputs, postselection=postselection))
    recovery = snapshot['recovery']
    return compile_circuit(parse_circuit(snapshot['circuit']), task=task,
                           options=snapshot['options'], sampling=snapshot['sampling'], ideal_state=ideal,
                           recovery=None if recovery is None else parse_circuit(recovery))


def _partitions(shots, offset, devices):
    """Balanced, contiguous, nonempty intervals in the user's device order."""
    quotient, remainder = divmod(shots, len(devices))
    result = []
    for i, device in enumerate(devices):
        count = quotient + (i < remainder)
        if count:
            result.append(dict(device=device, offset=offset, shots=count))
        offset += count
    return result


def _serve(connection, sampler):
    connection.send(('ready', None))
    while True:
        command = connection.recv()
        if command is None:
            return
        shots, seed, offset, batch_size = command
        sampler._experiment.validate()
        start = perf_counter()
        moments = _merge_moments(sampler._chunks(shots, seed, offset, batch_size))
        connection.send(('result', (moments, perf_counter()-start)))


def _worker_main(connection, device, snapshot):
    try:
        import cupy as cp
        # Device selection covers lazy allocation, JIT compilation and all runs.
        with cp.cuda.Device(device):
            sampler = CudaSampler(_compile_snapshot(snapshot))
            _serve(connection, sampler)
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


@dataclass
class _Worker:
    device: int
    process: object
    connection: object


class MultiGpuCudaSampler:
    """Reuse independent CUDA samplers on explicitly selected local devices.

    Create this object inside ``if __name__ == '__main__':`` in a script and
    use it as a context manager. Workers use ``spawn`` even on Linux. Device
    IDs are logical IDs after CUDA_VISIBLE_DEVICES filtering. No CUDA context
    is created in the parent. ``timeout`` bounds each run including startup;
    None disables that deadline. Failure closes the whole pool and raises,
    without returning a misleading partial estimate.
    """
    def __init__(self, compiled, *, devices, timeout=None):
        devices = tuple(_uint64(d, 'device') for d in devices)
        if not devices or len(set(devices)) != len(devices):
            raise ValueError('devices must be a nonempty sequence of distinct GPU IDs')
        if any(d > 2**31-1 for d in devices):
            raise ValueError('device ID exceeds int32 range')
        if timeout is not None:
            if isinstance(timeout, (bool, np.bool_)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('timeout must be finite and positive, or None')
        self._devices = devices
        self._timeout = timeout
        # All support checks run before any child or CUDA context is started.
        self._validation = CudaSampler(compiled)
        self._snapshot = _snapshot(compiled)
        self._context = mp.get_context('spawn')
        self._workers = {}
        self._closed = False
        self._owner = os.getpid()
        self._lock = Lock()

    @property
    def devices(self):
        return self._devices

    def _check_owner(self):
        if os.getpid() != self._owner:
            raise RuntimeError('a multi-GPU sampler must stay in its creating process')

    def _collect(self, devices, expected, deadline):
        pending = {d: self._workers[d] for d in devices}
        results = {}
        while pending:
            remaining = None if deadline is None else max(0., deadline-monotonic())
            if remaining == 0:
                raise TimeoutError('multi-GPU sampling exceeded timeout')
            handles = [w.connection for w in pending.values()]
            handles += [w.process.sentinel for w in pending.values()]
            ready = wait(handles, timeout=remaining)
            if not ready:
                raise TimeoutError('multi-GPU sampling exceeded timeout')
            for device, worker in list(pending.items()):
                if worker.connection in ready:
                    try:
                        kind, payload = worker.connection.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(f'GPU {device} worker disconnected') from exc
                    if kind == 'error':
                        raise RuntimeError(f'GPU {device} worker failed:\n{payload}')
                    if kind != expected:
                        raise RuntimeError(f'GPU {device} sent unexpected response {kind!r}')
                    results[device] = payload
                    del pending[device]
                elif worker.process.sentinel in ready:
                    worker.process.join()
                    raise RuntimeError(f'GPU {device} worker exited (code {worker.process.exitcode})')
        return results

    def _start(self, devices, deadline):
        new = []
        for device in devices:
            if device in self._workers:
                continue
            parent, child = self._context.Pipe()
            process = self._context.Process(target=_worker_main, args=(child, device, self._snapshot),
                                            name=f'magna-cuda-{device}', daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self._workers[device] = _Worker(device, process, parent)
            new.append(device)
        self._collect(new, 'ready', deadline)

    def run(self, shots, *, seed=0, offset=0, batch_size=65536):
        """Sample disjoint draw ranges, then merge centered statistics in range order."""
        self._check_owner()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('multi-GPU sampler is already in use')
        try:
            if self._closed:
                raise RuntimeError('multi-GPU sampler is closed')
            self._validation._experiment.validate()
            shots, seed, offset, batch_size = _run_arguments(
                shots, seed, offset, batch_size, self._validation._experiment.metadata['max_shots'])
            partitions = _partitions(shots, offset, self.devices)
            active = [p['device'] for p in partitions]
            total_start = perf_counter()
            deadline = None if self._timeout is None else monotonic()+self._timeout
            try:
                self._start(active, deadline)
                sample_start = perf_counter()
                for part in partitions:
                    self._workers[part['device']].connection.send(
                        (part['shots'], seed, part['offset'], batch_size))
                results = self._collect(active, 'result', deadline)
                for part in partitions:
                    moments, elapsed = results[part['device']]
                    if moments.count != part['shots']:
                        raise RuntimeError(f"GPU {part['device']} returned an incomplete shard")
                    part['sampling_seconds'] = elapsed
                totals = _merge_moments(results[d][0] for d in active)
                summary = _summarize_moments(totals, self._validation._has_error)
                finished = perf_counter()
            except BaseException:
                self._shutdown(abort=True)
                raise
            return dict(status='complete', requested=shots, completed=totals.count,
                        seed=seed, offset=offset, batch_size=batch_size,
                        devices=list(self.devices), shards=partitions,
                        timings=dict(total_seconds=finished-total_start,
                                     startup_seconds=sample_start-total_start,
                                     sampling_seconds=finished-sample_start), **summary)
        finally:
            self._lock.release()

    def _shutdown(self, *, abort=False):
        self._closed = True
        workers, self._workers = list(self._workers.values()), {}
        for worker in workers:
            if abort:
                if worker.process.is_alive():
                    worker.process.terminate()
            else:
                try:
                    worker.connection.send(None)
                except (OSError, EOFError):
                    pass
        # Bound cleanup even when another device hangs or dies in CUDA.
        deadline = monotonic()+2
        for worker in workers:
            worker.process.join(max(0., deadline-monotonic()))
        for worker in workers:
            if worker.process.is_alive():
                worker.process.kill()
        for worker in workers:
            worker.process.join()
            worker.connection.close()
            worker.process.close()

    def close(self):
        """Release all worker processes and their device allocations; idempotent."""
        self._check_owner()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('multi-GPU sampler is already in use')
        try:
            self._shutdown()
        finally:
            self._lock.release()

    def __enter__(self):
        self._check_owner()
        if self._closed:
            raise RuntimeError('multi-GPU sampler is closed')
        return self

    def __exit__(self, *exc):
        self.close()
