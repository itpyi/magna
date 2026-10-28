"""Spawnable CPU references and failing workers for multi-GPU control tests."""
import os
import time
import traceback

from magna.cuda_multi import _compile_snapshot, _serve
from magna.cuda_sampler import CudaSampler
from tests.support.cuda_reference import cpu_sample, cpu_sample_weighted, cpu_evaluate, cpu_moments


class CpuSampler(CudaSampler):
    def _chunks(self, shots, seed, offset, batch_size):
        for first in range(0, shots, batch_size):
            size = min(batch_size, shots-first)
            weights = None
            if self._noise.has_importance_weights:
                frames, weights = cpu_sample_weighted(self._noise, size, seed, offset=offset+first)
            else:
                frames = cpu_sample(self._noise, size, seed, offset=offset+first)
            exact = cpu_evaluate(self._queries, frames)
            yield cpu_moments(exact.coefficients, exact.powers, self._norm_power, weights)


def cpu_worker(connection, device, snapshot):
    try:
        _serve(connection, CpuSampler(_compile_snapshot(snapshot)))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def fail_worker(connection, device, snapshot):
    connection.send(('ready', None))
    connection.recv()
    connection.send(('error', 'injected device failure'))
    connection.close()


def crash_worker(connection, device, snapshot):
    connection.send(('ready', None))
    connection.recv()
    os._exit(7)


def hang_worker(connection, device, snapshot):
    connection.send(('ready', None))
    connection.recv()
    time.sleep(120)


def startup_failure(connection, device, snapshot):
    connection.send(('error', 'injected initialization failure'))
    connection.close()
