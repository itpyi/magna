"""Availability checks for opt-in device tests; never skip a kernel failure."""
import unittest


def require_cuda():
    """Return CuPy, or skip only when the optional runtime/device is absent.

    Source CI does not install CuPy and still runs all host-side CUDA controls.
    Once a device is present, compilation and execution failures remain test
    failures rather than being mistaken for an unavailable optional dependency.
    """
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:
        raise unittest.SkipTest(f'optional CuPy runtime unavailable: {exc}') from exc
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except (cp.cuda.runtime.CUDARuntimeError, OSError) as exc:
        raise unittest.SkipTest(f'CUDA device unavailable: {exc}') from exc
    if count == 0:
        raise unittest.SkipTest('CUDA device unavailable')
    return cp
