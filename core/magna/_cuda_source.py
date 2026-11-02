"""Load packaged CUDA C++ sources without importing a CUDA runtime."""
from pathlib import Path


def _read_cuda_source(name):
    return (Path(__file__).parent / 'native' / 'cuda' / name).read_text(encoding='utf-8')
