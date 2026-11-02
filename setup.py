"""C++17 extension configuration for PEP 517 builds."""
import sys
from pathlib import Path

from setuptools import Extension, setup


native_dir = Path("core/magna/native")
flags = ["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3", "-DNDEBUG", "-pthread"]
setup(ext_modules=[Extension(
    "magna._native",
    sources=[str(path) for path in sorted(native_dir.rglob("*.cpp"))],
    depends=[str(path) for path in sorted(native_dir.rglob("*.hpp"))],
    language="c++",
    extra_compile_args=flags,
    extra_link_args=[] if sys.platform == "win32" else ["-pthread"],
)])
