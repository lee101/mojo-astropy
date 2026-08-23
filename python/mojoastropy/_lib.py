"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get(
    "MOJOASTROPY_LIB", os.path.join(ROOT, "dist", "libmojo-astropy.so")
)

I = ctypes.c_int64
F = ctypes.c_double
PARALLEL_THRESHOLD = 262_144
PARALLEL_WORKERS = 4

_SIGNATURES = {
    "ma_time_convert": ([I, I, I, I, I, I, I], None),
    "ma_spherical_to_cartesian": ([I, I, I, I, I, I, I], None),
    "ma_cartesian_to_spherical": ([I, I, I, I, I, I, I], None),
    "ma_angular_separation": ([I, I, I, I, I, I], None),
    "ma_position_angle": ([I, I, I, I, I, I], None),
    "ma_rotate_spherical": ([I, I, I, I, I, I], None),
    "ma_wcs_pix2world": (
        [I, I, I, I, I, I] + [F] * 8 + [I, I, I, I, I],
        None,
    ),
    "ma_wcs_pix2world_points": ([I, I, I, I] + [F] * 8 + [I], None),
    "ma_wcs_world2pix": (
        [I, I, I, I, I, I, I] + [F] * 8 + [I, I, I, I, I, F, I],
        None,
    ),
    "ma_wcs_world2pix_points": ([I, I, I, I] + [F] * 8 + [I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and (not os.path.exists(SRC) or os.path.getmtime(LIB) >= os.path.getmtime(SRC))
    ):
        return LIB
    mojo = os.environ.get("MOJOASTROPY_MOJO") or shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside pixi or set MOJOASTROPY_MOJO")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "shared-lib",
            SRC,
            "-o",
            LIB,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None
_executor = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def parallel_call(size: int, call) -> None:
    global _executor
    if size < PARALLEL_THRESHOLD:
        call(0, size)
        return
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=PARALLEL_WORKERS)
    futures = []
    for worker in range(PARALLEL_WORKERS):
        begin = worker * size // PARALLEL_WORKERS
        end = (worker + 1) * size // PARALLEL_WORKERS
        futures.append(_executor.submit(call, begin, end))
    for future in futures:
        future.result()


def f64(value) -> np.ndarray:
    source = np.asanyarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be converted to float64 without data loss")
    if np.issubdtype(source.dtype, np.floating) and source.dtype.itemsize > 8:
        raise TypeError("floating-point values wider than float64 are unsupported")
    if np.issubdtype(source.dtype, np.integer) and source.size:
        limit = 1 << 53
        if np.any(source > limit) or np.any(source < -limit):
            raise ValueError("integer values outside the exact float64 range are unsupported")
    return np.ascontiguousarray(source, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype != np.float64 or not value.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    pointer = int(value.ctypes.data)
    if value.size and pointer == 0:
        raise ValueError("non-empty FFI buffer has a null pointer")
    return pointer


if __name__ == "__main__":
    print(build(force=True))
