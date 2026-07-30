"""Public-API benchmarks against Astropy on identical arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import astropy.units as u  # noqa: E402
from astropy import coordinates as ac  # noqa: E402
from astropy.time import Time as AstroTime  # noqa: E402
from astropy.wcs import WCS as AstroWCS  # noqa: E402

import mojoastropy as ma  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    model = platform.processor()
    if not model or model.lower() in {"x86_64", "amd64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as stream:
                model = next(
                    line.split(":", 1)[1].strip()
                    for line in stream
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            model = "unknown CPU"
    return f"{model}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(0)
    n = 2_000_000
    r = rng.uniform(0.1, 10.0, n)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    lon = rng.uniform(-np.pi, np.pi, n)
    lon2 = rng.uniform(-np.pi, np.pi, n)
    lat2 = rng.uniform(-np.pi / 2, np.pi / 2, n)
    jd = np.linspace(2451545.0, 2461545.0, n)
    pixels = rng.uniform([0, 0], [4096, 4096], (n, 2))
    header = {
        "CRPIX1": 2048.5,
        "CRPIX2": 2048.5,
        "CRVAL1": 202.48,
        "CRVAL2": 47.17,
        "CD1_1": -6.9e-5,
        "CD1_2": 1e-7,
        "CD2_1": -2e-7,
        "CD2_2": 6.9e-5,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }
    sip_header = {
        **header,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 3,
        "B_ORDER": 3,
        "A_2_0": 2e-6,
        "A_0_2": -1e-6,
        "B_2_0": -1e-6,
        "B_0_2": 2e-6,
    }
    awcs, mwcs = AstroWCS(header), ma.WCS(header)
    asip, msip = AstroWCS(sip_header), ma.WCS(sip_header)
    world = awcs.all_pix2world(pixels, 0)
    sip_pixels = pixels[:500_000]
    sip_world = asip.all_pix2world(sip_pixels, 0)

    cases = [
        (
            "spherical_to_cartesian (2M)",
            lambda: ma.spherical_to_cartesian(r, lat, lon),
            lambda: ac.spherical_to_cartesian(r, lat, lon),
        ),
        (
            "angular_separation (2M)",
            lambda: ma.angular_separation(lon, lat, lon2, lat2),
            lambda: ac.angular_separation(lon, lat, lon2, lat2),
        ),
        (
            "ICRS -> Galactic (2M)",
            lambda: ma.SkyCoord(lon, lat, unit="rad").galactic,
            lambda: ac.SkyCoord(lon * u.rad, lat * u.rad).galactic,
        ),
        (
            "UTC -> TCG (2M)",
            lambda: ma.Time(jd, format="jd", scale="utc").tcg,
            lambda: AstroTime(jd, format="jd", scale="utc").tcg,
        ),
        (
            "TAN pix2world (2M)",
            lambda: mwcs.all_pix2world(pixels, 0),
            lambda: awcs.all_pix2world(pixels, 0),
        ),
        (
            "TAN world2pix (2M)",
            lambda: mwcs.all_world2pix(world, 0),
            lambda: awcs.all_world2pix(world, 0),
        ),
        (
            "SIP world2pix (500k)",
            lambda: msip.all_world2pix(sip_world, 0),
            lambda: asip.all_world2pix(sip_world, 0),
        ),
    ]
    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-astropy | astropy | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_fn, astropy_fn in cases:
        mojo_fn()
        astropy_fn()
        mojo_time = timeit(mojo_fn)
        astropy_time = timeit(astropy_fn)
        ratio = astropy_time / mojo_time
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{astropy_time * 1000:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
