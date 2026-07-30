# mojo-astropy

`mojo-astropy` is a focused port of compute-heavy
[Astropy](https://www.astropy.org/) astronomy routines to
[Mojo](https://www.modular.com/mojo). It is a standalone Python package backed
by one native shared library; Astropy itself is needed only by the parity tests
and benchmarks.

The Python package is named `mojoastropy` so it can coexist with upstream
`astropy`. For the subset below, public names and call signatures follow
Astropy closely and accept scalar, broadcast, and contiguous array inputs.

## Coverage

| area | implemented |
| --- | --- |
| Coordinates | `spherical_to_cartesian`, `cartesian_to_spherical`, `angular_separation`, `position_angle`, `rotation_matrix`; `Angle`, `SkyCoord`, `ICRS`, and `Galactic`; vectorized ICRS–Galactic transforms and separations |
| Time | split-JD `Time`; `jd`, `mjd`, `unix`, `iso`, and `isot`; UTC, TAI, TT, TCG, geocentric TDB, and TCB scale properties |
| WCS | 2D FITS linear and celestial TAN projection; CD or PC/CDELT matrices; forward SIP and iterative inverse SIP; both Astropy calling conventions and pixel origins; `WCS` and `Sip` |

This is intentionally not all of Astropy. Coordinate velocities, distances,
FK4/FK5/ecliptic/AltAz frames, observer locations, UT1 and IERS data, time
arithmetic, non-TAN celestial projections, more than two WCS axes, axis
reordering, and lookup-table detector distortions are not covered.

UTC scale conversion is supported from 1972-01-01 onward with the complete
post-1972 leap-second table. Dates after the last announced leap second assume
TAI−UTC remains 37 seconds. TDB is the standard seven-term geocentric
approximation: it stays within 40 microseconds of Astropy/ERFA over the tested
1972–2025 interval, but it does not include a topocentric observer term.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/kernels.mojo` to
`dist/libmojo-astropy.so`. Outside Pixi, the Python loader also accepts
`MOJOASTROPY_LIB=/path/to/libmojo-astropy.so` or
`MOJOASTROPY_MOJO=/path/to/mojo`.

## Usage

```python
import numpy as np
import mojoastropy as astro

stars = astro.SkyCoord(
    ra=[266.4051, 83.6331],
    dec=[-28.936175, 22.0145],
    unit="deg",
    frame="icrs",
)
print(stars.galactic.l.deg, stars.galactic.b.deg)

times = astro.Time([51544.5, 57754.0], format="mjd", scale="utc")
print(times.tai.jd, times.tt.jd)

wcs = astro.WCS(
    {
        "CRPIX1": 512.5,
        "CRPIX2": 384.5,
        "CRVAL1": 202.4823228,
        "CRVAL2": 47.17511893,
        "CD1_1": -0.0002777778,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.0002777778,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }
)
world = wcs.all_pix2world(np.array([[511.5, 383.5], [100.0, 200.0]]), 0)
pixels = wcs.all_world2pix(world, 0)
assert np.allclose(pixels, [[511.5, 383.5], [100.0, 200.0]])
```

The example runs with `pixi run python example.py`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, using Python 3.13.14. Each row is the best of three
runs after warm-up, using the same input arrays through each library's public
Python API.

| case | mojo-astropy | astropy | result |
| --- | ---: | ---: | ---: |
| spherical_to_cartesian (2M) | 37.86 ms | 225.56 ms | 5.96x faster |
| angular_separation (2M) | 99.84 ms | 485.83 ms | 4.87x faster |
| ICRS -> Galactic (2M) | 133.96 ms | 980.67 ms | 7.32x faster |
| UTC -> TCG (2M) | 128.19 ms | 801.94 ms | 6.26x faster |
| TAN pix2world (2M) | 63.58 ms | 434.49 ms | 6.83x faster |
| TAN world2pix (2M) | 73.78 ms | 451.71 ms | 6.12x faster |
| SIP world2pix (500k) | 29.87 ms | 264.97 ms | 8.87x faster |

These values are the output of `pixi run bench` on this machine. Astropy 8.0.1
is the reference. Timings vary with hardware and system load; run the command
to measure them on another machine.

## How it works

All numerical kernels live in one Mojo compilation unit. Python normalizes
inputs to C-contiguous `float64` arrays, allocates outputs, and makes one
`ctypes` call per vectorized operation. Buffers cross the C ABI as integer
addresses and are reconstructed in Mojo as
`UnsafePointer[Float64, AnyOrigin[mut=True]]`; Python retains ownership for the
entire call.

Coordinates and time values use flat contiguous buffers. WCS accepts arbitrary
broadcast shapes in Python, flattens them for the kernel, and restores the
original shape. Point matrices use row-major `(n, 2)` layout, FITS matrices use
the conventional axis order, and SIP coefficients use dense row-major
`coefficient[i, j]` storage. No Mojo kernel allocates memory.

Coordinate and TAN loops use native-width `float64` SIMD loads and stores with
scalar remainder loops. Arrays of at least 262,144 elements are divided into
16,384-element chunks across four workers; smaller calls stay serial. SIP
values and both Jacobian rows are evaluated together with nested Horner
polynomials instead of repeated power loops.

Numerical behavior is checked in 45 tests, primarily against Astropy 8.0.1,
including FFI boundary validation, UTC
leap-day fractions, split-JD scale conversions, pole and antipode geometry,
both FITS pixel origins, TAN round trips, and cubic SIP forward/inverse paths.

## License

MIT
