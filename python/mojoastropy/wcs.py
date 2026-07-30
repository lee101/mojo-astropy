"""Two-dimensional FITS linear/TAN/SIP world coordinate systems."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class NoConvergence(RuntimeError):
    pass


class Sip:
    def __init__(self, a, b, ap, bp, crpix):
        self.a = f64(a)
        self.b = f64(b)
        self.ap = None if ap is None else f64(ap)
        self.bp = None if bp is None else f64(bp)
        self.crpix = f64(crpix)
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError("SIP coefficient arrays must be two-dimensional")
        if self.a.shape[0] != self.a.shape[1] or self.b.shape[0] != self.b.shape[1]:
            raise ValueError("SIP coefficient arrays must be square")
        if self.crpix.shape != (2,):
            raise ValueError("SIP crpix must contain exactly two values")
        self.a_order = self.a.shape[0] - 1
        self.b_order = self.b.shape[0] - 1


class _Wcsprm:
    def __init__(self):
        self.crpix = np.array([0.0, 0.0])
        self.crval = np.array([0.0, 0.0])
        self.cdelt = np.array([1.0, 1.0])
        self.pc = np.eye(2)
        self.cd = None
        self.ctype = ["", ""]
        self.cunit = ["deg", "deg"]


def _get(header, key, default=None):
    try:
        return header[key]
    except (KeyError, TypeError):
        return default


def _sip_from_header(header, crpix):
    order = max(int(_get(header, "A_ORDER", 0)), int(_get(header, "B_ORDER", 0)))
    if order < 2:
        return None
    a = np.zeros((order + 1, order + 1))
    b = np.zeros_like(a)
    for i in range(order + 1):
        for j in range(order + 1 - i):
            a[i, j] = float(_get(header, f"A_{i}_{j}", 0.0))
            b[i, j] = float(_get(header, f"B_{i}_{j}", 0.0))
    return Sip(a, b, None, None, crpix)


class WCS:
    def __init__(
        self,
        header=None,
        fobj=None,
        key=" ",
        minerr=0.0,
        relax=True,
        naxis=None,
        keysel=None,
        colsel=None,
        fix=True,
        translate_units="",
        _do_set=True,
        preserve_units=False,
    ):
        if naxis not in (None, 2):
            raise ValueError("the covered WCS implementation is two-dimensional")
        self.naxis = 2
        self.wcs = _Wcsprm()
        self.sip = None
        self.pixel_shape = None
        self.array_shape = None
        if header is not None:
            self.wcs.crpix = np.array(
                [float(_get(header, "CRPIX1", 0.0)), float(_get(header, "CRPIX2", 0.0))]
            )
            self.wcs.crval = np.array(
                [float(_get(header, "CRVAL1", 0.0)), float(_get(header, "CRVAL2", 0.0))]
            )
            self.wcs.cdelt = np.array(
                [float(_get(header, "CDELT1", 1.0)), float(_get(header, "CDELT2", 1.0))]
            )
            self.wcs.pc = np.array(
                [
                    [float(_get(header, "PC1_1", 1.0)), float(_get(header, "PC1_2", 0.0))],
                    [float(_get(header, "PC2_1", 0.0)), float(_get(header, "PC2_2", 1.0))],
                ]
            )
            if _get(header, "CD1_1") is not None:
                self.wcs.cd = np.array(
                    [
                        [float(_get(header, "CD1_1", 1.0)), float(_get(header, "CD1_2", 0.0))],
                        [float(_get(header, "CD2_1", 0.0)), float(_get(header, "CD2_2", 1.0))],
                    ]
                )
            self.wcs.ctype = [
                str(_get(header, "CTYPE1", "")),
                str(_get(header, "CTYPE2", "")),
            ]
            n1, n2 = _get(header, "NAXIS1"), _get(header, "NAXIS2")
            if n1 is not None and n2 is not None:
                self.pixel_shape = (int(n1), int(n2))
                self.array_shape = (int(n2), int(n1))
            self.sip = _sip_from_header(header, self.wcs.crpix)

    def _cd(self):
        if self.wcs.cd is not None:
            return np.asarray(self.wcs.cd, dtype=np.float64)
        return np.asarray(self.wcs.cdelt, dtype=np.float64)[:, None] * np.asarray(
            self.wcs.pc, dtype=np.float64
        )

    def _kernel_params(self):
        cd = self._cd()
        if cd.shape != (2, 2) or not np.all(np.isfinite(cd)):
            raise ValueError("CD/PC matrix must be a finite 2 by 2 matrix")
        if abs(float(np.linalg.det(cd))) <= np.finfo(np.float64).tiny:
            raise ValueError("CD/PC matrix is singular")
        tan = int(
            "-TAN" in str(self.wcs.ctype[0]).upper()
            and "-TAN" in str(self.wcs.ctype[1]).upper()
        )
        if self.sip is None:
            order, a, b = 0, np.zeros((1, 1)), np.zeros((1, 1))
        else:
            order = max(self.sip.a_order, self.sip.b_order)
            shape = (order + 1, order + 1)
            a, b = np.zeros(shape), np.zeros(shape)
            a[: self.sip.a.shape[0], : self.sip.a.shape[1]] = self.sip.a
            b[: self.sip.b.shape[0], : self.sip.b.shape[1]] = self.sip.b
        return cd, tan, f64(a), f64(b), order

    @staticmethod
    def _parse_args(args):
        if len(args) == 2:
            points, origin = args
            points = f64(points)
            if points.shape[-1] != 2:
                raise ValueError("coordinate array must have final dimension 2")
            return points[..., 0], points[..., 1], int(origin), True
        if len(args) == 3:
            return args[0], args[1], int(args[2]), False
        raise TypeError("expected (N, 2), origin or x, y, origin")

    def _pix2world(self, x, y, origin, use_sip):
        bx, by = np.broadcast_arrays(x, y)
        shape = bx.shape
        xv, yv = f64(bx).reshape(-1), f64(by).reshape(-1)
        lon, lat = np.empty_like(xv), np.empty_like(yv)
        cd, tan, a, b, order = self._kernel_params()
        if xv.size:
            lib().ma_wcs_pix2world(
                addr(xv),
                addr(yv),
                addr(lon),
                addr(lat),
                xv.size,
                origin,
                *self.wcs.crpix,
                *self.wcs.crval,
                *cd.reshape(-1),
                tan,
                int(use_sip and self.sip is not None),
                addr(a),
                addr(b),
                order,
            )
        return lon.reshape(shape), lat.reshape(shape)

    def _world2pix(self, lon, lat, origin, use_sip, tolerance, maxiter, quiet=False):
        if tolerance <= 0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be a positive finite number")
        if maxiter < 1:
            raise ValueError("maxiter must be at least 1")
        blon, blat = np.broadcast_arrays(lon, lat)
        shape = blon.shape
        lonv, latv = f64(blon).reshape(-1), f64(blat).reshape(-1)
        x, y = np.empty_like(lonv), np.empty_like(latv)
        converged = np.empty_like(lonv)
        cd, tan, a, b, order = self._kernel_params()
        if lonv.size:
            lib().ma_wcs_world2pix(
                addr(lonv),
                addr(latv),
                addr(x),
                addr(y),
                addr(converged),
                lonv.size,
                origin,
                *self.wcs.crpix,
                *self.wcs.crval,
                *cd.reshape(-1),
                tan,
                int(use_sip and self.sip is not None),
                addr(a),
                addr(b),
                order,
                tolerance,
                maxiter,
            )
        if not quiet and (
            not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.all(converged != 0.0)
        ):
            failed = int(np.count_nonzero(converged == 0.0))
            raise NoConvergence(
                f"world-to-pixel iteration failed to converge for {failed} coordinate(s)"
            )
        return x.reshape(shape), y.reshape(shape)

    def all_pix2world(self, *args, **kwargs):
        x, y, origin, joined = self._parse_args(args)
        lon, lat = self._pix2world(x, y, origin, True)
        return np.stack((lon, lat), axis=-1) if joined else (lon, lat)

    def wcs_pix2world(self, *args, **kwargs):
        x, y, origin, joined = self._parse_args(args)
        lon, lat = self._pix2world(x, y, origin, False)
        return np.stack((lon, lat), axis=-1) if joined else (lon, lat)

    def all_world2pix(
        self,
        *args,
        tolerance=0.0001,
        maxiter=20,
        adaptive=False,
        detect_divergence=True,
        quiet=False,
        **kwargs,
    ):
        lon, lat, origin, joined = self._parse_args(args)
        x, y = self._world2pix(lon, lat, origin, True, tolerance, maxiter, quiet)
        return np.stack((x, y), axis=-1) if joined else (x, y)

    def wcs_world2pix(self, *args, **kwargs):
        lon, lat, origin, joined = self._parse_args(args)
        x, y = self._world2pix(lon, lat, origin, False, 1e-10, 1)
        return np.stack((x, y), axis=-1) if joined else (x, y)

    def pixel_to_world_values(self, *pixel_arrays):
        return self._pix2world(pixel_arrays[0], pixel_arrays[1], 0, True)

    def world_to_pixel_values(self, *world_arrays):
        return self._world2pix(world_arrays[0], world_arrays[1], 0, True, 0.0001, 20)

    def to_header(self, relax=None, key=None):
        cd = self._cd()
        return {
            "WCSAXES": 2,
            "CRPIX1": self.wcs.crpix[0],
            "CRPIX2": self.wcs.crpix[1],
            "CRVAL1": self.wcs.crval[0],
            "CRVAL2": self.wcs.crval[1],
            "CTYPE1": self.wcs.ctype[0],
            "CTYPE2": self.wcs.ctype[1],
            "CD1_1": cd[0, 0],
            "CD1_2": cd[0, 1],
            "CD2_1": cd[1, 0],
            "CD2_2": cd[1, 1],
        }
