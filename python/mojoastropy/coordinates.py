"""Vectorized coordinate transforms compatible with Astropy's covered API."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib

_DEG = np.pi / 180.0
_ICRS_TO_GALACTIC = np.array(
    [
        [-0.05487565771259163, -0.8734370519556159, -0.48383507361671546],
        [0.4941094371927268, -0.4448297212232952, 0.7469821839866676],
        [-0.8676661375596576, -0.19807633727300053, 0.4559838136873016],
    ],
    dtype=np.float64,
)


class Quantity(np.ndarray):
    def __new__(cls, value, unit=""):
        obj = np.asarray(value, dtype=np.float64).view(cls)
        obj.unit = unit
        return obj

    def __array_finalize__(self, obj):
        self.unit = getattr(obj, "unit", "")

    @property
    def value(self):
        return np.asarray(self)

    def to_value(self, unit=None):
        unit_name = "" if unit is None else str(unit).lower()
        own_unit = str(self.unit).lower()
        if unit is None or unit_name in (own_unit, ""):
            return np.asarray(self)
        if own_unit in ("rad", "radian") and unit_name in ("deg", "degree"):
            return np.asarray(self) / _DEG
        if own_unit in ("rad", "radian") and unit_name == "hourangle":
            return np.asarray(self) / (15.0 * _DEG)
        if own_unit in ("deg", "degree") and unit_name in ("rad", "radian"):
            return np.asarray(self) * _DEG
        raise ValueError(f"cannot convert {self.unit!r} to {unit!r}")


class Angle(Quantity):
    def __new__(cls, value, unit="rad"):
        if hasattr(value, "to_value"):
            radians = value.to_value("rad")
        elif str(unit).lower() in ("deg", "degree", "degrees"):
            radians = np.asarray(value, dtype=np.float64) * _DEG
        elif str(unit).lower() == "hourangle":
            radians = np.asarray(value, dtype=np.float64) * 15.0 * _DEG
        else:
            radians = value
        return super().__new__(cls, radians, "rad")

    @property
    def rad(self):
        return np.asarray(self)

    @property
    def degree(self):
        return np.asarray(self) / _DEG

    @property
    def deg(self):
        return self.degree

    def wrap_at(self, angle):
        limit = float(Angle(angle).rad)
        return Angle((self.rad + limit) % (2.0 * limit) - limit)


Longitude = Angle
Latitude = Angle


def _angle_value(value):
    if hasattr(value, "to_value"):
        return value.to_value("rad")
    return value


def _broadcast(*values):
    return [f64(x).reshape(-1) for x in np.broadcast_arrays(*values)]


def _shape(*values):
    return np.broadcast_shapes(*(np.shape(v) for v in values))


def _restore(value, shape):
    result = value.reshape(shape)
    return result[()] if shape == () else result


def spherical_to_cartesian(r, lat, lon):
    shape = _shape(np.asanyarray(r), np.asanyarray(_angle_value(lat)), np.asanyarray(_angle_value(lon)))
    rv, latv, lonv = _broadcast(r, _angle_value(lat), _angle_value(lon))
    x, y, z = np.empty_like(rv), np.empty_like(rv), np.empty_like(rv)
    if rv.size:
        lib().ma_spherical_to_cartesian(
            addr(rv), addr(latv), addr(lonv), addr(x), addr(y), addr(z), rv.size
        )
    unit = str(getattr(r, "unit", ""))
    return tuple(Quantity(_restore(v, shape), unit) for v in (x, y, z))


def cartesian_to_spherical(x, y, z):
    shape = _shape(np.asanyarray(x), np.asanyarray(y), np.asanyarray(z))
    xv, yv, zv = _broadcast(x, y, z)
    r, lat, lon = np.empty_like(xv), np.empty_like(xv), np.empty_like(xv)
    if xv.size:
        lib().ma_cartesian_to_spherical(
            addr(xv), addr(yv), addr(zv), addr(r), addr(lat), addr(lon), xv.size
        )
    unit = str(getattr(x, "unit", ""))
    return (
        Quantity(_restore(r, shape), unit),
        Angle(_restore(lat, shape)),
        Angle(_restore(lon, shape)),
    )


def angular_separation(lon1, lat1, lon2, lat2):
    values = [_angle_value(v) for v in (lon1, lat1, lon2, lat2)]
    shape = _shape(*values)
    a, b, c, d = _broadcast(*values)
    result = np.empty_like(a)
    if result.size:
        lib().ma_angular_separation(
            addr(a), addr(b), addr(c), addr(d), addr(result), result.size
        )
    return _restore(result, shape)


def position_angle(lon1, lat1, lon2, lat2):
    values = [_angle_value(v) for v in (lon1, lat1, lon2, lat2)]
    shape = _shape(*values)
    a, b, c, d = _broadcast(*values)
    result = np.empty_like(a)
    if result.size:
        lib().ma_position_angle(
            addr(a), addr(b), addr(c), addr(d), addr(result), result.size
        )
    return Angle(_restore(result, shape))


def rotation_matrix(angle, axis="z", unit=None):
    if hasattr(angle, "to_value"):
        theta = np.asarray(angle.to_value("rad"))
    else:
        theta = np.asarray(angle, dtype=float) * (_DEG if unit is None or str(unit).startswith("deg") else 1.0)
    c, s = np.cos(theta), np.sin(theta)
    result = np.zeros(theta.shape + (3, 3))
    result[..., 0, 0] = result[..., 1, 1] = result[..., 2, 2] = 1.0
    index = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}
    if isinstance(axis, str):
        if axis not in index:
            raise ValueError("axis must be 'x', 'y', 'z', or a three-vector")
        i, j = index[axis]
        result[..., i, i] = result[..., j, j] = c
        result[..., i, j], result[..., j, i] = s, -s
        return result
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    k = np.array(
        [[0.0, vector[2], -vector[1]], [-vector[2], 0.0, vector[0]], [vector[1], -vector[0], 0.0]]
    )
    eye = np.eye(3)
    return c[..., None, None] * eye + (1 - c[..., None, None]) * np.outer(vector, vector) + s[..., None, None] * k


def _rotate(lon, lat, matrix):
    shape = _shape(lon, lat)
    lonv, latv = _broadcast(lon, lat)
    matrix = f64(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3)")
    dst_lon, dst_lat = np.empty_like(lonv), np.empty_like(latv)
    if lonv.size:
        lib().ma_rotate_spherical(
            addr(lonv), addr(latv), addr(matrix), addr(dst_lon), addr(dst_lat), lonv.size
        )
        np.mod(dst_lon, 2.0 * np.pi, out=dst_lon)
    return _restore(dst_lon, shape), _restore(dst_lat, shape)


class ICRS:
    name = "icrs"


class Galactic:
    name = "galactic"


def _frame_name(frame):
    if frame is None:
        return "icrs"
    if isinstance(frame, str):
        return frame.lower()
    if isinstance(frame, type):
        return frame.name
    return frame.name


class SkyCoord:
    def __init__(self, *args, copy=True, **kwargs):
        frame = _frame_name(kwargs.pop("frame", None))
        unit = kwargs.pop("unit", None)
        if kwargs and not set(kwargs) <= {"ra", "dec", "l", "b"}:
            raise TypeError(f"unsupported SkyCoord arguments: {sorted(kwargs)}")
        if args:
            if len(args) != 2:
                raise TypeError("covered SkyCoord accepts two coordinate arrays")
            first, second = args
        elif frame == "galactic" or "l" in kwargs:
            first, second = kwargs.get("l"), kwargs.get("b")
            frame = "galactic"
        else:
            first, second = kwargs.get("ra"), kwargs.get("dec")
        if first is None or second is None:
            raise TypeError("both longitude and latitude are required")
        units = unit if isinstance(unit, (tuple, list)) else (unit, unit)
        self._lon = Angle(first, units[0] or ("rad" if not hasattr(first, "to_value") else "rad"))
        self._lat = Angle(second, units[1] or ("rad" if not hasattr(second, "to_value") else "rad"))
        self.frame = Galactic() if frame == "galactic" else ICRS()
        if copy:
            self._lon = self._lon.copy()
            self._lat = self._lat.copy()

    @property
    def shape(self):
        return np.broadcast_shapes(self._lon.shape, self._lat.shape)

    @property
    def ra(self):
        if self.frame.name != "icrs":
            raise AttributeError("ra is only available in the ICRS frame")
        return self._lon

    @property
    def dec(self):
        if self.frame.name != "icrs":
            raise AttributeError("dec is only available in the ICRS frame")
        return self._lat

    @property
    def l(self):
        if self.frame.name != "galactic":
            raise AttributeError("l is only available in the Galactic frame")
        return self._lon

    @property
    def b(self):
        if self.frame.name != "galactic":
            raise AttributeError("b is only available in the Galactic frame")
        return self._lat

    def transform_to(self, frame):
        target = _frame_name(frame)
        if target == self.frame.name:
            return SkyCoord(self._lon, self._lat, unit="rad", frame=target)
        if {target, self.frame.name} != {"icrs", "galactic"}:
            raise ValueError("covered frames are ICRS and Galactic")
        matrix = _ICRS_TO_GALACTIC if target == "galactic" else _ICRS_TO_GALACTIC.T.copy()
        lon, lat = _rotate(self._lon.rad, self._lat.rad, matrix)
        return SkyCoord(lon, lat, unit="rad", frame=target, copy=False)

    @property
    def icrs(self):
        return self.transform_to("icrs")

    @property
    def galactic(self):
        return self.transform_to("galactic")

    def separation(self, other):
        return Angle(angular_separation(self._lon, self._lat, other._lon, other._lat))
