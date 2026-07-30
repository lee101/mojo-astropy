"""Array-oriented astronomical time scales."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from ._lib import addr, f64, lib

_SCALES = {"utc": 0, "tai": 1, "tt": 2, "tcg": 3, "tdb": 4, "tcb": 5}
_UNIX_JD = 2440587.5


def _split_jd(jd):
    value = f64(jd)
    day = np.floor(value + 0.5)
    return day, value - day


def _parse_iso(value):
    values = np.asarray(value)
    flat = []
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for text in values.reshape(-1):
        parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        flat.append(_UNIX_JD + (parsed - epoch).total_seconds() / 86400.0)
    return np.asarray(flat).reshape(values.shape)


class Time:
    SCALES = tuple(_SCALES)

    def __init__(
        self,
        val,
        val2=None,
        format=None,
        scale=None,
        precision=None,
        in_subfmt=None,
        out_subfmt=None,
        location=None,
        copy=False,
    ):
        self.scale = (scale or "utc").lower()
        if self.scale not in _SCALES:
            raise ValueError(f"supported scales are {', '.join(_SCALES)}")
        self.format = format or ("isot" if np.asarray(val).dtype.kind in "USO" else "jd")
        self.precision = 3 if precision is None else precision
        self.in_subfmt = in_subfmt
        self.out_subfmt = out_subfmt
        self.location = location
        if self.format in ("isot", "iso"):
            jd = _parse_iso(val)
        elif self.format == "jd":
            jd = f64(val)
        elif self.format == "mjd":
            jd = f64(val) + 2400000.5
        elif self.format == "unix":
            jd = f64(val) / 86400.0 + _UNIX_JD
            if self.scale != "utc":
                raise ValueError("unix format is defined on the UTC scale")
        else:
            raise ValueError("covered formats are jd, mjd, unix, iso, and isot")
        self._jd1, self._jd2 = _split_jd(jd)
        if val2 is not None:
            extra = f64(val2)
            if self.format == "unix":
                extra = extra / 86400.0
            self._jd2 = self._jd2 + extra
            carry = np.floor(self._jd2 + 0.5)
            self._jd1, self._jd2 = self._jd1 + carry, self._jd2 - carry
        if copy:
            self._jd1, self._jd2 = self._jd1.copy(), self._jd2.copy()

    @classmethod
    def _from_split(cls, jd1, jd2, scale, format="jd", precision=3):
        result = cls.__new__(cls)
        result.scale = scale
        result.format = format
        result.precision = precision
        result.in_subfmt = None
        result.out_subfmt = None
        result.location = None
        result._jd1 = np.asarray(jd1)
        result._jd2 = np.asarray(jd2)
        return result

    @property
    def shape(self):
        return self._jd1.shape

    @property
    def isscalar(self):
        return self.shape == ()

    @property
    def jd1(self):
        return self._jd1[()] if self.isscalar else self._jd1

    @property
    def jd2(self):
        return self._jd2[()] if self.isscalar else self._jd2

    @property
    def jd(self):
        value = self._jd1 + self._jd2
        return value[()] if self.isscalar else value

    @property
    def mjd(self):
        return self.jd - 2400000.5

    def _to_scale(self, scale):
        scale = scale.lower()
        if scale not in _SCALES:
            raise ValueError(f"supported scales are {', '.join(_SCALES)}")
        if (self.scale == "utc" or scale == "utc") and np.any(
            self._jd1 + self._jd2 < 2441317.5
        ):
            raise ValueError("UTC scale conversion is covered from 1972-01-01 onward")
        if scale == self.scale:
            return Time._from_split(
                self._jd1.copy(), self._jd2.copy(), scale, self.format, self.precision
            )
        src1 = f64(self._jd1).reshape(-1)
        src2 = f64(self._jd2).reshape(-1)
        dst1, dst2 = np.empty_like(src1), np.empty_like(src2)
        if src1.size:
            lib().ma_time_convert(
                addr(src1),
                addr(src2),
                addr(dst1),
                addr(dst2),
                src1.size,
                _SCALES[self.scale],
                _SCALES[scale],
            )
        return Time._from_split(
            dst1.reshape(self.shape),
            dst2.reshape(self.shape),
            scale,
            self.format,
            self.precision,
        )

    @property
    def utc(self):
        return self._to_scale("utc")

    @property
    def tai(self):
        return self._to_scale("tai")

    @property
    def tt(self):
        return self._to_scale("tt")

    @property
    def tcg(self):
        return self._to_scale("tcg")

    @property
    def tdb(self):
        return self._to_scale("tdb")

    @property
    def tcb(self):
        return self._to_scale("tcb")

    @property
    def unix(self):
        utc = self.utc
        return (utc.jd - _UNIX_JD) * 86400.0

    @property
    def isot(self):
        utc = self.utc
        values = np.asarray(utc.unix)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        result = []
        for seconds in values.reshape(-1):
            dt = epoch + timedelta(seconds=float(seconds))
            result.append(
                dt.strftime("%Y-%m-%dT%H:%M:%S")
                + (f".{dt.microsecond:06d}"[: self.precision + 1] if self.precision else "")
            )
        array = np.asarray(result).reshape(values.shape)
        return array[()] if array.shape == () else array

    @property
    def iso(self):
        value = self.isot
        if np.ndim(value) == 0:
            return str(value).replace("T", " ")
        return np.char.replace(value, "T", " ")

    @property
    def value(self):
        return self.to_value(self.format)

    def to_value(self, format, subfmt="*"):
        if format == "jd":
            return self.jd
        if format == "mjd":
            return self.mjd
        if format == "unix":
            return self.unix
        if format == "isot":
            return self.isot
        if format == "iso":
            return self.iso
        raise ValueError("covered formats are jd, mjd, unix, iso, and isot")

    def __len__(self):
        return len(self._jd1)

    def __getitem__(self, item):
        return Time._from_split(
            self._jd1[item], self._jd2[item], self.scale, self.format, self.precision
        )

    def __repr__(self):
        return f"<Time object: scale='{self.scale}' format='{self.format}' value={self.value!r}>"
