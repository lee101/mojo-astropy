import numpy as np
import pytest

from astropy.time import Time as AstroTime

from mojoastropy import Time


def split_error_seconds(actual, expected):
    return (
        (actual.jd1 - expected.jd1) + (actual.jd2 - expected.jd2)
    ) * 86400.0


def test_utc_tai_parity_across_leap_seconds():
    jd = np.array(
        [
            2441317.5,
            2441499.49999,
            2441499.5,
            2457203.5,
            2457203.75,
            2457204.49999,
            2457204.5,
            2457753.5,
            2457753.99999,
            2457754.49999,
            2457754.5,
        ]
    )
    actual = Time(jd, format="jd", scale="utc").tai
    expected = AstroTime(jd, format="jd", scale="utc").tai
    assert np.max(np.abs(split_error_seconds(actual, expected))) < 2e-9


@pytest.mark.parametrize(
    ("scale", "tolerance_seconds"),
    [("tt", 2e-9), ("tcg", 2e-9), ("tdb", 4e-5), ("tcb", 4e-5)],
)
def test_time_scales_match_astropy(scale, tolerance_seconds):
    jd = np.linspace(2441317.5, 2460676.5, 10_000)
    actual = getattr(Time(jd, format="jd", scale="utc"), scale)
    expected = getattr(AstroTime(jd, format="jd", scale="utc"), scale)
    assert np.max(np.abs(split_error_seconds(actual, expected))) < tolerance_seconds


@pytest.mark.parametrize("scale", ["tai", "tt", "tcg", "tdb", "tcb"])
def test_scale_roundtrip(scale):
    jd = np.linspace(2451545.0, 2461545.0, 1000)
    original = Time(jd, format="jd", scale="utc")
    roundtrip = getattr(getattr(original, scale), "utc")
    assert np.max(np.abs((roundtrip.jd1 - original.jd1) + (roundtrip.jd2 - original.jd2))) < 3e-15


def test_mjd_and_unix_formats_match_astropy():
    unix = np.array([63_072_000.0, 946_684_800.0, 1_483_228_800.0, 1_700_000_000.25])
    actual = Time(unix, format="unix", scale="utc")
    expected = AstroTime(unix, format="unix", scale="utc")
    assert actual.mjd == pytest.approx(expected.mjd, abs=5e-10)
    assert actual.tt.jd == pytest.approx(expected.tt.jd, abs=5e-10)
    rebuilt = Time(actual.mjd, format="mjd", scale="utc")
    assert rebuilt.unix == pytest.approx(unix, abs=5e-5)
    split = Time(unix, val2=np.full(unix.shape, 0.25), format="unix", scale="utc")
    assert split.unix == pytest.approx(unix + 0.25, abs=5e-5)


def test_iso_input_and_output():
    values = ["2000-01-01T12:00:00", "2017-01-01T00:00:00", "2025-06-30T01:02:03"]
    actual = Time(values, format="isot", scale="utc", precision=0)
    expected = AstroTime(values, format="isot", scale="utc")
    assert actual.jd == pytest.approx(expected.jd, abs=5e-10)
    assert actual.isot.tolist() == values


def test_scalar_shape_index_and_value():
    vector = Time([58000.0, 58001.0], format="mjd", scale="utc")
    scalar = vector[0]
    assert scalar.isscalar
    assert scalar.value == pytest.approx(58000.0)
    assert len(vector) == 2


def test_pre_1972_utc_conversion_is_explicitly_rejected():
    with pytest.raises(ValueError, match="1972"):
        _ = Time(2440000.5, format="jd", scale="utc").tai


def test_empty_time_conversion():
    converted = Time([], format="jd", scale="utc").tt
    assert converted.shape == (0,)
