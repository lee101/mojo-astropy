import numpy as np
import pytest

import astropy.units as u
from astropy import coordinates as astro
from astropy.coordinates.matrix_utilities import rotation_matrix as astro_rotation_matrix

import mojoastropy as ma


@pytest.fixture(scope="module")
def spherical():
    rng = np.random.default_rng(3)
    return (
        rng.uniform(0.01, 100.0, 10_003),
        rng.uniform(-np.pi / 2, np.pi / 2, 10_003),
        rng.uniform(-4 * np.pi, 4 * np.pi, 10_003),
    )


def test_spherical_to_cartesian_matches_astropy(spherical):
    expected = astro.spherical_to_cartesian(*spherical)
    actual = ma.spherical_to_cartesian(*spherical)
    for got, want in zip(actual, expected):
        assert np.allclose(got, want.value, rtol=2e-15, atol=6e-14)


def test_spherical_to_cartesian_broadcasts():
    r = np.array([[1.0], [2.0]])
    lat = np.array([0.1, 0.2, 0.3])
    actual = ma.spherical_to_cartesian(r, lat, 0.5)
    expected = astro.spherical_to_cartesian(r, lat, 0.5)
    assert actual[0].shape == (2, 3)
    for got, want in zip(actual, expected):
        assert np.allclose(got, want.value)


def test_empty_inputs_do_not_cross_ffi_boundary():
    empty = np.array([], dtype=np.float64)
    assert ma.spherical_to_cartesian(empty, empty, empty)[0].shape == (0,)
    assert ma.angular_separation(empty, empty, empty, empty).shape == (0,)


def test_lossy_complex_and_large_integer_inputs_are_rejected():
    with pytest.raises(TypeError, match="complex"):
        ma.angular_separation(np.array([1 + 2j]), 0, 0, 0)
    with pytest.raises(ValueError, match="exact float64"):
        ma.spherical_to_cartesian(np.array([2**53 + 1]), 0, 0)
    with pytest.raises(TypeError, match="wider than float64"):
        ma.angular_separation(np.array([1], dtype=np.longdouble), 0, 0, 0)


def test_cartesian_to_spherical_matches_astropy(spherical):
    xyz = ma.spherical_to_cartesian(*spherical)
    expected = astro.cartesian_to_spherical(*xyz)
    actual = ma.cartesian_to_spherical(*xyz)
    assert np.allclose(actual[0], expected[0].value, atol=2e-14)
    assert np.allclose(actual[1].rad, expected[1].rad, atol=3e-15)
    assert np.allclose(actual[2].rad, expected[2].rad, atol=3e-15)


def test_angular_separation_matches_at_poles_and_antipodes():
    lon1 = np.array([0.0, 0.1, 1.0, np.pi])
    lat1 = np.array([np.pi / 2, -np.pi / 2, 0.2, 0.0])
    lon2 = np.array([2.0, -1.0, 1.0 + np.pi, 0.0])
    lat2 = np.array([np.pi / 2, -np.pi / 2, -0.2, 0.0])
    expected = astro.angular_separation(lon1, lat1, lon2, lat2)
    assert ma.angular_separation(lon1, lat1, lon2, lat2) == pytest.approx(
        expected, abs=2e-15
    )


def test_angular_separation_simd_tail():
    rng = np.random.default_rng(40)
    args = [
        rng.uniform(-np.pi, np.pi, 7),
        rng.uniform(-np.pi / 2, np.pi / 2, 7),
        rng.uniform(-np.pi, np.pi, 7),
        rng.uniform(-np.pi / 2, np.pi / 2, 7),
    ]
    assert ma.angular_separation(*args) == pytest.approx(
        astro.angular_separation(*args), abs=2e-15
    )


def test_angular_separation_parallel_threshold_and_tail():
    count = 262_147
    lon1 = np.linspace(-np.pi, np.pi, count)
    lat1 = np.linspace(-1.2, 1.2, count)
    lon2 = lon1[::-1].copy()
    lat2 = lat1[::-1].copy()
    actual = ma.angular_separation(lon1, lat1, lon2, lat2)
    assert actual == pytest.approx(
        astro.angular_separation(lon1, lat1, lon2, lat2), abs=2e-15
    )


def test_position_angle_matches_astropy():
    rng = np.random.default_rng(4)
    args = [rng.uniform(-1.4, 1.4, 2000) for _ in range(4)]
    expected = astro.position_angle(*args)
    actual = ma.position_angle(*args)
    assert actual.rad == pytest.approx(expected.rad, abs=2e-15)


@pytest.mark.parametrize("axis", ["x", "y", "z", [1.0, 2.0, 3.0]])
def test_rotation_matrix_matches_astropy(axis):
    expected = astro_rotation_matrix(37.5, axis=axis)
    actual = ma.rotation_matrix(37.5, axis=axis)
    assert actual == pytest.approx(expected, abs=2e-15)


def test_angle_units_and_wrapping():
    angle = ma.Angle([0.0, 180.0, 370.0], unit="deg")
    assert angle.rad == pytest.approx([0.0, np.pi, 370 * np.pi / 180])
    assert angle.wrap_at(ma.Angle(180, "deg")).deg == pytest.approx([0, -180, 10])
    assert ma.Angle(12, unit="hourangle").deg == pytest.approx(180.0)
    assert ma.Angle(45, unit=u.deg).deg == pytest.approx(45.0)


def test_skycoord_icrs_to_galactic_matches_astropy():
    rng = np.random.default_rng(5)
    ra = rng.uniform(0.0, 360.0, 20_000)
    dec = rng.uniform(-89.0, 89.0, 20_000)
    expected = astro.SkyCoord(ra * u.deg, dec * u.deg).galactic
    actual = ma.SkyCoord(ra, dec, unit="deg", frame="icrs").galactic
    lon_error = (actual.l.deg - expected.l.deg + 180.0) % 360.0 - 180.0
    assert np.all((actual.l.deg >= 0.0) & (actual.l.deg < 360.0))
    assert np.max(np.abs(lon_error)) < 8e-12
    assert np.max(np.abs(actual.b.deg - expected.b.deg)) < 8e-12


def test_skycoord_rotation_simd_tail_wraps_longitude():
    ra = np.linspace(0.0, 359.0, 7)
    dec = np.linspace(-70.0, 70.0, 7)
    expected = astro.SkyCoord(ra * u.deg, dec * u.deg).galactic
    actual = ma.SkyCoord(ra, dec, unit="deg").galactic
    lon_error = (actual.l.deg - expected.l.deg + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(lon_error)) < 8e-12
    assert np.max(np.abs(actual.b.deg - expected.b.deg)) < 8e-12


def test_skycoord_galactic_roundtrip():
    lon = np.linspace(0.0, 359.0, 1000)
    lat = np.linspace(-80.0, 80.0, 1000)
    coord = ma.SkyCoord(l=lon, b=lat, unit="deg", frame="galactic")
    back = coord.icrs.galactic
    error = (back.l.deg - lon + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(error)) < 2e-12
    assert np.max(np.abs(back.b.deg - lat)) < 2e-12


def test_skycoord_copy_does_not_alias_radian_inputs():
    lon = np.array([0.1, 0.2])
    lat = np.array([-0.3, 0.4])
    coord = ma.SkyCoord(lon, lat, unit="rad")
    lon[:] = 1.0
    lat[:] = 1.0
    assert coord.ra.rad == pytest.approx([0.1, 0.2])
    assert coord.dec.rad == pytest.approx([-0.3, 0.4])


def test_skycoord_separation_matches_astropy():
    a = ma.SkyCoord([10, 20, 30], [-5, 0, 5], unit="deg")
    b = ma.SkyCoord([11, 18, 35], [-4, 2, 0], unit="deg")
    expected = astro.SkyCoord([10, 20, 30] * u.deg, [-5, 0, 5] * u.deg).separation(
        astro.SkyCoord([11, 18, 35] * u.deg, [-4, 2, 0] * u.deg)
    )
    assert a.separation(b).rad == pytest.approx(expected.rad, abs=2e-15)
