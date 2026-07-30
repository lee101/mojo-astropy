import numpy as np
import pytest

from astropy.wcs import Sip as AstroSip
from astropy.wcs import WCS as AstroWCS

from mojoastropy import NoConvergence, Sip, WCS


@pytest.fixture(scope="module")
def tan_header():
    return {
        "NAXIS": 2,
        "NAXIS1": 1024,
        "NAXIS2": 768,
        "CRPIX1": 512.5,
        "CRPIX2": 384.5,
        "CRVAL1": 202.4823228,
        "CRVAL2": 47.17511893,
        "CD1_1": -0.0002777778,
        "CD1_2": 2.0e-6,
        "CD2_1": -1.0e-6,
        "CD2_2": 0.0002777778,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
    }


@pytest.fixture(scope="module")
def sip_header(tan_header):
    return {
        **tan_header,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "A_ORDER": 3,
        "B_ORDER": 3,
        "A_2_0": 2.0e-5,
        "A_0_2": -1.0e-6,
        "A_1_1": 3.0e-6,
        "A_3_0": 1.0e-9,
        "B_2_0": -1.0e-6,
        "B_0_2": 1.5e-5,
        "B_1_1": -2.0e-6,
        "B_0_3": -8.0e-10,
    }


def test_linear_wcs_matches_astropy():
    header = {
        "CRPIX1": 10.0,
        "CRPIX2": 20.0,
        "CRVAL1": 100.0,
        "CRVAL2": -30.0,
        "CDELT1": -0.05,
        "CDELT2": 0.08,
        "PC1_1": 0.9,
        "PC1_2": -0.1,
        "PC2_1": 0.2,
        "PC2_2": 1.1,
        "CTYPE1": "LINEAR",
        "CTYPE2": "LINEAR",
    }
    points = np.array([[0.0, 0.0], [10.0, 20.0], [100.5, -4.0]])
    assert WCS(header).all_pix2world(points, 0) == pytest.approx(
        AstroWCS(header).all_pix2world(points, 0), abs=2e-14
    )


@pytest.mark.parametrize("origin", [0, 1])
def test_tan_pix2world_matches_astropy(tan_header, origin):
    rng = np.random.default_rng(10)
    points = rng.uniform([-200, -200], [1200, 1000], (20_000, 2))
    actual = WCS(tan_header).all_pix2world(points, origin)
    expected = AstroWCS(tan_header).all_pix2world(points, origin)
    assert actual == pytest.approx(expected, abs=8e-13)


def test_tan_separate_array_call_style(tan_header):
    x = np.arange(12.0).reshape(3, 4) * 20
    y = np.linspace(10, 500, 4)
    actual = WCS(tan_header).all_pix2world(x, y, 0)
    expected = AstroWCS(tan_header).all_pix2world(x, y, 0)
    assert actual[0].shape == (3, 4)
    assert actual[0] == pytest.approx(expected[0], abs=8e-13)
    assert actual[1] == pytest.approx(expected[1], abs=8e-13)


def test_tan_world2pix_matches_and_roundtrips(tan_header):
    rng = np.random.default_rng(11)
    points = rng.uniform([0, 0], [1024, 768], (10_000, 2))
    reference, actual = AstroWCS(tan_header), WCS(tan_header)
    world = reference.all_pix2world(points, 0)
    assert actual.all_world2pix(world, 0) == pytest.approx(
        reference.all_world2pix(world, 0), abs=2e-9
    )
    assert actual.all_world2pix(actual.all_pix2world(points, 0), 0) == pytest.approx(
        points, abs=3e-10
    )


def test_sip_forward_matches_astropy(sip_header):
    rng = np.random.default_rng(12)
    points = rng.uniform([-100, -100], [1100, 900], (20_000, 2))
    actual = WCS(sip_header).all_pix2world(points, 0)
    expected = AstroWCS(sip_header).all_pix2world(points, 0)
    assert actual == pytest.approx(expected, abs=8e-13)


def test_sip_inverse_matches_astropy_and_roundtrips(sip_header):
    rng = np.random.default_rng(13)
    points = rng.uniform([0, 0], [1024, 768], (10_000, 2))
    reference, actual = AstroWCS(sip_header), WCS(sip_header)
    world = reference.all_pix2world(points, 0)
    got = actual.all_world2pix(world, 0, tolerance=1e-10, maxiter=30)
    assert got == pytest.approx(points, abs=3e-9)
    assert got == pytest.approx(
        reference.all_world2pix(world, 0, tolerance=1e-10, maxiter=30), abs=3e-8
    )


def test_sip_inverse_parallel_threshold_and_simd_tail(sip_header):
    count = 262_147
    points = np.column_stack(
        (np.linspace(0.0, 1024.0, count), np.linspace(768.0, 0.0, count))
    )
    actual = WCS(sip_header)
    world = actual.all_pix2world(points, 0)
    got = actual.all_world2pix(world, 0, tolerance=1e-10, maxiter=30)
    assert got == pytest.approx(points, abs=3e-9)


def test_wcs_methods_deliberately_skip_sip(sip_header):
    points = np.array([[100.0, 200.0], [900.0, 600.0]])
    actual, reference = WCS(sip_header), AstroWCS(sip_header)
    assert actual.wcs_pix2world(points, 0) == pytest.approx(
        reference.wcs_pix2world(points, 0), abs=8e-13
    )


def test_pixel_world_value_aliases_and_shapes(tan_header):
    wcs = WCS(tan_header)
    lon, lat = wcs.pixel_to_world_values([1, 2, 3], 5)
    x, y = wcs.world_to_pixel_values(lon, lat)
    assert x == pytest.approx([1, 2, 3], abs=3e-10)
    assert y == pytest.approx([5, 5, 5], abs=3e-10)
    assert wcs.pixel_shape == (1024, 768)
    assert wcs.array_shape == (768, 1024)


def test_programmatic_wcs_and_sip_objects():
    ours, theirs = WCS(naxis=2), AstroWCS(naxis=2)
    for w in (ours, theirs):
        w.wcs.crpix = [100.0, 100.0]
        w.wcs.crval = [30.0, 40.0]
        w.wcs.cdelt = [-0.01, 0.01]
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[2, 0], b[0, 2] = 2e-5, -1e-5
    ours.sip = Sip(a, b, None, None, [100, 100])
    theirs.sip = AstroSip(a, b, None, None, [100, 100])
    points = np.array([[0.0, 0.0], [100.0, 100.0], [250.0, 300.0]])
    assert ours.all_pix2world(points, 0) == pytest.approx(
        theirs.all_pix2world(points, 0), abs=8e-13
    )


def test_empty_wcs_input_and_invalid_kernel_parameters(tan_header):
    assert WCS(tan_header).all_pix2world(np.empty((0, 2)), 0).shape == (0, 2)
    singular = WCS(tan_header)
    singular.wcs.cd = np.zeros((2, 2))
    with pytest.raises(ValueError, match="singular"):
        singular.all_pix2world([[1.0, 2.0]], 0)
    with pytest.raises(ValueError, match="maxiter"):
        WCS(tan_header).all_world2pix([[1.0, 2.0]], 0, maxiter=0)


def test_nonfinite_inverse_is_reported_unless_quiet(tan_header):
    wcs = WCS(tan_header)
    with pytest.raises(NoConvergence, match="failed to converge"):
        wcs.all_world2pix([[np.nan, 2.0]], 0)
    result = wcs.all_world2pix([[np.nan, 2.0]], 0, quiet=True)
    assert np.isnan(result).any()


def test_invalid_sip_shapes_are_rejected():
    with pytest.raises(ValueError, match="square"):
        Sip(np.zeros((2, 3)), np.zeros((3, 3)), None, None, [1, 1])
    with pytest.raises(TypeError, match="complex"):
        WCS().all_pix2world(np.array([[1 + 2j, 3]]), 0)
