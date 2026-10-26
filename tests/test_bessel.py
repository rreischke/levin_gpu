import jax
import numpy as np
import pytest
import scipy.special as sp

from levin import bessel

ELLS = np.array([0, 1, 2, 3, 5, 10, 20, 50, 100, 300, 1000, 3000])


def envelope_error(result, reference):
    """Max error relative to sqrt(J_ell^2 + J_{ell+1}^2), which never vanishes."""
    a, b = (np.asarray(r) for r in result)
    ra, rb = reference
    envelope = np.hypot(ra, rb)
    mask = envelope > 1e-280
    return (np.maximum(np.abs(a - ra), np.abs(b - rb))[mask] / envelope[mask]).max()


def test_spherical_matches_scipy():
    ell, z = np.meshgrid(ELLS, np.geomspace(1e-8, 1e4, 2000), indexing="ij")
    reference = sp.spherical_jn(ell, z), sp.spherical_jn(ell + 1, z)
    # Tolerance is set by scipy's own error at large ell; mpmath checks are tighter.
    assert envelope_error(bessel.bessel_pair(0, ell, z), reference) < 5e-12


def test_cylindrical_matches_scipy_moderate_z():
    # scipy.special.jv loses accuracy for z >~ 100, see the mpmath test for large z.
    ell, z = np.meshgrid(ELLS, np.geomspace(1e-8, 100, 2000), indexing="ij")
    reference = sp.jv(ell, z), sp.jv(ell + 1, z)
    assert envelope_error(bessel.bessel_pair(1, ell, z), reference) < 1e-12


@pytest.mark.parametrize("bessel_type", [0, 1])
def test_matches_mpmath_across_regimes(bessel_type):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    # Points on both sides of every regime boundary (z = 1, 25, ell + 1) and at large z.
    points = [(0, 0.999), (0, 1.0), (7, 0.5), (62, 0.009), (3, 24.9), (3, 25.1),
              (40, 40.5), (40, 41.5), (100, 4942.0), (1000, 1100.0), (1000, 900.0),
              (3000, 2431.0), (3000, 9.275e5), (5, 1e5)]
    ell = np.array([p[0] for p in points])
    z = np.array([p[1] for p in points])

    def exact(m, x):
        x = mp.mpf(float(x))
        if bessel_type == 0:
            return float(mp.sqrt(mp.pi / (2 * x)) * mp.besselj(m + 0.5, x, maxterms=10**7, maxprec=50000))
        return float(mp.besselj(m, x, maxterms=10**7, maxprec=50000))

    reference = (np.array([exact(int(l), x) for l, x in zip(ell, z)]),
                 np.array([exact(int(l) + 1, x) for l, x in zip(ell, z)]))
    assert envelope_error(bessel.bessel_pair(bessel_type, ell, z), reference) < 1e-13


@pytest.mark.parametrize("bessel_type", [0, 1])
def test_zero_argument(bessel_type):
    a, b = bessel.bessel_pair(bessel_type, np.array([0, 1, 5]), 0.0)
    np.testing.assert_array_equal(np.asarray(a), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(b), [0.0, 0.0, 0.0])


@pytest.mark.parametrize("bessel_type", [0, 1])
def test_deep_underflow_is_zero_not_nan(bessel_type):
    a, b = bessel.bessel_pair(bessel_type, 3000, np.array([1e-6, 0.5, 2.0, 100.0]))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(np.abs(np.asarray(a)) < 1e-280)


def test_public_wrappers_and_shapes():
    ell = np.array([[2], [7]])
    z = np.linspace(0.1, 50, 5)
    assert bessel.spherical_jn(ell, z).shape == (2, 5)
    np.testing.assert_allclose(bessel.cylindrical_jn(ell, z), sp.jv(ell, z), rtol=1e-12, atol=1e-15)
    w = bessel.levin_w(0, ell, z)
    assert w.shape == (2, 5, 2)
    np.testing.assert_allclose(w[..., 1], sp.spherical_jn(ell + 1, z), rtol=1e-12, atol=1e-15)


def test_traceable_under_jit_and_vmap():
    ell = np.arange(0, 40)
    z = np.linspace(0.5, 80, 40)
    jitted = jax.jit(lambda l, x: bessel.spherical_jn(l, x))(ell, z)
    mapped = jax.vmap(bessel.spherical_jn)(ell, z)
    np.testing.assert_allclose(jitted, sp.spherical_jn(ell, z), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(mapped, jitted, rtol=1e-15)
