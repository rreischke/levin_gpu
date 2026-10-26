import warnings

import numpy as np
import pytest
import scipy.special as sp
from scipy import integrate

from levin import collocation, interpolation

BESSEL = {0: sp.spherical_jn, 1: sp.jv}


def make_spline():
    x = np.geomspace(1e-3, 30, 400)
    return interpolation.build(x, (x ** 3 + x ** 2 + x)[:, None], log_x=True, log_y=True)


def levin_estimate(a, b, k, ell, bessel_type, n_col, spline):
    k, ell = np.array([k], dtype=float), np.array([ell])
    w_a, w_b = (collocation.endpoint_w(bessel_type, ell, k, v) for v in (a, b))
    return float(collocation.integrate(a, b, k, ell, w_a, w_b, 0, spline, bessel_type, n_col))


def test_weights_equal_direct_collocation_solve():
    """L . G^{-1} F (pylevin's order of operations) equals v . f(x_j)."""
    a, b, n_col = 0.5, 4.0, 8
    k, ell = np.array([10.0]), np.array([3])
    matrix, x = collocation.collocation_matrix(a, b, k, ell, 0, n_col)
    w_a, w_b = (collocation.endpoint_w(0, ell, k, v) for v in (a, b))
    f = np.cos(np.asarray(x)) + 2
    rhs = np.concatenate([0.5 * (b - a) * f, np.zeros(n_col)])
    direct = collocation.endpoint_functional(w_a, w_b, n_col) @ np.linalg.solve(np.asarray(matrix), rhs)
    _, v = collocation.quadrature_weights(a, b, k, ell, w_a, w_b, 0, n_col)
    assert float(np.asarray(v) @ f) == pytest.approx(float(direct), rel=1e-12)


@pytest.mark.parametrize("bessel_type", [0, 1])
@pytest.mark.parametrize("interval", [(1.0, 2.0, 3.0, 5), (0.5, 4.0, 10.0, 2), (2.0, 3.0, 50.0, 20)])
def test_single_interval_against_quad(bessel_type, interval):
    a, b, k, ell = interval
    spline = make_spline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        expected = integrate.quad(lambda s: (s ** 3 + s ** 2 + s) * BESSEL[bessel_type](ell, k * s), a, b,
                                  epsabs=0, epsrel=1e-12, limit=500)[0]
    # One polynomial over many spline knots: accuracy is limited by the
    # piecewise-cubic interpolant, not by the solver.
    assert levin_estimate(a, b, k, ell, bessel_type, 16, spline) == pytest.approx(expected, rel=2e-6)


def test_empty_and_reversed_intervals_are_zero():
    spline = make_spline()
    assert levin_estimate(2.0, 2.0, 5.0, 1, 0, 8, spline) == 0.0
    assert levin_estimate(3.0, 2.0, 5.0, 1, 0, 8, spline) == 0.0


def test_batched_shapes():
    a = np.full((3, 4), 1.0)
    b = np.full((3, 4), 2.0)
    k = (np.geomspace(1, 100, 4)[None, :] * np.ones((3, 1)))[..., None]
    ell = (np.arange(3)[:, None] * np.ones((1, 4), dtype=int))[..., None]
    w_a, w_b = (collocation.endpoint_w(1, ell, k, v) for v in (a, b))
    x, v = collocation.quadrature_weights(a, b, k, ell, w_a, w_b, 1, 8)
    assert x.shape == v.shape == (3, 4, 8)


def test_interpolatory_weights_are_exact_for_polynomials():
    for n_col in (4, 8, 16):
        y = np.asarray(collocation.chebyshev.nodes(n_col))
        q = collocation.interpolatory_weights(n_col)
        for degree in range(n_col):
            exact = (1 - (-1) ** (degree + 1)) / (degree + 1)
            assert q @ y ** degree == pytest.approx(exact, abs=1e-13)


@pytest.mark.parametrize("bessel_type", [0, 1])
def test_non_oscillatory_piece_is_accurate(bessel_type):
    """k (b - a) = 0.05: Levin's matrix is nearly singular, interpolatory quadrature is not."""
    a, b, k, ell = 2.93, 2.97, 1.376, 5
    spline = make_spline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        expected = integrate.quad(lambda s: float(interpolation.evaluate(spline, s, 0)) * BESSEL[bessel_type](ell, k * s),
                                  a, b, epsabs=0, epsrel=1e-13)[0]
    k_, ell_ = np.array([k]), np.array([ell])
    w_a, w_b = (collocation.endpoint_w(bessel_type, ell_, k_, v) for v in (a, b))
    result = float(collocation.integrate(a, b, k_, ell_, w_a, w_b, 0, spline, bessel_type, 8))
    assert result == pytest.approx(expected, rel=1e-10)
