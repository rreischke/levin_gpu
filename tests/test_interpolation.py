import numpy as np
import pytest
from scipy.interpolate import Akima1DInterpolator

from levin import interpolation


def evaluate_all(spline, x, n_integrand):
    return np.asarray(interpolation.evaluate(spline, np.asarray(x)[:, None], np.arange(n_integrand)[None, :]))


def test_matches_scipy_akima_on_generic_data():
    # GSL and scipy only differ where both Akima weights vanish, which random data avoids.
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 40))
    y = rng.normal(size=(40, 3))
    spline = interpolation.build(x, y, log_x=False, log_y=False)
    xq = np.linspace(x[0], x[-1], 2000)
    expected = Akima1DInterpolator(x, y, axis=0)(xq)
    np.testing.assert_allclose(evaluate_all(spline, xq, 3), expected, rtol=1e-12, atol=1e-12)


def test_log_interpolation_is_exact_for_power_laws():
    x = np.geomspace(1e-3, 1e3, 30)
    spline = interpolation.build(x, (x ** 2.5)[:, None], log_x=True, log_y=True)
    xq = np.geomspace(1e-3, 1e3, 500)
    np.testing.assert_allclose(evaluate_all(spline, xq, 1)[:, 0], xq ** 2.5, rtol=1e-12)


def test_log_y_falls_back_to_linear_for_non_positive_integrands():
    x = np.linspace(1, 2, 10)
    spline = interpolation.build(x, np.stack([x, x - 1.5], axis=1), log_x=False, log_y=True)
    np.testing.assert_array_equal(np.asarray(spline.log_y), [True, False])
    # The second integrand is linear, so the linear-space spline reproduces it exactly.
    assert evaluate_all(spline, [1.25], 2)[0, 1] == pytest.approx(-0.25, rel=1e-12)


def test_outside_support_is_zero():
    x = np.geomspace(1e-2, 1e2, 20)
    spline = interpolation.build(x, (x ** 2)[:, None], log_x=True, log_y=True)
    np.testing.assert_array_equal(evaluate_all(spline, [0.0, 1e-3, 1e3], 1), 0.0)
    np.testing.assert_allclose(evaluate_all(spline, [1e-2, 1e2], 1)[:, 0], [1e-4, 1e4], rtol=1e-12)


def test_input_validation():
    with pytest.raises(ValueError, match="dimensions"):
        interpolation.build(np.arange(10.0), np.ones((9, 1)), False, False)
    with pytest.raises(ValueError, match="at least"):
        interpolation.build(np.arange(4.0), np.ones((4, 1)), False, False)
    with pytest.raises(ValueError, match="increasing"):
        interpolation.build(np.array([0, 1, 1, 2, 3.0]), np.ones((5, 1)), False, False)


@pytest.mark.reference
def test_matches_gsl_through_pylevin():
    pylevin = pytest.importorskip("pylevin")
    x = np.geomspace(1e-5, 100, 60)
    f = np.stack([x ** 3 + x ** 2 + x, np.sin(x) + 0.1 * x, np.exp(-x) * x ** 2], axis=1)
    xq = np.geomspace(1e-5, 100, 1000)  # GSL aborts outside the support
    for log_x in (True, False):
        for log_y in (True, False):
            expected = np.array(pylevin.pylevin(0, x, f, log_x, log_y, 1).get_integrand(xq))
            spline = interpolation.build(x, f, log_x, log_y)
            np.testing.assert_allclose(evaluate_all(spline, xq, 3), expected, rtol=1e-11, atol=1e-14)
