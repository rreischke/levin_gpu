import numpy as np
import pytest
import scipy.special as sp

import levin

X = np.geomspace(1e-5, 20, 1000)
K = np.geomspace(1e-2, 15, 100)


def weber_case(bessel_type, ell, b_over_a):
    """Weber's second exponential integral for J_nu(a x) J_nu(b x) against a Gaussian.

    int_0^inf x e^(-x^2/2) J_nu(ax) J_nu(bx) dx = e^(-(a^2+b^2)/2) I_nu(ab), and for
    spherical functions (nu = ell + 1/2) with an extra x and a factor pi / (2 sqrt(ab)).
    """
    a, b = K, b_over_a * K
    scaled = np.exp(-(a - b) ** 2 / 2)  # e^(-(a^2+b^2)/2) I_nu(ab) = ive(nu, ab) e^(-(a-b)^2/2)
    if bessel_type == 1:
        return X * np.exp(-X ** 2 / 2), a, b, sp.ive(ell, a * b) * scaled
    return X ** 2 * np.exp(-X ** 2 / 2), a, b, np.pi / (2 * np.sqrt(a * b)) * sp.ive(ell + 0.5, a * b) * scaled


def integrate(integrator, k_1, k_2, ell_1, ell_2):
    ones = np.ones_like(k_1)
    return np.asarray(integrator.integrate(X[0] * ones, X[-1] * ones, k_1, k_2,
                                           np.full(k_1.shape, ell_1), np.full(k_1.shape, ell_2)))


@pytest.mark.parametrize("bessel_type, ell, b_over_a", [(1, 0, 0.5), (1, 3, 2.0), (0, 2, 0.5), (0, 5, 1.0)])
def test_closed_form(bessel_type, ell, b_over_a):
    f, a, b, exact = weber_case(bessel_type, ell, b_over_a)
    integrator = levin.LevinDouble(bessel_type, X, f)
    integrator.set_levin(n_col=8, max_subintervals=32, rel_acc=1e-6)
    result = integrate(integrator, a, b, ell, ell)[:, 0]
    assert np.max(np.abs(result - exact)) < 1e-5 * np.max(np.abs(exact))


@pytest.mark.parametrize("bessel_type", [0, 1])
def test_reduces_to_single_for_k2_zero(bessel_type):
    """With k_2 = 0 and ell_2 = 0 the second factor is J_0(0) = 1."""
    f = X ** 3 * np.exp(-X)
    k = np.geomspace(0.1, 100, 40)
    double = levin.LevinDouble(bessel_type, X, f)
    double.set_levin(rel_acc=1e-8, max_ratio=None)
    single = levin.LevinSingle(bessel_type, X, f)
    single.set_levin(rel_acc=1e-8, max_ratio=None)
    expected = np.asarray(single.integrate(X[0] * np.ones_like(k), X[-1] * np.ones_like(k), k, np.full(k.shape, 4)))
    result = integrate(double, k, np.zeros_like(k), 4, 0)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-12 * np.abs(expected).max())


def test_batched_integrands_and_repeat_call():
    powers = np.linspace(0.5, 2.0, 3)
    f = X[:, None] ** powers[None, :] * np.exp(-X[:, None])
    k = np.geomspace(0.1, 100, 30)
    batch = levin.LevinDouble(0, X, f)
    first = integrate(batch, k, 2 * k, 3, 7)
    np.testing.assert_allclose(integrate(batch, k, 2 * k, 3, 7), first, rtol=0, atol=1e-10 * np.abs(first).max())
    for i in range(len(powers)):
        single = integrate(levin.LevinDouble(0, X, f[:, i]), k, 2 * k, 3, 7)[:, 0]
        np.testing.assert_allclose(first[:, i], single, rtol=1e-12, atol=1e-14 * np.abs(single).max())


def test_input_validation():
    integrator = levin.LevinDouble(0, X, X)
    with pytest.raises(ValueError, match="matching sizes"):
        integrator.integrate([1.0], [2.0], [1.0], [1.0, 2.0], [0], [0])
    with pytest.raises(ValueError, match="ell_2 must be integer"):
        integrator.integrate([1.0], [2.0], [1.0], [1.0], [0], [0.5])
