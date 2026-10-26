"""End-to-end comparison with the C++ pylevin implementation (dense solver)."""

import numpy as np
import pytest

pylevin = pytest.importorskip("pylevin")

import levin  # noqa: E402

X = np.geomspace(1e-5, 100, 100)
POWERS = np.linspace(1, 2, 3)


# pylevin builds with the ratio cap and non-oscillatory quadrature (both on by default).
PYLEVIN_HAS_FIXES = "max_ratio" in pylevin.pylevin.set_levin.__doc__


def set_levin(reference, *args, fixes):
    """pylevin.set_levin with (fixes=True) or without the ratio cap and non-oscillatory quadrature."""
    if fixes or not PYLEVIN_HAS_FIXES:
        reference.set_levin(*args)
    else:
        reference.set_levin(*args, max_ratio=0.0, non_oscillatory=0.0)


def run_both(bessel_type, f, k, ell, n_col, rel_acc, x_min=X[0]):
    ell = np.full(k.shape, ell)
    x_min, x_max = x_min * np.ones_like(k), X[-1] * np.ones_like(k)
    reference = pylevin.pylevin(bessel_type, X, f, True, True, 1)
    set_levin(reference, n_col, 32, rel_acc, False, False, 0.0, False, fixes=False)
    expected = np.zeros((len(k), f.shape[1]))
    reference.levin_integrate_bessel_single(x_min, x_max, k, ell, expected)

    integrator = levin.LevinSingle(bessel_type, X, f, logx=True, logy=True)
    # Switch off the ratio cap and the quadrature on non-oscillatory pieces, which pylevin does not have.
    integrator.set_levin(n_col=n_col, max_subintervals=32, rel_acc=rel_acc, max_ratio=None, non_oscillatory=None)
    return np.asarray(integrator.integrate(x_min, x_max, k, ell)), expected


@pytest.mark.reference
@pytest.mark.parametrize("rel_acc", [1e-4, 1e-7])
@pytest.mark.parametrize("case", ["paper", "powers"])
def test_spherical_matches_pylevin(case, rel_acc):
    if case == "paper":
        f, k, ell = (X ** 3 + X ** 2 + X)[:, None], np.geomspace(1e-3, 1e4, 1000), 5
    else:
        f, k, ell = X[:, None] ** (3 * POWERS) + (X ** 2 + X)[:, None], np.geomspace(1e-3, 1e4, 500), 10
    result, expected = run_both(0, f, k, ell, 8, rel_acc)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


@pytest.mark.reference
def test_cylindrical_matches_pylevin():
    # Below k (b - a) ~ 1 the collocation matrix is nearly singular (condition
    # number ~1e18) in both codes, and their results differ at the noise level.
    # Above that, a few integrals take different bisection paths, so they agree
    # to the requested accuracy rather than to rounding.
    f = X[:, None] ** POWERS + (X ** 2 + X)[:, None]
    rel_acc = 1e-6
    result, expected = run_both(1, f, np.geomspace(0.1, 1e4, 500), 0, 8, rel_acc)
    np.testing.assert_allclose(result, expected, rtol=rel_acc)
    assert np.median(np.abs(result / expected - 1)) < 1e-9


@pytest.mark.reference
def test_high_order_matches_pylevin():
    # GSL's j_ell aborts on underflow, so start away from x = 0 at ell = 50.
    f = (X ** 2 * np.exp(-X ** 2 / 200) + 1e-30)[:, None]
    result, expected = run_both(0, f, np.geomspace(1e-2, 1e3, 500), 50, 8, 1e-4, x_min=0.1)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def run_both_double(bessel_type, f, k_1, k_2, ell_1, ell_2, rel_acc):
    ell_1, ell_2 = np.full(k_1.shape, ell_1), np.full(k_1.shape, ell_2)
    x_min, x_max = X[0] * np.ones_like(k_1), X[-1] * np.ones_like(k_1)
    reference = pylevin.pylevin(bessel_type + 2, X, f, True, True, 1)
    # super_accurate (Boost Bessel functions): with GSL's j_ell, pylevin's
    # two-Bessel results are off by up to 25% at k ~ 1e3 to 1e4.
    set_levin(reference, 8, 32, rel_acc, True, False, 0.0, False, fixes=False)
    expected = np.zeros((len(k_1), f.shape[1]))
    reference.levin_integrate_bessel_double(x_min, x_max, k_1, k_2, ell_1, ell_2, expected)

    integrator = levin.LevinDouble(bessel_type, X, f, logx=True, logy=True)
    integrator.set_levin(n_col=8, max_subintervals=32, rel_acc=rel_acc, max_ratio=None, non_oscillatory=None)
    return np.asarray(integrator.integrate(x_min, x_max, k_1, k_2, ell_1, ell_2)), expected


@pytest.mark.reference
@pytest.mark.parametrize("case", ["paper", "different k"])
def test_double_spherical_matches_pylevin(case):
    k = np.geomspace(1e-3, 1e4, 300)
    if case == "paper":
        f, k_2, ell_1, ell_2 = (X ** 3 + X ** 2 + X)[:, None], k, 10, 5
    else:
        f, k_2, ell_1, ell_2 = (X ** 2 + X)[:, None], 2 * k, 3, 7
    rel_acc = 1e-4
    result, expected = run_both_double(0, f, k, k_2, ell_1, ell_2, rel_acc)
    # I(k) oscillates through zero; compare relative to the local amplitude.
    amplitude = np.maximum.accumulate(np.abs(expected[::-1]), axis=0)[::-1]
    assert np.max(np.abs(result - expected) / amplitude) < rel_acc
    assert np.median(np.abs(result - expected) / amplitude) < 1e-9


@pytest.mark.reference
def test_double_cylindrical_matches_pylevin():
    k = np.geomspace(0.1, 1e4, 300)
    rel_acc = 1e-4
    result, expected = run_both_double(1, (X ** 1.5 + X)[:, None], k, 0.5 * k, 0, 1, rel_acc)
    amplitude = np.maximum.accumulate(np.abs(expected[::-1]), axis=0)[::-1]
    assert np.max(np.abs(result - expected) / amplitude) < rel_acc
    assert np.median(np.abs(result - expected) / amplitude) < 1e-9


@pytest.mark.reference
@pytest.mark.parametrize("bessel_type, ell", [(0, 5), (0, 20), (1, 0)])
def test_defaults_match_fixed_pylevin(bessel_type, ell):
    """pylevin with the ratio cap and non-oscillatory quadrature runs the same algorithm as the defaults here."""
    f = (X ** 3 + X ** 2 + X)[:, None] if ell < 20 else (X ** 2 * np.exp(-X / 30))[:, None]
    k = np.geomspace(1e-2, 1e4, 300)
    ell_array = np.full(k.shape, ell)
    x_min, x_max = X[0] * np.ones_like(k), X[-1] * np.ones_like(k)
    if not PYLEVIN_HAS_FIXES:
        pytest.skip("installed pylevin has no max_ratio / non_oscillatory options")
    reference = pylevin.pylevin(bessel_type, X, f, True, True, 1)
    set_levin(reference, 8, 32, 1e-4, True, False, 0.0, False, fixes=True)
    expected = np.zeros((len(k), 1))
    reference.levin_integrate_bessel_single(x_min, x_max, k, ell_array, expected)
    integrator = levin.LevinSingle(bessel_type, X, f)
    integrator.set_levin(n_col=8, max_subintervals=32, rel_acc=1e-4)
    result = np.asarray(integrator.integrate(x_min, x_max, k, ell_array))
    amplitude = np.maximum.accumulate(np.abs(expected[::-1]), axis=0)[::-1]
    assert np.median(np.abs(result - expected) / amplitude) < 1e-10
    assert np.max(np.abs(result - expected) / amplitude) < 1e-4
