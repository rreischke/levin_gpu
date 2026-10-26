import numpy as np
import pytest

import levin

X = np.geomspace(1e-5, 20, 1000)
K = np.geomspace(1e-2, 15, 200)


def gaussian_case(bessel_type, ell):
    """int_0^inf x^(nu+1) e^(-x^2/2) J_nu(kx) dx = k^nu e^(-k^2/2) (nu = ell + 1/2 for spherical)."""
    if bessel_type == 0:
        return X ** (ell + 2) * np.exp(-X ** 2 / 2), np.sqrt(np.pi / 2) * K ** ell * np.exp(-K ** 2 / 2)
    return X ** (ell + 1) * np.exp(-X ** 2 / 2), K ** ell * np.exp(-K ** 2 / 2)


def integrate(integrator, k=K, ell=5, x_min=X[0], x_max=X[-1]):
    ones = np.ones_like(k)
    return np.asarray(integrator.integrate(x_min * ones, x_max * ones, k, np.full(k.shape, ell)))


@pytest.mark.parametrize("bessel_type, ell", [(0, 5), (0, 1), (1, 4), (0, 0), (1, 0)])
def test_closed_form(bessel_type, ell):
    f, exact = gaussian_case(bessel_type, ell)
    integrator = levin.LevinSingle(bessel_type, X, f, logx=True, logy=True)
    # 1e-6 is reachable with this table (its spline limits accuracy to ~3e-7);
    # unreachable tolerances exhaust the slot budget and leave path-dependent results.
    integrator.set_levin(n_col=8, max_subintervals=32, rel_acc=1e-6)
    result = integrate(integrator, ell=ell)[:, 0]
    # All k, including ell = 0 at small k, where Levin's own system is nearly singular
    # (pylevin is off by O(1) there); non-oscillatory pieces use interpolatory quadrature.
    assert np.max(np.abs(result - exact)) < 2e-6 * np.max(np.abs(exact))
    assert integrator.converged.shape == (len(K), 1)


def test_repeat_call_reuses_subintervals():
    f, _ = gaussian_case(0, 5)
    integrator = levin.LevinSingle(0, X, f)
    first = integrate(integrator)
    np.testing.assert_allclose(integrate(integrator), first, rtol=0, atol=1e-12 * np.abs(first).max())


def test_update_integrand():
    f, _ = gaussian_case(0, 5)
    # A narrow bump the subintervals found for f know nothing about.
    g = f * (1 + 3 * np.exp(-((X - 3) / 0.1) ** 2))
    reference = levin.LevinSingle(0, X, g)
    reference.set_levin(n_col=32, max_subintervals=200, rel_acc=1e-12, max_ratio=1.5)
    truth = integrate(reference)
    peak = np.abs(truth).max()

    # rel_acc must be reachable within the slot budget, otherwise there is nothing left to refine.
    integrator = levin.LevinSingle(0, X, f)
    integrator.set_levin(rel_acc=1e-4)
    integrate(integrator)
    integrator.update_integrand(X, g)
    reused = integrate(integrator)
    integrator.update_integrand(X, g, recheck=True)
    rechecked = integrate(integrator)

    error = lambda result: np.abs(result - truth) / peak
    # Reusing subintervals found for f misses the bump; refining from them recovers it.
    assert error(reused).max() > 1e-3
    assert error(rechecked).max() < 1e-3
    assert np.median(error(rechecked)) < 0.2 * np.median(error(reused))
    # After a recheck the refined subintervals are cached for repeat calls. The tiny
    # pieces around the bump are non-oscillatory (k dx ~ 0.05); with Levin there the
    # matrix is nearly singular and two compiled code paths differed by ~1e-4.
    # Pieces with k dx < 0.5 now use interpolatory quadrature; the remaining Levin
    # pieces (cond ~1e12) agree to ~1e-9.
    np.testing.assert_allclose(integrate(integrator), rechecked, rtol=0, atol=1e-7 * peak)


def test_batched_integrands_match_single_runs():
    powers = np.linspace(0.5, 3.0, 5)
    f = X[:, None] ** powers[None, :] * np.exp(-X[:, None])
    k = np.geomspace(0.1, 100, 50)
    batch = integrate(levin.LevinSingle(0, X, f), k=k, ell=3)
    for i in range(len(powers)):
        single = integrate(levin.LevinSingle(0, X, f[:, i]), k=k, ell=3)[:, 0]
        np.testing.assert_allclose(batch[:, i], single, rtol=1e-12)


def test_varying_ell_and_limits():
    f, _ = gaussian_case(1, 0)
    integrator = levin.LevinSingle(1, X, f)
    k = np.array([1.0, 2.0, 3.0])
    out = np.asarray(integrator.integrate([1e-5, 1.0, 2.0], [20.0, 10.0, 2.0], k, [0, 3, 7]))
    reference = [np.asarray(levin.LevinSingle(1, X, f).integrate(a, b, kk, l))[0, 0]
                 for a, b, kk, l in zip([1e-5, 1.0], [20.0, 10.0], k[:2], [0, 3])]
    np.testing.assert_allclose(out[:2, 0], reference, rtol=1e-12)
    assert out[2, 0] == 0.0  # empty interval


def test_input_validation():
    with pytest.raises(ValueError, match="bessel_type"):
        levin.LevinSingle(2, X, X)
    integrator = levin.LevinSingle(0, X, X)
    with pytest.raises(ValueError, match="matching sizes"):
        integrator.integrate([1.0], [2.0, 3.0], [1.0], [0])
    with pytest.raises(ValueError, match="integer"):
        integrator.integrate([1.0], [2.0], [1.0], [0.5])
    with pytest.raises(ValueError, match="same number of integrands"):
        integrator.update_integrand(X, np.ones((len(X), 2)))


def test_ratio_cap_prevents_false_convergence():
    """On wide intervals, I_{n_col} and I_{n_col/2} can agree while both are off by ~1e-2."""
    x = np.geomspace(1e-5, 100, 100)
    f = (x ** 3 + x ** 2 + x)[:, None]
    k = np.geomspace(0.5, 5, 60)
    args = (x[0] * np.ones_like(k), x[-1] * np.ones_like(k), k, np.full(k.shape, 5))
    reference = levin.LevinSingle(0, x, f)
    reference.set_levin(n_col=32, max_subintervals=200, rel_acc=1e-12, max_ratio=1.5)
    truth = np.asarray(reference.integrate(*args))[:, 0]
    scale = np.abs(truth).max()

    errors = {}
    for max_ratio in (5.0, None):
        integrator = levin.LevinSingle(0, x, f)
        integrator.set_levin(n_col=8, max_subintervals=32, rel_acc=1e-4, max_ratio=max_ratio)
        result = np.asarray(integrator.integrate(*args))[:, 0]
        assert integrator.converged.all()
        errors[max_ratio] = np.abs(result - truth).max() / scale
    assert errors[5.0] < 1e-4
    assert errors[None] > 1e-3  # pylevin's behaviour, which the cap fixes
