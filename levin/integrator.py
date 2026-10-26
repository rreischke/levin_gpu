"""User-facing integrators, mirroring the ``pylevin`` API."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from levin import adaptive, collocation, interpolation


_WEIGHT_CHUNK = 2 ** 16


@partial(jax.jit, static_argnums=(4, 5))
def _weights_chunk(left, right, k, ell, bessel_type, n_col, non_oscillatory):
    w_left = collocation.endpoint_w(bessel_type, ell, k, left)
    w_right = collocation.endpoint_w(bessel_type, ell, k, right)
    return collocation.quadrature_weights(left, right, k, ell, w_left, w_right, bessel_type, n_col, non_oscillatory)


@partial(jax.jit, static_argnames="n_integrals")
def _apply_weights(x, v, interval_owner, interval_integral, spline, n_integrals):
    f = interpolation.evaluate(spline, x, interval_owner[:, None])
    return jax.ops.segment_sum(jnp.sum(v * f, axis=-1), interval_integral, num_segments=n_integrals)


def _as_orders(ell, name):
    ell = np.atleast_1d(np.asarray(ell))
    if not np.issubdtype(ell.dtype, np.integer):
        if np.any(ell != np.round(ell)):
            raise ValueError(f"{name} must be integer")
        ell = ell.astype(np.int64)
    if np.any(ell < 0):
        raise ValueError(f"{name} must be non-negative")
    return ell


class _Levin:
    """Shared machinery for integrals over a product of ``n_bessel`` Bessel functions.

    The first call to ``integrate`` runs the adaptive bisection, which finds
    separate subintervals for every (k, ell, integrand) combination, and
    caches them with their quadrature weights. Later calls with the same
    arguments reuse them, so each integral costs one weighted sum of f at the
    stored nodes. This is the analogue of the precomputed LU factorisations
    in ``pylevin``. Calling with different arguments runs the bisection again.

    After :meth:`update_integrand`, ``recheck=False`` reuses the stored
    subintervals unchanged, as ``pylevin`` does. ``recheck=True`` runs the
    bisection again, starting from the stored subintervals, so integrals only
    get refined where the new integrand needs it.

    Unlike ``pylevin``, results are returned instead of written in place,
    because JAX arrays are immutable.
    """

    n_bessel = None

    def __init__(self, bessel_type, x, integrand, logx=True, logy=True):
        if bessel_type not in (0, 1):
            raise ValueError("bessel_type must be 0 (spherical) or 1 (cylindrical)")
        self.bessel_type = bessel_type
        integrand = np.asarray(integrand, dtype=float)
        if integrand.ndim == 1:
            integrand = integrand[:, None]
        self.n_integrand = integrand.shape[1]
        self._spline = interpolation.build(x, integrand, logx, logy)
        self._recheck_pending = False
        self.set_levin()

    def set_levin(self, n_col=8, max_subintervals=32, rel_acc=1e-4, abs_acc=0.0, max_ratio=5.0,
                  non_oscillatory=collocation.NON_OSCILLATORY):
        """Set the collocation and convergence parameters. Clears cached subintervals.

        ``n_col`` is rounded up to an even number, because the error estimate
        compares n_col with n_col/2 collocation points.

        ``max_ratio`` caps b / max(a, x_t) of every subinterval, where
        x_t = (ell + 1) / k is roughly where the Bessel function starts to
        oscillate (for products: the earliest factor). Wider subintervals are
        always split. Without the cap, the error estimate can report
        convergence on intervals spanning many decades while the result is
        off by up to ~1e-2 (see ``adaptive.py``). The pieces count towards
        the ``max_subintervals`` budget.

        ``non_oscillatory``: pieces with (sum_i k_i)(b - a) below this use
        interpolatory quadrature of f w_0 on the same nodes instead of Levin.
        There, Levin's collocation matrix is nearly singular (condition number
        up to 1e19) and only good to ~1e-4, or O(1) for ell = 0 at small k
        (see ``collocation.py``).

        ``max_ratio=None, non_oscillatory=None`` reproduces ``pylevin`` exactly.
        """
        if max_ratio is not None and not max_ratio > 1:
            raise ValueError("max_ratio must be larger than 1 (or None)")
        n_col = 2 * ((int(n_col) + 1) // 2)
        if n_col < 4:
            raise ValueError("n_col must be at least 4")
        self.n_col = n_col
        self.max_subintervals = int(max_subintervals)
        self.rel_acc = float(rel_acc)
        self.abs_acc = float(abs_acc)
        self.max_ratio = None if max_ratio is None else float(max_ratio)
        self.non_oscillatory = None if non_oscillatory is None else float(non_oscillatory)
        self._cache = None
        self.converged = None

    def update_integrand(self, x, integrand, logx=True, logy=True, recheck=False):
        """Replace the integrands (same number of integrands as before).

        Keeps the cached subintervals and weights. With ``recheck=True``, the
        next ``integrate`` call refines them for the new integrand.
        """
        integrand = np.asarray(integrand, dtype=float)
        if integrand.ndim == 1:
            integrand = integrand[:, None]
        if integrand.shape[1] != self.n_integrand:
            raise ValueError("If you update the integrand, it must have the same number of "
                             f"integrands as in the constructor ({self.n_integrand})")
        self._spline = interpolation.build(x, integrand, logx, logy)
        self._recheck_pending = self._recheck_pending or recheck

    def get_integrand(self, x):
        """Evaluate the interpolated integrands at ``x``; shape (len(x), n_integrand)."""
        x = jnp.asarray(x, dtype=float)
        return interpolation.evaluate(self._spline, x[:, None], jnp.arange(self.n_integrand)[None, :])

    def _integrate(self, x_min, x_max, ks, ells):
        """Integrals for per-factor sequences ``ks`` and ``ells``; shape (len(x_min), n_integrand)."""
        x_min, x_max = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x_min, x_max))
        ks = [np.atleast_1d(np.asarray(v, dtype=float)) for v in ks]
        ells = [_as_orders(v, name) for v, name in zip(ells, self._order_names)]
        if x_min.ndim != 1 or any(v.shape != x_min.shape for v in [x_max, *ks, *ells]):
            raise ValueError(f"{', '.join(self._argument_names)} must be one-dimensional with matching sizes")
        k, ell = np.stack(ks, axis=-1), np.stack(ells, axis=-1)

        key = tuple(v.tobytes() for v in (x_min, x_max, k, ell))
        n_k = x_min.shape[0]
        if self._cache is not None and self._cache["key"] == key:
            if not self._recheck_pending:
                result = _apply_weights(*self._cache["weights"], self._spline, n_k * self.n_integrand)
                return result.reshape(n_k, self.n_integrand)
            state = adaptive.recheck(self._cache["state"], *self._cache["flat"], self._spline,
                                     **self._bisection_options())
        else:
            flat = self._flatten(k, ell)
            state = adaptive.bisect(np.repeat(x_min, self.n_integrand), np.repeat(x_max, self.n_integrand),
                                    *flat, self._spline, **self._bisection_options())
            self._cache = {"key": key, "flat": flat}
        self._recheck_pending = False
        self._cache["state"] = state
        self._cache["weights"] = self._compute_weights(state, *self._cache["flat"])
        self.converged = np.asarray(state.done).reshape(n_k, self.n_integrand)
        return state.result.reshape(n_k, self.n_integrand)

    def _flatten(self, k, ell):
        """Per-integral arrays, ordered (argument index, integrand index)."""
        owner = np.tile(np.arange(self.n_integrand), k.shape[0])
        return (jnp.asarray(np.repeat(k, self.n_integrand, axis=0)),
                jnp.asarray(np.repeat(ell, self.n_integrand, axis=0)), jnp.asarray(owner))

    def _bisection_options(self):
        return dict(bessel_type=self.bessel_type, n_col=self.n_col, max_subintervals=self.max_subintervals,
                    tol_rel=self.rel_acc, tol_abs=self.abs_acc, max_ratio=self.max_ratio,
                    non_oscillatory=self.non_oscillatory)

    def _compute_weights(self, state, k, ell, owner):
        """Compact the used slots and solve once for their quadrature weights.

        The solves run in fixed-size chunks: one batch for all subintervals
        would hold a (d n_col)^2 matrix per subinterval at once, several GB
        for ~1e6 subintervals. The last chunk is padded, so one shape is compiled.
        """
        integral, slot = np.nonzero(np.asarray(state.used))
        left, right = np.asarray(state.left)[integral, slot], np.asarray(state.right)[integral, slot]
        k, ell = np.asarray(k)[integral], np.asarray(ell)[integral]
        n_intervals = integral.size
        chunk = min(_WEIGHT_CHUNK, max(n_intervals, 1))
        n_padded = -(-n_intervals // chunk) * chunk
        pad = lambda v: np.concatenate([v, np.repeat(v[-1:], n_padded - n_intervals, axis=0)])
        left, right, k, ell = (pad(v) for v in (left, right, k, ell))
        x, v = [], []
        for start in range(0, n_padded, chunk):
            part = slice(start, start + chunk)
            x_part, v_part = _weights_chunk(left[part], right[part], k[part], ell[part], self.bessel_type,
                                            self.n_col, 0.0 if self.non_oscillatory is None else self.non_oscillatory)
            x.append(x_part)
            v.append(v_part)
        x = jnp.concatenate(x)[:n_intervals]
        v = jnp.concatenate(v)[:n_intervals]
        return x, v, owner[integral], jnp.asarray(integral)


class LevinSingle(_Levin):
    """Levin integrator for int_a^b f(x) J_ell(k x) dx on GPU/CPU via JAX.

    Parameters
    ----------
    bessel_type : int
        0 for spherical, 1 for cylindrical Bessel functions (``pylevin`` types 0 and 1).
    x : array_like, shape (n_x,)
        Support of the tabulated integrands; strictly increasing, at least 5 points.
    integrand : array_like, shape (n_x, n_integrand) or (n_x,)
        Tabulated integrands f(x).
    logx, logy : bool
        Interpolate in log x and/or log f. As in ``pylevin``, log f is only
        used for integrands that are strictly positive.
    """

    n_bessel = 1
    _order_names = ("ell",)
    _argument_names = ("x_min", "x_max", "k", "ell")
    __doc__ += _Levin.__doc__.split("\n", 1)[1]

    def integrate(self, x_min, x_max, k, ell):
        """Return the integrals, an array of shape (len(k), n_integrand)."""
        return self._integrate(x_min, x_max, [k], [ell])


class LevinDouble(_Levin):
    """Levin integrator for int_a^b f(x) J_ell_1(k_1 x) J_ell_2(k_2 x) dx on GPU/CPU via JAX.

    Parameters
    ----------
    bessel_type : int
        0 for spherical, 1 for cylindrical Bessel functions (``pylevin`` types 2 and 3).
    x : array_like, shape (n_x,)
        Support of the tabulated integrands; strictly increasing, at least 5 points.
    integrand : array_like, shape (n_x, n_integrand) or (n_x,)
        Tabulated integrands f(x).
    logx, logy : bool
        Interpolate in log x and/or log f. As in ``pylevin``, log f is only
        used for integrands that are strictly positive.
    """

    n_bessel = 2
    _order_names = ("ell_1", "ell_2")
    _argument_names = ("x_min", "x_max", "k_1", "k_2", "ell_1", "ell_2")
    __doc__ += _Levin.__doc__.split("\n", 1)[1]

    def integrate(self, x_min, x_max, k_1, k_2, ell_1, ell_2):
        """Return the integrals, an array of shape (len(k_1), n_integrand)."""
        return self._integrate(x_min, x_max, [k_1, k_2], [ell_1, ell_2])
