"""Interpolation of tabulated integrands f(x) on the device.

Reproduces the GSL Akima spline (``gsl_interp_akima``, natural boundary
slopes) that ``pylevin`` uses, so the two codes integrate the same f(x). The
integrands are stored as a batch of shape ``(n_x, n_integrand)``. They can be
interpolated in log x and/or log f, as with the ``logx``/``logy`` flags in
``pylevin``. As in ``pylevin``, log f is only used for an integrand that is
strictly positive everywhere.

Evaluation uses ``jnp.searchsorted`` plus gathers, so it is jit- and
vmap-compatible. Points outside the tabulated range evaluate to zero,
matching the GSL error branch in ``pylevin::inhomogeneity``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_MIN_POINTS = 5  # gsl_interp_akima.min_size


class Spline(NamedTuple):
    """Piecewise cubics y + b t + c t^2 + d t^3 with t = x - knots[i], one column per integrand."""

    knots: jax.Array  # (n_x,), in log x if log_x
    y: jax.Array      # (n_x, n_integrand), in log f where log_y
    b: jax.Array      # (n_x - 1, n_integrand)
    c: jax.Array      # (n_x - 1, n_integrand)
    d: jax.Array      # (n_x - 1, n_integrand)
    log_x: jax.Array  # () bool
    log_y: jax.Array  # (n_integrand,) bool


def akima_coefficients(x, y):
    """Akima coefficients (b, c, d) as in GSL's ``akima_calc``.

    ``x`` has shape ``(n_x,)`` and ``y`` has shape ``(n_x, n_integrand)``.
    """
    h = jnp.diff(x)[:, None]
    m = jnp.diff(y, axis=0) / h
    m = jnp.concatenate([3 * m[:1] - 2 * m[1:2], 2 * m[:1] - m[1:2], m,
                         2 * m[-1:] - m[-2:-1], 3 * m[-1:] - 2 * m[-2:-1]], axis=0)
    n_segments = h.shape[0]
    m_minus_2, m_minus_1, m_0, m_plus_1, m_plus_2 = (m[o:o + n_segments] for o in range(5))

    ne = jnp.abs(m_plus_1 - m_0) + jnp.abs(m_minus_1 - m_minus_2)
    ne_next = jnp.abs(m_plus_2 - m_plus_1) + jnp.abs(m_0 - m_minus_1)
    alpha = jnp.abs(m_minus_1 - m_minus_2) / jnp.where(ne == 0, 1.0, ne)
    alpha_next = jnp.abs(m_0 - m_minus_1) / jnp.where(ne_next == 0, 1.0, ne_next)
    t_next = jnp.where(ne_next == 0, m_0, (1 - alpha_next) * m_0 + alpha_next * m_plus_1)
    b = (1 - alpha) * m_minus_1 + alpha * m_0
    c = (3 * m_0 - 2 * b - t_next) / h
    d = (b + t_next - 2 * m_0) / (h * h)
    flat = ne == 0
    return jnp.where(flat, m_0, b), jnp.where(flat, 0.0, c), jnp.where(flat, 0.0, d)


def build(x, integrand, log_x, log_y):
    """Build the splines for all integrands; ``integrand`` has shape ``(n_x, n_integrand)``."""
    x = np.asarray(x, dtype=float)
    integrand = np.asarray(integrand, dtype=float)
    if x.ndim != 1 or integrand.ndim != 2 or integrand.shape[0] != x.shape[0]:
        raise ValueError("support dimensions must match integrand dimensions: "
                         f"x {x.shape}, integrand {integrand.shape}")
    if x.shape[0] < _MIN_POINTS:
        raise ValueError(f"the Akima spline needs at least {_MIN_POINTS} support points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    log_y = np.full(integrand.shape[1], bool(log_y)) & np.all(integrand > 0, axis=0)
    knots = np.log(x) if log_x else x
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(log_y, np.log(integrand), integrand)
    knots, y = jnp.asarray(knots), jnp.asarray(y)
    b, c, d = akima_coefficients(knots, y)
    return Spline(knots, y, b, c, d, jnp.asarray(bool(log_x)), jnp.asarray(log_y))


def evaluate(spline, x, owner):
    """Evaluate integrand ``owner`` at ``x``; both broadcast to a common shape."""
    x, owner = jnp.broadcast_arrays(jnp.asarray(x, dtype=float), jnp.asarray(owner))
    t = jnp.where(spline.log_x, jnp.log(jnp.where(x > 0, x, 1.0)), x)
    n_x = spline.knots.shape[0]
    segment = jnp.clip(jnp.searchsorted(spline.knots, t, side="right") - 1, 0, n_x - 2)
    delta = t - spline.knots[segment]
    value = spline.y[segment, owner] + delta * (
        spline.b[segment, owner] + delta * (spline.c[segment, owner] + delta * spline.d[segment, owner]))
    value = jnp.where(spline.log_y[owner], jnp.exp(value), value)
    inside = (t >= spline.knots[0]) & (t <= spline.knots[-1]) & ((x > 0) | ~spline.log_x)
    return jnp.where(inside, value, 0.0)
