"""Bessel functions of the first kind for integer order, vectorised in JAX.

JAX has no accurate general-order spherical or cylindrical Bessel function,
so they are implemented here. Every routine returns the pair
(J_ell, J_{ell+1}), which is exactly what the Levin basis vector w needs, and
works elementwise on broadcast arrays of integer ``ell >= 0`` and ``z >= 0``.

Each element is evaluated in one of three regimes:

* ``z < 1``: power series, summed with the prefactor in log space so that
  z^ell / (2 ell + 1)!! underflows to zero instead of producing NaN.
* ``z >= ell + 1`` (and ``z >= 25`` for cylindrical): forward recurrence
  from J_0 and J_1, which is stable for orders below z. The starting values
  are sin/cos closed forms (spherical) or the Hankel asymptotic expansion
  (cylindrical).
* Otherwise: Miller's backward recurrence, started above max(ell, z) and
  rescaled whenever it grows past 1e100. It is normalised by a least-squares
  fit to (J_0, J_1), which cannot both vanish. For cylindrical functions
  with z < 25, it uses the sum rule J_0 + 2 sum_k J_2k = 1 instead.

The recurrence lengths depend on ell and z, so the loops run up to the batch
maximum and are masked per element. They compile to a single fused loop
under ``jit``.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

_SERIES_Z_MAX = 1.0
_SERIES_TERMS = 14
_HANKEL_Z_MIN = 25.0
_HANKEL_TERMS = 24
_MILLER_ACCURACY = 160.0
_MILLER_EXTRA_ORDERS = 20
_RESCALE = 1e100


def _power_series(bessel_type, nu, z):
    """J_nu(z) from its power series; accurate for z < 1 and any order nu >= 0."""
    z_safe = jnp.maximum(z, jnp.finfo(z.dtype).tiny)
    if bessel_type == 0:
        log_prefactor = (nu * jnp.log(z_safe) - (nu + 1) * math.log(2.0)
                         - gammaln(nu + 1.5) + 0.5 * math.log(math.pi))
        x = -0.5 * z * z
    else:
        log_prefactor = nu * jnp.log(0.5 * z_safe) - gammaln(nu + 1)
        x = -0.25 * z * z
    term = jnp.ones_like(z)
    total = jnp.ones_like(z)
    for k in range(1, _SERIES_TERMS + 1):
        if bessel_type == 0:
            term = term * x / (k * (2 * nu + 2 * k + 1))
        else:
            term = term * x / (k * (nu + k))
        total = total + term
    value = jnp.exp(log_prefactor) * total
    return jnp.where(z == 0, (nu == 0).astype(z.dtype), value)


def _hankel_j0_j1(z):
    """Cylindrical J_0 and J_1 from the Hankel asymptotic expansion; for z >= 25."""
    sin_z, cos_z = jnp.sin(z), jnp.cos(z)
    prefactor = jnp.sqrt(2.0 / (jnp.pi * z)) / math.sqrt(2.0)
    inverse_z = 1.0 / z
    values = []
    for nu in (0, 1):
        mu = 4.0 * nu * nu
        term = jnp.ones_like(z)
        p = jnp.ones_like(z)
        q = jnp.zeros_like(z)
        for k in range(1, _HANKEL_TERMS + 1):
            term = term * (mu - (2 * k - 1) ** 2) / (8 * k) * inverse_z
            if k % 4 == 1:
                q = q + term
            elif k % 4 == 2:
                p = p - term
            elif k % 4 == 3:
                q = q - term
            else:
                p = p + term
        # cos and sin of z - (nu/2 + 1/4) pi, expanded to avoid cancellation in the phase.
        if nu == 0:
            cos_chi, sin_chi = cos_z + sin_z, sin_z - cos_z
        else:
            cos_chi, sin_chi = sin_z - cos_z, -sin_z - cos_z
        values.append(prefactor * (p * cos_chi - q * sin_chi))
    return values[0], values[1]


def _start_values(bessel_type, z):
    """J_0(z) and J_1(z) for z >= 1 (spherical) or z >= 25 (cylindrical)."""
    if bessel_type == 0:
        sin_z, cos_z = jnp.sin(z), jnp.cos(z)
        return sin_z / z, (sin_z / z - cos_z) / z
    return _hankel_j0_j1(z)


def _forward_recurrence(bessel_type, ell, z, f0, f1, active):
    """(J_ell, J_{ell+1}) by upward recurrence from J_0 = f0 and J_1 = f1."""
    c = 1.0 if bessel_type == 0 else 0.0
    out0 = jnp.where(ell == 0, f0, jnp.where(ell == 1, f1, 0.0))
    out1 = jnp.where(ell == 0, f1, 0.0)
    n_max = jnp.max(jnp.where(active, ell, 0))

    def body(n, state):
        f_previous, f_current, out0, out1 = state
        f_next = (2 * n + c) / z * f_current - f_previous
        out0 = jnp.where(n + 1 == ell, f_next, out0)
        out1 = jnp.where(n + 1 == ell + 1, f_next, out1)
        # Freeze elements whose orders are done so they cannot overflow.
        running = n <= ell
        return (jnp.where(running, f_current, f_previous),
                jnp.where(running, f_next, f_current), out0, out1)

    _, _, out0, out1 = lax.fori_loop(1, n_max + 1, body, (f0, f1, out0, out1))
    return out0, out1


def _miller_recurrence(bessel_type, ell, z, n_start):
    """Unnormalised backward recurrence from order ``n_start`` down to 0.

    Returns (f_ell, f_{ell+1}, f_0, f_1, sum of f_2k for k >= 1), all with the
    same unknown scale factor.
    """
    c = 1.0 if bessel_type == 0 else 0.0
    n_max = jnp.max(n_start)
    zeros = jnp.zeros_like(z)

    def body(i, state):
        f_above, f, out0, out1, even_sum = state
        n = n_max - i
        f = jnp.where(n == n_start, 1.0, f)
        f_below = (2 * n + c) / z * f - f_above
        out0 = jnp.where(n - 1 == ell, f_below, out0)
        out1 = jnp.where(n - 1 == ell + 1, f_below, out1)
        even_sum = even_sum + jnp.where(((n - 1) % 2 == 0) & (n - 1 >= 2), f_below, 0.0)
        scale = jnp.where(jnp.abs(f_below) > _RESCALE, 1.0 / _RESCALE, 1.0)
        return f * scale, f_below * scale, out0 * scale, out1 * scale, even_sum * scale

    f1, f0, out0, out1, even_sum = lax.fori_loop(
        0, n_max, body, (zeros, zeros, zeros, zeros, zeros))
    return out0, out1, f0, f1, even_sum


@partial(jax.jit, static_argnums=0)
def bessel_pair(bessel_type, ell, z):
    """Return (J_ell(z), J_{ell+1}(z)) elementwise.

    Parameters
    ----------
    bessel_type : int
        0 for spherical j_ell, 1 for cylindrical J_ell (static).
    ell : array_like of int
        Orders, ``ell >= 0``.
    z : array_like of float
        Arguments, ``z >= 0``. Broadcast against ``ell``.
    """
    z = jnp.asarray(z, dtype=float)
    ell = jnp.asarray(ell, dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
    ell, z = jnp.broadcast_arrays(ell, z)
    nu = ell.astype(z.dtype)
    z_forward_min = 1.0 if bessel_type == 0 else _HANKEL_Z_MIN

    series = z < _SERIES_Z_MAX
    forward = ~series & (z >= jnp.maximum(nu + 1, z_forward_min))
    miller = ~series & ~forward

    # Substitute harmless arguments where a regime does not apply, so every
    # branch stays finite before the final selection.
    z_series = jnp.where(series, z, 0.5)
    series_pair = (_power_series(bessel_type, nu, z_series),
                   _power_series(bessel_type, nu + 1, z_series))

    z_forward = jnp.where(forward, z, jnp.maximum(nu + 1, z_forward_min))
    f0, f1 = _start_values(bessel_type, z_forward)
    forward_pair = _forward_recurrence(bessel_type, ell, z_forward, f0, f1, forward)

    z_miller = jnp.where(miller, z, 1.0)
    top = jnp.maximum(nu + 1, z_miller)
    n_start = jnp.where(
        miller,
        jnp.ceil(top) + _MILLER_EXTRA_ORDERS + jnp.ceil(jnp.sqrt(_MILLER_ACCURACY * top)),
        0).astype(ell.dtype)
    out0, out1, g0, g1, even_sum = _miller_recurrence(bessel_type, ell, z_miller, n_start)
    j0, j1 = _start_values(bessel_type, jnp.maximum(z_miller, z_forward_min))
    g_max = jnp.maximum(jnp.abs(g0), jnp.abs(g1))
    g_max = jnp.where(g_max > 0, g_max, 1.0)
    h0, h1 = g0 / g_max, g1 / g_max
    scale = (j0 * h0 + j1 * h1) / (h0 * h0 + h1 * h1) / g_max
    if bessel_type == 1:
        norm = g0 + 2.0 * even_sum
        norm = jnp.where(norm != 0, norm, 1.0)
        scale = jnp.where(z_miller >= _HANKEL_Z_MIN, scale, 1.0 / norm)
    miller_pair = (out0 * scale, out1 * scale)

    return tuple(jnp.where(series, s, jnp.where(forward, f, m))
                 for s, f, m in zip(series_pair, forward_pair, miller_pair))


def spherical_jn(ell, z):
    """Spherical Bessel function j_ell(z) for integer ell >= 0, elementwise."""
    return bessel_pair(0, ell, z)[0]


def cylindrical_jn(ell, z):
    """Cylindrical Bessel function J_ell(z) for integer ell >= 0, elementwise."""
    return bessel_pair(1, ell, z)[0]


def levin_w(bessel_type, ell, z):
    """Return the Levin basis vector w = (J_ell(z), J_{ell+1}(z)), shape ``z.shape + (2,)``.

    ``bessel_type`` is 0 (spherical) or 1 (cylindrical).
    """
    return jnp.stack(bessel_pair(bessel_type, ell, z), axis=-1)
