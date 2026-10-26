"""Dense Levin collocation for products of Bessel functions, batched over intervals.

For one Bessel function, w = (J_ell(kx), J_{ell+1}(kx)) satisfies w' = A(x) w with

    A = [[ ell/x,  -k                ],
         [ k,      -(ell + c)/x      ]],   c = 2 (spherical), 1 (cylindrical).

For a product of N Bessel functions, w is the outer product of the N pairs,
with component index i = i_1 + 2 i_2 + 4 i_3 as in ``pylevin``, so w_0 is the
product itself. A is the Kronecker sum of the single-Bessel matrices; this
reproduces ``pylevin::A_matrix_double`` and ``A_matrix_triple``.

Levin's method looks for p with p' + A^T p = (f, 0, ..., 0). Then
I = p(b).w(b) - p(a).w(a). Writing each of the d = 2^N components of p as a
Chebyshev expansion with ``n_col`` modes and collocating at ``n_col`` nodes
gives a dense (d n_col) x (d n_col) system G c = F, as in
``pylevin::solve_LSE_single``/``_double``/``_triple``.

The integral is linear in the sampled integrand:

    I = L^T G^{-1} F = v . f(x_j),   with   G^T v = L,

where L collects the endpoint evaluations of the basis weighted by w(a),
w(b), and only the first n_col entries of F are nonzero
(F_j = (b - a)/2 f(x_j)). G and L depend on the interval, k and ell but not
on f, so each subinterval reduces to an ``n_col``-point quadrature rule with
weights v. Every subinterval belongs to one integral (see ``adaptive.py``),
so evaluating an integral on its stored subintervals is a weighted sum of f
at that subinterval's nodes.

**Non-oscillatory pieces.** When (sum_i k_i) (b - a) is small, w varies
slowly over the piece. Then Levin's system is nearly singular: the
homogeneous solutions of p' + A^T p = 0 are themselves almost polynomial,
and the condition number of G reaches 1e17 to 1e19 for k dx ~ 0.05. Results
are then only good to ~1e-4 (also in pylevin), and for ell = 0 at small
k (b - a) they can be off by O(1) in pylevin. On such pieces the integrand
f w_0 is smooth, so an interpolatory quadrature on the same nodes is
accurate and well-conditioned: v_j = (b - a)/2 q_j w_0(x_j), with q the
weights that integrate T_0 .. T_{n-1} exactly on [-1, 1]. This is still
linear in f, so it fits the cached-weights design, and the bisection's
n_col vs n_col/2 error estimate applies unchanged.

``k`` and ``ell`` carry a trailing axis with one entry per Bessel factor;
all other dimensions broadcast as batch dimensions. ``w_a`` and ``w_b`` are
passed in so that callers can share Bessel evaluations between neighbouring
subintervals.
"""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from levin import bessel, chebyshev, interpolation

# Default threshold: pieces with (sum_i k_i) (b - a) below it use interpolatory quadrature.
NON_OSCILLATORY = 0.5


@lru_cache(maxsize=None)
def interpolatory_weights(n_col):
    """Weights q with sum_j q_j T_m(y_j) = int_{-1}^{1} T_m(y) dy for m < n_col, at the Levin nodes."""
    m = np.arange(n_col)
    even = m % 2 == 0
    moments = np.where(even, 2.0 / np.where(even, 1.0 - m.astype(float) ** 2, 1.0), 0.0)
    # Same nodes as chebyshev.nodes, in NumPy so this also runs while tracing.
    y = -np.cos((2 * m + 1) * np.pi / (2 * n_col)) / np.cos(np.pi / (2 * n_col))
    values = np.cos(m[None, :] * np.arccos(np.clip(y[:, None], -1.0, 1.0)))  # T_m(y_j)
    return np.linalg.solve(values.T, moments)


def _kron(p, q):
    """Batched Kronecker product over the last two axes; q is the fast index."""
    shape = p.shape[:-2] + (p.shape[-2] * q.shape[-2], p.shape[-1] * q.shape[-1])
    return jnp.einsum("...ac,...be->...abce", p, q).reshape(shape)


def endpoint_w(bessel_type, ell, k, x):
    """Levin basis vector w(x), the outer product of (J_ell(k x), J_{ell+1}(k x)) over factors.

    ``ell`` and ``k`` have shape ``batch + (n_bessel,)``, ``x`` has shape
    ``batch``. Returns shape ``batch + (2^n_bessel,)``.
    """
    w = None
    for factor in range(k.shape[-1]):
        pair = bessel.levin_w(bessel_type, ell[..., factor], k[..., factor] * x)
        w = pair if w is None else (pair[..., :, None] * w[..., None, :]).reshape(pair.shape[:-1] + (-1,))
    return w


def derivative_matrix(k, ell, x, bessel_type):
    """A(x) with w' = A w; shape ``batch + (2^n_bessel, 2^n_bessel)``."""
    offset = 2.0 if bessel_type == 0 else 1.0
    matrix = None
    for factor in range(k.shape[-1]):
        kf, nu = k[..., factor], jnp.asarray(ell[..., factor], dtype=float)
        single = jnp.stack([jnp.stack([nu / x, -kf * jnp.ones_like(x)], axis=-1),
                            jnp.stack([kf * jnp.ones_like(x), -(nu + offset) / x], axis=-1)], axis=-2)
        if matrix is None:
            matrix = single
        else:
            size = matrix.shape[-1]
            matrix = _kron(jnp.broadcast_to(jnp.eye(2), single.shape), matrix) + \
                _kron(single, jnp.broadcast_to(jnp.eye(size), matrix.shape))
    return matrix


def collocation_matrix(a, b, k, ell, bessel_type, n_col):
    """Assemble G and the collocation nodes for a batch of intervals.

    Rows are ordered (component i, node j) and columns (component q, mode m),
    as in ``pylevin``:
    G[i n + j, q n + m] = (b - a)/2 A_qi(x_j) T_m(y_j) + delta_iq T'_m(y_j).

    Returns ``G`` with shape ``batch + (d n_col, d n_col)`` and ``x`` with
    shape ``batch + (n_col,)``.
    """
    a, b, k = (jnp.asarray(v, dtype=float) for v in (a, b, k))
    ell = jnp.asarray(ell)
    y = chebyshev.nodes(n_col)
    values, derivatives = chebyshev.basis(y, n_col)  # (node j, mode m)
    x = chebyshev.map_to_interval(y, a[..., None], b[..., None])
    matrix = derivative_matrix(k[..., None, :], ell[..., None, :], x, bessel_type)  # (..., j, q, i)
    d = matrix.shape[-1]
    half = 0.5 * (b - a)
    system = half[..., None, None, None, None] * jnp.einsum("...jqi,jm->...ijqm", matrix, values)
    system = system + jnp.eye(d)[:, None, :, None] * derivatives[None, :, None, :]
    return system.reshape(system.shape[:-4] + (d * n_col, d * n_col)), x


def endpoint_functional(w_a, w_b, n_col):
    """L with I = L . c, from p_i(b) = sum_m c_im and p_i(a) = sum_m (-1)^m c_im."""
    alternating = (-1.0) ** jnp.arange(n_col)
    functional = w_b[..., :, None] - alternating * w_a[..., :, None]
    return functional.reshape(functional.shape[:-2] + (-1,))


@partial(jax.jit, static_argnums=(6, 7))
def quadrature_weights(a, b, k, ell, w_a, w_b, bessel_type, n_col, non_oscillatory=NON_OSCILLATORY):
    """Solve G^T u = L (batched LU) and return the nodes and weights for f(x_j).

    Returns ``x`` and ``v``, both of shape ``batch + (n_col,)``, with
    I = sum_j v_j f(x_j). The (b - a)/2 Jacobian is folded into ``v``. As in
    ``pylevin``, an interval with b <= a has zero weights. Non-oscillatory
    pieces, (sum_i k_i)(b - a) < ``non_oscillatory``, use interpolatory
    quadrature instead (see the module docstring); 0 disables this.
    """
    a, b = jnp.asarray(a, dtype=float), jnp.asarray(b, dtype=float)
    k = jnp.asarray(k, dtype=float)
    valid = b > a
    # Replace empty intervals by [a, a + 1] so the solve stays regular.
    b_safe = jnp.where(valid, b, a + 1.0)
    matrix, x = collocation_matrix(a, b_safe, k, ell, bessel_type, n_col)
    functional = endpoint_functional(w_a, w_b, n_col)
    u = jnp.linalg.solve(jnp.swapaxes(matrix, -1, -2), functional[..., None])[..., 0]
    half = (0.5 * (b - a))[..., None]
    levin_weights = half * u[..., :n_col]
    smooth = (jnp.sum(k, axis=-1) * (b - a) < non_oscillatory)[..., None]

    def quadrature():
        # Bessel functions at the nodes are only needed on smooth pieces. Elsewhere evaluate
        # at x = 0 (power-series branch), so those pieces do not lengthen the recurrence loops.
        x_smooth = jnp.where(smooth, x, 0.0)
        w_nodes = endpoint_w(bessel_type, jnp.asarray(ell)[..., None, :], k[..., None, :], x_smooth)[..., 0]
        return half * jnp.asarray(interpolatory_weights(n_col)) * w_nodes

    # Evaluating Bessel functions at every node costs more than the Levin solve; skip it
    # for batches without smooth pieces (most bisection rounds after the first few).
    quadrature = jax.lax.cond(jnp.any(smooth), quadrature, lambda: jnp.zeros_like(levin_weights))
    return x, jnp.where(valid[..., None], jnp.where(smooth, quadrature, levin_weights), 0.0)


def integrate(a, b, k, ell, w_a, w_b, owner, spline, bessel_type, n_col, non_oscillatory=NON_OSCILLATORY):
    """Levin estimate on each interval for integrand ``owner``; shape ``batch``."""
    x, v = quadrature_weights(a, b, k, ell, w_a, w_b, bessel_type, n_col, non_oscillatory)
    f = interpolation.evaluate(spline, x, jnp.asarray(owner)[..., None])
    return jnp.sum(v * f, axis=-1)
