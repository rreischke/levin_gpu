"""Chebyshev collocation nodes and basis functions on the reference interval [-1, 1].

Conventions follow ``pylevin::setNodes_cheby`` and ``fill_chebyshev_basis``:
the collocation nodes are Chebyshev-Gauss nodes rescaled so that the first
and last node coincide with the interval endpoints -1 and +1.
"""

import jax.numpy as jnp


def nodes(n_col):
    """Return ``n_col`` endpoint-including Chebyshev nodes on [-1, 1], ascending."""
    j = jnp.arange(n_col)
    return -jnp.cos((2 * j + 1) * jnp.pi / (2 * n_col)) / jnp.cos(jnp.pi / (2 * n_col))


def basis(y, n_col):
    """Evaluate T_m(y) and T_m'(y) for m = 0, ..., n_col - 1.

    Parameters
    ----------
    y : array_like
        Points in [-1, 1], arbitrary shape ``S``.
    n_col : int
        Number of basis functions (static).

    Returns
    -------
    values, derivatives : jax.Array
        Arrays of shape ``S + (n_col,)``.
    """
    y = jnp.asarray(y)
    values = [jnp.ones_like(y), y]
    derivatives = [jnp.zeros_like(y), jnp.ones_like(y)]
    for m in range(2, n_col):
        values.append(2 * y * values[m - 1] - values[m - 2])
        derivatives.append(2 * values[m - 1] + 2 * y * derivatives[m - 1] - derivatives[m - 2])
    return jnp.stack(values[:n_col], axis=-1), jnp.stack(derivatives[:n_col], axis=-1)


def map_to_interval(y, a, b):
    """Map y in [-1, 1] to x in [a, b], hitting both endpoints exactly (as ``pylevin::map_y_to_x``)."""
    return -(a * (y - 1) - b * (y + 1)) / 2
