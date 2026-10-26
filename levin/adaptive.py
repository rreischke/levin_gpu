"""Adaptive bisection for all integrals at once, using fixed-size buffers.

An "integral" is one (k, ell, integrand) combination. As in ``pylevin``,
every integral gets its own subintervals; nothing is shared between
integrands.

``pylevin::iterate_single`` refines one integral at a time. Here, all
integrals are refined together inside a single jitted ``lax.while_loop``.
Each integral has a fixed buffer of ``max_subintervals + 2`` slots (the most
``iterate_single`` can produce), so every per-slot array has shape
``(n_integrals, max_subintervals + 2)``.

One round, for each integral that is not yet done:

1. Test convergence with the ``pylevin`` criterion:
   ``|result - previous| <= tol`` and ``sqrt(sum(error)) <= tol``, where
   ``tol = max(tol_rel * |result|, tol_abs)``. Converged integrals are done.
2. Otherwise, if fewer than ``max_subintervals + 2`` slots are in use,
   split the subinterval with the largest error at its midpoint. Ties go to
   the leftmost subinterval, which matches ``pylevin``'s sorted order. The
   left half reuses the old slot and the right half goes into slot
   ``n_used``. Slots are not kept sorted, because only sums are needed.
3. Evaluate both halves at n_col and n_col/2. The error estimate of a
   subinterval is (I_{n_col} - I_{n_col/2})^2.

**Ratio cap (``max_ratio``, not in pylevin).** Let x_t = (ell + 1) / k,
roughly where the Bessel function starts to oscillate. A subinterval [a, b]
with b > ``max_ratio`` * max(a, x_t) gets an infinite error estimate, so it
is never treated as converged. It is split at x_t if a < x_t, otherwise at
its geometric midpoint. Integrals that need this start out split straight
into these pieces, so no rounds are spent on forced splits. The reason: the slowly varying Levin solution carries terms like
ell / (k x), which come from the ell / x entries of A. A Chebyshev polynomial
in linear x cannot resolve them on an interval spanning many decades. There,
I_{n_col} and I_{n_col/2} share the same resolution error, agree with each
other, and the error estimate reports convergence while the result is off by
up to ~1e-2. A linear midpoint split does not help, because the left child
keeps almost the same ratio. ``max_ratio=inf`` reproduces pylevin.

Most integrals converge after a few rounds, while a few may run all
``max_subintervals + 1`` rounds. Masking finished integrals in place would
waste most of the work (about 90% in typical runs). So the rounds run in a
jitted ``lax.while_loop`` on a batch that holds only the active integrals.
When fewer than a quarter of that batch is still active, a short host loop
gathers the remaining integrals into a smaller batch. Batch sizes are
16 * 4^j, so only a handful of shapes are ever compiled, and wasted work
stays below 4x. To bound memory, a batch holds at most 2^16 integrals;
larger problems are refined batch by batch.

:func:`recheck` re-evaluates stored subintervals for a new integrand and then
runs the same loop. Integrals whose subintervals still meet the tolerance
stop after the first check; the rest are refined further.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from levin import collocation


class BisectionState(NamedTuple):
    """Per-integral subinterval buffers; see the module docstring."""

    left: jax.Array      # (n_integrals, n_slots)
    right: jax.Array     # (n_integrals, n_slots)
    estimate: jax.Array  # (n_integrals, n_slots), Levin estimate with n_col nodes
    error: jax.Array     # (n_integrals, n_slots), (I_{n_col} - I_{n_col/2})^2
    used: jax.Array      # (n_integrals, n_slots), bool
    n_used: jax.Array    # (n_integrals,)
    result: jax.Array    # (n_integrals,)
    previous: jax.Array  # (n_integrals,)
    done: jax.Array      # (n_integrals,), bool: converged, or input interval empty


def _converged(state, tol_rel, tol_abs):
    tolerance = jnp.maximum(tol_rel * jnp.abs(state.result), tol_abs)
    total_error = jnp.sqrt(jnp.sum(jnp.where(state.used, state.error, 0.0), axis=-1))
    return (jnp.abs(state.result - state.previous) <= tolerance) & (total_error <= tolerance)


def _active(state, max_subintervals, tol_rel, tol_abs):
    finished = state.done | _converged(state, tol_rel, tol_abs)
    return ~finished & (state.n_used <= max_subintervals + 1), finished


# The cap applies above x_t = _TURNING * (ell + 1) / k, roughly where the
# Bessel function starts to oscillate (for products: the earliest factor).
# Below that, wide subintervals converge fine and need no extra pieces.
_TURNING = 1.0


def _cap_start(k, ell):
    """x_t per integral; ``k`` and ``ell`` have a trailing axis over Bessel factors."""
    k = jnp.asarray(k, dtype=float)
    x_turn = jnp.where(k > 0, _TURNING * (jnp.asarray(ell) + 1) / jnp.where(k > 0, k, 1.0), jnp.inf)
    return jnp.min(x_turn, axis=-1)


def _too_wide(left, right, k, ell, max_ratio):
    """b > max_ratio * max(a, x_t): too wide where the Bessel function oscillates."""
    floor = jnp.maximum(left, _cap_start(k, ell))
    return (floor > 0) & (right > max_ratio * floor)


def _estimates(left, right, w_left, w_right, k, ell, owner, spline, bessel_type, n_col, max_ratio, non_oscillatory):
    """Levin estimate at n_col and the error estimate (I_{n_col} - I_{n_col/2})^2.

    Subintervals wider than ``max_ratio`` get an infinite error estimate.
    """
    args = (left, right, k, ell, w_left, w_right, owner, spline, bessel_type)
    full = collocation.integrate(*args, n_col, non_oscillatory)
    half = collocation.integrate(*args, n_col // 2, non_oscillatory)
    return full, jnp.where(_too_wide(left, right, k, ell, max_ratio), jnp.inf, (full - half) ** 2)


def _refine_round(state, k, ell, owner, spline, bessel_type, n_col, max_subintervals, tol_rel, tol_abs,
                  max_ratio, non_oscillatory):
    active, finished = _active(state, max_subintervals, tol_rel, tol_abs)
    n_slots = state.left.shape[-1]
    slots = jnp.arange(n_slots)

    error = jnp.where(state.used, state.error, -jnp.inf)
    worst = error == jnp.max(error, axis=-1, keepdims=True)
    slot = jnp.argmin(jnp.where(worst & state.used, state.left, jnp.inf), axis=-1)

    def at(values):
        return jnp.take_along_axis(values, slot[:, None], axis=-1)[:, 0]

    left, right, old = at(state.left), at(state.right), at(state.estimate)
    # Too-wide subintervals split at x_t if they start below it, else geometrically.
    x_turn = _cap_start(k, ell)
    middle = jnp.where(_too_wide(left, right, k, ell, max_ratio),
                       jnp.where(left < x_turn, x_turn, jnp.sqrt(left * right)), 0.5 * (left + right))
    w_left, w_middle, w_right = (collocation.endpoint_w(bessel_type, ell, k, x) for x in (left, middle, right))
    pair = lambda a, b: jnp.stack([a, b], axis=1)
    full, err = _estimates(pair(left, middle), pair(middle, right), pair(w_left, w_middle),
                           pair(w_middle, w_right), k[:, None], ell[:, None], owner[:, None],
                           spline, bessel_type, n_col, max_ratio, non_oscillatory)

    at_slot = (slots == slot[:, None]) & active[:, None]
    at_new = (slots == state.n_used[:, None]) & active[:, None]

    def put(values, for_slot, for_new):
        return jnp.where(at_slot, for_slot[:, None], jnp.where(at_new, for_new[:, None], values))

    return BisectionState(
        left=put(state.left, left, middle),
        right=put(state.right, middle, right),
        estimate=put(state.estimate, full[:, 0], full[:, 1]),
        error=put(state.error, err[:, 0], err[:, 1]),
        used=state.used | at_new,
        n_used=state.n_used + active,
        result=jnp.where(active, state.result - old + full[:, 0] + full[:, 1], state.result),
        previous=jnp.where(active, state.result, state.previous),
        done=finished,
    )


_MIN_BATCH = 16
_BATCH_GROWTH = 4
# Largest batch refined at once (16 * 4^6). Each integral in a batch holds
# collocation matrices for two children at n_col and n_col/2, so this bounds
# the temporary memory (~1 GB at n_col = 8); more integrals run batch by batch.
_MAX_BATCH = 2 ** 16

_STATIC = ("bessel_type", "n_col", "max_subintervals")


@partial(jax.jit, static_argnames=("max_subintervals",))
def _active_mask(state, *, max_subintervals, tol_rel, tol_abs):
    return _active(state, max_subintervals, tol_rel, tol_abs)[0]


@partial(jax.jit, static_argnames=_STATIC)
def _refine_batch(state, rows, k, ell, owner, spline, stop_below, *, bessel_type, n_col,
                  max_subintervals, tol_rel, tol_abs, max_ratio, non_oscillatory):
    """Refine the integrals ``rows`` until at most ``stop_below`` remain active.

    Rows equal to ``n_integrals`` are padding: they gather zeros, are marked
    done, and are dropped when the batch is scattered back.
    """
    padding = rows >= state.result.shape[0]
    batch = jax.tree_util.tree_map(lambda v: v.at[rows].get(mode="fill", fill_value=0), state)
    batch = batch._replace(done=batch.done | padding)
    k, ell, owner = (v.at[rows].get(mode="clip") for v in (k, ell, owner))

    def keep_going(batch):
        return jnp.sum(_active(batch, max_subintervals, tol_rel, tol_abs)[0]) > stop_below

    def body(batch):
        return _refine_round(batch, k, ell, owner, spline, bessel_type, n_col,
                             max_subintervals, tol_rel, tol_abs, max_ratio, non_oscillatory)

    batch = lax.while_loop(keep_going, body, batch)
    return jax.tree_util.tree_map(lambda full, part: full.at[rows].set(part, mode="drop"), state, batch)


def _batch_size(n_active):
    size = _MIN_BATCH
    while size < n_active:
        size *= _BATCH_GROWTH
    return size


def _run(state, k, ell, owner, spline, bessel_type, n_col, max_subintervals, tol_rel, tol_abs, max_ratio,
         non_oscillatory):
    """Host loop: refine the active integrals in shrinking batches until none is left."""
    options = dict(bessel_type=bessel_type, n_col=n_col, max_subintervals=max_subintervals,
                   tol_rel=tol_rel, tol_abs=tol_abs, max_ratio=max_ratio, non_oscillatory=non_oscillatory)
    n_integrals = state.result.shape[0]
    while True:
        active = np.flatnonzero(np.asarray(_active_mask(state, max_subintervals=max_subintervals,
                                                        tol_rel=tol_rel, tol_abs=tol_abs)))
        if active.size == 0:
            break
        active = active[:_MAX_BATCH]
        size = _batch_size(active.size)
        rows = np.full(size, n_integrals)
        rows[:active.size] = active
        stop_below = size // _BATCH_GROWTH if size > _MIN_BATCH else 0
        state = _refine_batch(state, jnp.asarray(rows), k, ell, owner, spline, stop_below, **options)
    return state._replace(done=_finalise(state, tol_rel, tol_abs))


@jax.jit
def _finalise(state, tol_rel, tol_abs):
    return _converged(state, tol_rel, tol_abs) | state.done


def bisect(a, b, k, ell, owner, spline, *, bessel_type, n_col, max_subintervals, tol_rel, tol_abs,
           max_ratio=None, non_oscillatory=None):
    """Run adaptive bisection from [a, b] for every integral; returns the final state.

    ``a``, ``b``, ``k``, ``ell`` and ``owner`` (the integrand index) have shape
    ``(n_integrals,)``. As in ``pylevin``, an integral with b <= a is zero.
    ``max_ratio=None`` disables the ratio cap and ``non_oscillatory=None`` the
    interpolatory quadrature on non-oscillatory pieces.
    """
    max_ratio = np.inf if max_ratio is None else float(max_ratio)
    non_oscillatory = 0.0 if non_oscillatory is None else float(non_oscillatory)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lead, n_geometric = _initial_pieces(a, b, np.asarray(k, dtype=float), np.asarray(ell), max_ratio,
                                        max_subintervals + 2)
    state = _initial_state(a, b, k, ell, owner, spline, max_ratio, non_oscillatory, jnp.asarray(lead),
                           jnp.asarray(n_geometric),
                           bessel_type=bessel_type, n_col=n_col, max_subintervals=max_subintervals,
                           max_pieces=int((lead + n_geometric).max(initial=1)))
    return _run(state, k, ell, owner, spline, bessel_type, n_col, max_subintervals, tol_rel, tol_abs, max_ratio,
                non_oscillatory)


def _initial_pieces(a, b, k, ell, max_ratio, n_slots):
    """Host-side piece counts, mirroring :func:`_too_wide`.

    Returns ``lead`` (1 if [a, x_t] is a separate first piece) and the number
    of geometric pieces from max(a, x_t) to b with ratio below ``max_ratio``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        x_turn = np.where(k > 0, _TURNING * (ell + 1) / np.where(k > 0, k, 1.0), np.inf).min(axis=-1)
        start = np.maximum(a, x_turn)
        wide = np.isfinite(max_ratio) & (start > 0) & (b > max_ratio * start)
        lead = (wide & (a < x_turn)).astype(np.int32)
        log_base = np.log(max_ratio) if np.isfinite(max_ratio) else 1.0
        n_geometric = np.where(wide, np.floor(np.log(np.where(wide, b / start, 1.0)) / log_base) + 1, 1)
    n_geometric = np.minimum(n_geometric, n_slots - lead).astype(np.int32)
    return lead, n_geometric


def recheck(state, k, ell, owner, spline, *, bessel_type, n_col, max_subintervals, tol_rel, tol_abs,
            max_ratio=None, non_oscillatory=None):
    """Re-evaluate stored subintervals for a new integrand and refine only where needed.

    The first convergence check uses only the error estimate (``previous``
    is set to the new result). Refinement then continues as in :func:`bisect`,
    within the same slot budget.
    """
    max_ratio = np.inf if max_ratio is None else float(max_ratio)
    non_oscillatory = 0.0 if non_oscillatory is None else float(non_oscillatory)
    state = _reestimate(state, k, ell, owner, spline, max_ratio, non_oscillatory, bessel_type=bessel_type, n_col=n_col)
    return _run(state, k, ell, owner, spline, bessel_type, n_col, max_subintervals, tol_rel, tol_abs, max_ratio,
                non_oscillatory)


@partial(jax.jit, static_argnames=_STATIC + ("max_pieces",))
def _initial_state(a, b, k, ell, owner, spline, max_ratio, non_oscillatory, lead, n_geometric, *, bessel_type, n_col,
                   max_subintervals, max_pieces):
    """Start every integral with [a, b] split into its initial pieces.

    Where the ratio cap applies, [a, b] is split straight into [a, x_t] (if
    a < x_t) and geometric pieces from max(a, x_t) to b with ratio below
    ``max_ratio``. Each piece is solved once, instead of forcing one split per
    round and discarding the intermediate halves. A single piece reproduces
    pylevin's start: I_half = 0, so its first error estimate is I_full^2. All
    integrals start with ``previous = 0``, which forces at least one
    bisection, as in pylevin.
    """
    a, b, k = (jnp.asarray(v, dtype=float) for v in (a, b, k))
    valid = b > a
    piece = jnp.arange(max_pieces)[None, :]
    lead2, n = lead[:, None], n_geometric[:, None]
    start = jnp.where(lead == 1, _cap_start(k, ell), a)[:, None]
    log_ratio = jnp.log(jnp.where((start > 0) & (b[:, None] > start), b[:, None] / jnp.where(start > 0, start, 1.0), 1.0))
    q = piece - lead2  # index within the geometric pieces
    is_lead = (lead2 == 1) & (piece == 0)
    left = jnp.where(is_lead, a[:, None], jnp.where(q == 0, start, start * jnp.exp(log_ratio * q / n)))
    right = jnp.where(is_lead, start, jnp.where(q == n - 1, b[:, None], start * jnp.exp(log_ratio * (q + 1) / n)))
    in_use = piece < lead2 + n
    left, right = jnp.where(in_use, left, 0.0), jnp.where(in_use, right, 0.0)

    k2, ell2, owner2 = k[:, None], ell[:, None], owner[:, None]
    w_left = collocation.endpoint_w(bessel_type, ell2, k2, left)
    w_right = collocation.endpoint_w(bessel_type, ell2, k2, right)
    full, err = _estimates(left, right, w_left, w_right, k2, ell2, owner2, spline, bessel_type, n_col, max_ratio,
                           non_oscillatory)
    single_piece = (lead + n_geometric == 1)[:, None]
    err = jnp.where(single_piece, jnp.where(_too_wide(left, right, k2, ell2, max_ratio), jnp.inf, full ** 2), err)
    usable = in_use & valid[:, None]
    full, err = jnp.where(usable, full, 0.0), jnp.where(usable, err, 0.0)
    result = jnp.sum(full, axis=-1)

    widen = lambda v: jnp.pad(v, ((0, 0), (0, max_subintervals + 2 - max_pieces)))
    return BisectionState(
        left=widen(left),
        right=widen(right),
        estimate=widen(full),
        error=widen(err),
        used=widen(in_use),
        n_used=(lead + n_geometric).astype(jnp.int32),
        result=result,
        previous=jnp.zeros_like(result),
        done=~valid,
    )


@partial(jax.jit, static_argnames=("bessel_type", "n_col"))
def _reestimate(state, k, ell, owner, spline, max_ratio, non_oscillatory, *, bessel_type, n_col):
    k2, ell2, owner2 = k[:, None], ell[:, None], owner[:, None]
    w_left = collocation.endpoint_w(bessel_type, ell2, k2, state.left)
    w_right = collocation.endpoint_w(bessel_type, ell2, k2, state.right)
    full, err = _estimates(state.left, state.right, w_left, w_right, k2, ell2, owner2,
                           spline, bessel_type, n_col, max_ratio, non_oscillatory)
    full = jnp.where(state.used, full, 0.0)
    err = jnp.where(state.used, err, 0.0)
    result = jnp.sum(full, axis=-1)
    valid = state.right[:, 0] > state.left[:, 0]
    return state._replace(estimate=full, error=err, result=result, previous=result, done=~valid)
