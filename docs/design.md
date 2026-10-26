# Design: GPU Levin integration in JAX

Scope so far: integrals over one or two Bessel functions,

$$I(k_1, k_2) = \int_a^b f(x)\, \mathcal{J}_{\ell_1}(k_1x)\, [\mathcal{J}_{\ell_2}(k_2x)]\, \mathrm{d}x,$$

spherical (`bessel_type=0`) or cylindrical (`bessel_type=1`), matching
`pylevin.levin_integrate_bessel_single` (`LevinSingle`; pylevin types 0, 1)
and `levin_integrate_bessel_double` (`LevinDouble`; pylevin types 2, 3).

## Products of Bessel functions

`collocation.py` handles a product of N Bessel functions in one code path.
The Levin basis w is the outer product of the per-factor pairs
(J_ℓ(kx), J_ℓ+1(kx)), with `pylevin`'s component order i = i₁ + 2i₂ + 4i₃, so
w₀ is the product itself. A(x) is the Kronecker sum of the single-Bessel
2 × 2 matrices; this reproduces `pylevin::A_matrix_double` entry by entry.
The collocation system has d = 2^N components, so it is (2^N n_col)²:
16 × 16 for one Bessel function and 32 × 32 for two, at n_col = 8. The
bisection, ratio cap, caching and weights are shared. `k` and `ell` carry a
trailing axis over factors, and the cap uses the earliest turning point,
min over factors of (ℓᵢ + 1)/kᵢ. Three Bessel functions need only a
`LevinTriple` front end.

## Solver choice: dense collocation, not banded

`pylevin` has two single-Bessel solvers: dense Chebyshev collocation, and a
banded Chebyshev–Galerkin solver. The C++ code only uses the banded solver when
`n_col >= 64`. We port the **dense** solver because:

| | dense | banded |
|---|---|---|
| system size at default `n_col=8` | 16 × 16 | 12 × 12, bandwidth 7 |
| GPU primitive | batched LU (`jnp.linalg.solve`, cuSOLVER) | none; needs a custom kernel |
| parallel work inside one system | O(n³) dense flops | O(n) sequential elimination steps |
| control flow | uniform | no pivoting; falls back to dense; residual checks |
| intervals containing x = 0 | yes | no, needs a dense fallback |
| cost to re-run with new f(x) | one matrix product (see below) | banded solve plus boundary solve |

On a GPU the parallelism comes from the batch of thousands of subintervals,
not from inside each system. Small dense solves saturate the device, while
the banded solver's sequential recurrence and branching do not. If very large
`n_col` turns out to be necessary, the banded solver can be added later as a
`jax.lax.scan` kernel without changing the API.

## Subintervals belong to each integral

An *integral* is one (k, ℓ, integrand) combination; in diagonal mode, one
(k, ℓ) per integrand. As in `pylevin`, the adaptive bisection finds
**separate subintervals for every integral**, and nothing is shared between
integrands. `pylevin` stores them on the first call (`bisection_set`) and
reuses them on every later call, including after `update_integrand`. That
reuse is what makes repeat calls an order of magnitude faster.

## Levin as a quadrature rule on each subinterval

On a given subinterval, the dense Levin estimate is linear in the sampled
integrand:

$$I = L^T G^{-1} F = \mathbf v \cdot f(\mathbf x_j), \qquad G^T \mathbf v = L.$$

`G` and `L` depend on (a, b, k, ℓ) but not on f. So:

* **First call:** run the adaptive bisection (below), then do one batched
  solve per stored subinterval to get its weights `v` (`n_col` values each).
* **Repeat call:** interpolate f at each subinterval's nodes and sum
  `v · f(x_j)` over that integral's subintervals. This is elementwise work
  followed by a reduction, and it is differentiable with respect to f.

This replaces `pylevin`'s cached LU factors and `w_precomp`. It stores
`n_col` numbers per subinterval instead of `(2 n_col)²`.

## Adaptive bisection on the GPU

`pylevin::iterate_single` refines one integral at a time, which is serial
and data-dependent. That is a GPU problem in any framework, not only in JAX:
in PyTorch every convergence check would read a value back to the CPU and
stall the GPU. The remedy in both is to refine **all integrals together**.

`levin/adaptive.py` does this with fixed-size buffers, so JAX's static-shape
requirement is not an obstacle:

* Each integral has `max_subintervals + 2` slots (the most
  `iterate_single` can produce). All per-subinterval arrays have shape
  `(n_integrals, max_subintervals + 2)`, and unused slots are masked.
* Each round, every unconverged integral splits its worst subinterval, with
  ties going to the leftmost subinterval as in `pylevin`. All new halves are
  solved at `n_col` and `n_col/2` in one batched call.
* The convergence criterion is the same as `pylevin::iterate_single`,
  including its start value `I_half = 0`. So the subintervals, and the
  results, match `pylevin` up to rounding.
* The rounds run in a jitted `lax.while_loop`, with no dynamic shapes inside.

**Shrinking batches.** Most integrals converge within a few rounds, but a few
can run all `max_subintervals + 1` rounds. If finished integrals were only
masked, about 90% of the work would be wasted (measured on the example from
the paper). So the while loop runs on a batch that holds only the active
integrals. When fewer than a quarter of that batch is still active, a short
host loop gathers the rest into a batch 4x smaller (sizes 16·4ʲ). Wasted
work stays below 4x, and only a few shapes are ever compiled. This made the
first call about 6x faster on CPU.

**New integrands.** By default, `update_integrand` reuses the stored
subintervals, as `pylevin` does. With `recheck=True`, the bisection loop
restarts from the stored subintervals instead of from [a, b]. Integrals that
still meet the tolerance converge after one round and the rest are refined
further. This is an improvement over `pylevin`, which never checks whether
the old subintervals still suit a new integrand.

## Modules

| module | contents | status |
|---|---|---|
| `chebyshev.py` | nodes, T_m and T_m′, interval map | done, tested |
| `bessel.py` | j_ℓ, J_ℓ for integer ℓ in JAX: power series / forward recurrence / Miller | done, tested (≲1e-13 vs mpmath) |
| `interpolation.py` | GSL-compatible Akima spline, log x / log f options | done, tested (matches GSL to 1e-13) |
| `collocation.py` | G, L, weights `v`, per-interval estimate, for products of N Bessel functions | done, tested |
| `adaptive.py` | fixed-buffer bisection for all integrals in shrinking batches, `recheck` | done, tested |
| `integrator.py` | `LevinSingle`, `LevinDouble` public API | done, matches `pylevin` to ~1e-9 (median) |

## Precision and platforms

* float64 throughout (`jax_enable_x64` is set on import). Collocation systems
  are too ill-conditioned for float32.
* Apple MPS has no float64, so development on macOS runs on the JAX CPU
  backend. GPU runs need CUDA (`pip install levin-gpu[cuda]`).

## Testing

* Unit tests per module, checked against NumPy/SciPy, with mpmath as the
  high-precision reference. SciPy's `jv` is off by up to 1e-9 at z ~ 1e6,
  so it is not used as a reference at large z.
* `notebooks/levin_single_tests.ipynb` checks `LevinSingle` end to end:
  integrals with closed-form results, the example from the paper, repeat calls
  and `update_integrand`, and batching over many integrands.
* `tests/test_reference.py` compares results against the C++ `pylevin` when it
  is installed.

## False convergence and the ratio cap

`pylevin`'s error estimate compares the Levin estimate with `n_col` and
`n_col/2` nodes on each subinterval. On subintervals that span many decades
in x, both estimates share the same resolution error and agree with each
other. The bisection then stops while the result is off by up to ~1e-2. On
the example from the paper with `pylevin`'s default settings, 77 of 1000
k values were affected, and `pylevin` failed at exactly the same k with the
same errors.

Cause: the slowly varying Levin solution carries ℓ/(kx)-type terms (from the
ℓ/x entries of A), which a Chebyshev polynomial in linear x cannot resolve on
an interval with large b/a. Increasing `n_col` barely helps; on
[1e-5, 50] the error stays at 2e-2 for every `n_col` up to 32. Splitting at
the linear midpoint does not help either: the left child keeps almost the
same ratio. It is not the spline: a 100x denser table gives the same errors.

Fix: `set_levin(max_ratio=5)` (the default). Let x_t = (ℓ + 1)/k, roughly
where the Bessel function starts to oscillate. A subinterval [a, b] with
b > 5 · max(a, x_t) counts as unconverged (infinite error estimate). It is
split at x_t if it starts below it, otherwise at its geometric midpoint.
Below x_t, wide subintervals converge fine: [1e-5, 1] at k = 1.7 is accurate
to 2e-5 with 8 nodes. So the cap only adds pieces where they are needed.
Integrals that need it start out already split into [a, x_t] plus geometric
pieces up to b, so each piece is solved once and no rounds are spent on
forced splits.

Measured against tight references (n_col = 32, rel_acc = 1e-12) on four
integrands (paper example j_5, cylindrical J_0, j_20, j_100), 1000 k each,
rel_acc 1e-4 and 1e-6: no false convergence. Without the cap: 93 (paper
example) and 390 (j_20) of 1000 k were falsely converged.

Tried and rejected:
* capping b/a everywhere, not only above x_t: also correct, but about 2x
  more subintervals;
* x_t = 2(ℓ + 1)/k or 4(ℓ + 1)/k: false convergence returns (j_100, paper
  example);
* starting pre-split integrals with `previous = result` (skipping
  pylevin's forced first bisection): no measurable speed-up, so the safer
  forced bisection is kept.

Cost (first call, 1000 k, rel_acc = 1e-4, CPU): 10 integrands 330 ms without
the cap and 783 ms with it (1378 ms with the first, cap-everywhere
version); 50 integrands 1.8 s and 3.4 s (6.0 s). The cost follows the
subinterval count (3.4 vs 7.4 per integral), which is the price of the fix.
Repeat calls stay one weighted sum per subinterval. `max_ratio=None`
reproduces `pylevin` exactly and is used by the reference tests.

## Memory

Peak memory is now dominated by stored data: the bisection state
(n_integrals × (max_subintervals + 2) slots) and the cached nodes and
weights (2 × n_col values per subinterval). Temporaries are bounded by
solving weights in chunks of 2^16 subintervals and refining at most 2^16
integrals per batch. Before this, solving all weights at once peaked at
6.4 GB for 50 integrands × 1000 k. Measured on CPU with `rel_acc=1e-6`:
~1 GB for 10 integrands × 1000 k, ~1.8 GB for 50, ~3.3 GB for 200.

## Two Bessel functions: validation and speed

* **Exact mode vs `pylevin`** (`max_ratio=None`, `super_accurate=True`):
  median agreement 1e-12 to 1e-14; worst case 5e-5 at `rel_acc=1e-4`, where
  a few integrals take different bisection paths.
* **`pylevin`'s default Bessel functions are inaccurate at large argument.**
  With GSL's j_ℓ, `pylevin`'s result for the paper's I₂ (j₁₀ j₅) is off by up
  to 25% of the local amplitude at k > 300, compared with a tight reference.
  With Boost (`super_accurate=True`) it is off by at most 5e-5, and this code by
  3e-5. The reference tests therefore use `super_accurate=True`.
* **Closed forms** (Weber's second exponential integral, spherical and
  cylindrical, k₂/k₁ = 0.5, 1, 2): 2e-7 to 3e-6 of the peak at `rel_acc=1e-6`.
* **False convergence** is as common for products as for single Bessel
  functions: without the cap, 783 cases over four test integrands and two
  tolerances (j₃ j₇: up to 60% off). With the cap: none (3 apparent cases
  were within the spread of the tight references themselves).
* **Speed** (500 k, `rel_acc=1e-4`, CPU, warm, ms):

  | | `pylevin` 1 thread | `pylevin` 4 threads | exact | default |
  |---|---|---|---|---|
  | j₁₀ j₅ (k₁ = k₂), 10 integrands, first call | 1421 | 734 | 1569 | 1592 |
  | j₁₀ j₅, 10 integrands, repeat call | 69 | 66 | 13.5 | 3.9 |
  | j₃ j₇ (k₂ = 2k₁), 10 integrands, first call | 386 | 183 | 474 | 977 |
  | j₃ j₇, 10 integrands, repeat call | 24 | 27 | 1.8 | 4.5 |

## Non-oscillatory pieces

On a piece where (Σᵢ kᵢ)·Δx is small, w varies slowly. Levin's collocation
matrix is then nearly singular: the homogeneous solutions of p′ + Aᵀp = 0
are almost polynomial. Measured condition numbers: 2e11 at kΔx = 0.9, 2e14
at 0.2, 1e17 to 1e19 at 0.05. Two symptoms follow, both shared with `pylevin`:

* when the bisection zooms in on a narrow feature of f at small k, those
  pieces are only good to 1e-4 to 1e-3, and different compiled code paths
  (bisection estimate vs. cached weights) disagree at that level;
* for ℓ = 0 and k(b − a) ≲ 1, `pylevin` was off by O(1) (closed form: 170%
  of the peak); this code by ~1%.

Fix: `set_levin(non_oscillatory=0.5)` (the default). Pieces with
(Σᵢ kᵢ)·Δx < 0.5 use interpolatory quadrature of f·w₀ on the same nodes:
v_j = (b − a)/2 · q_j · w₀(x_j), where q integrates T₀ … T_{n−1} exactly. The
weights are still linear in f, so caching and the n_col vs n_col/2 error
estimate are unchanged. Results:

* ℓ = 0 closed forms, all k: 8e-2 → 3e-7 (spherical), 6e-4 → 1e-7
  (cylindrical) of the peak;
* bump case (narrow feature at small k): repeat call vs. `recheck` result
  7.5e-5 → 1e-9.

Choosing the threshold. On smooth pieces the error estimate measures the
error of the n_col/2-node rule, which is cruder than the result it judges.
Too large a threshold therefore over-splits: at 1.0 an integral of the ν = 4
closed form ran out of subintervals (error 2e-5). Too small a threshold
brings back Levin's ill-conditioning: at 0.25 the bump case's repeat call
differs by 1e-5. 0.5 keeps all closed forms below 4e-6 of the peak at
`rel_acc = 1e-6`, with no false convergence on any test case.

Cost: Bessel functions at every node of the smooth pieces, computed with
`jit`'s usual all-branches evaluation. This makes the first call about 1.8x
slower on CPU (783 → ~1400 ms for 10 integrands × 1000 k). Masking and
skipping batches without smooth pieces did not help; this is the main open
optimisation. `non_oscillatory=None` disables it.

## Fixes ported to `pylevin`

Both fixes (ratio cap with pre-split at x_t, and interpolatory quadrature on
non-oscillatory pieces) are implemented in the C++ `pylevin`
(`set_levin(..., max_ratio=5.0, non_oscillatory=1.0)`; 0 disables either).
The three `iterate_*` routines now share one `iterate_pieces`. Smooth pieces
store their weights for `pylevin`'s cached-LU repeat calls. Validation:

* with both set to 0, all 13 baseline cases (single/double/triple, dense and
  banded, diagonal mode, calls 1–3) reproduce the old build bit for bit;
* with the defaults, `pylevin` and this code agree to 1e-11 to 1e-13
  (paper example, j₂₀) and 2e-6 (J₀ with ℓ = 0, where pieces with
  kΔx ≈ 0.5 to 1 are still ill-conditioned);
* false convergence: 93 → 0 (paper example), 392 → 0 (j₂₀); ℓ = 0 closed
  form 1.7 → 3e-7 of the peak; call 3 vs call 1 in the bump case 6e-5 → 2e-9;
* cost: first and repeat calls about 2x slower (more subintervals).

## Known limitations

* **Compilation.** The first call in a fresh process spends several seconds compiling
  on CPU. JAX's persistent compilation cache
  (`jax.config.update("jax_compilation_cache_dir", ...)`) avoids paying this
  again in later processes.
