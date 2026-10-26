# levin-gpu

A JAX implementation of Levin's method for oscillatory integrals over Bessel
functions. It runs on CPU and CUDA GPUs and is ported from
[`pylevin`](https://github.com/rreischke/levin_bessel)
([arXiv:2502.12142](https://arxiv.org/abs/2502.12142)).

**Status:** integrals over one (`LevinSingle`) and two (`LevinDouble`) Bessel functions work and match `pylevin` (median agreement ~1e-9 to 1e-12). Three Bessel functions are not implemented yet.

$$I(k_1, k_2) = \int_a^b f(x)\, \mathcal{J}_{\ell_1}(k_1x)\, [\mathcal{J}_{\ell_2}(k_2x)]\, \mathrm{d}x .$$

See [docs/design.md](docs/design.md) for the algorithm, the GPU strategy and known limitations.

## Installation

```shell
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"             # CPU
pip install -e ".[test,cuda]"        # NVIDIA GPU (CUDA 12)
pip install -e ".[test,notebook]"    # plus Jupyter for the test notebook
```

Current JAX releases need NumPy 2, so use a fresh environment if your base
environment pins NumPy 1.x.

## Usage

```python
import numpy as np
import levin

x = np.geomspace(1e-5, 100, 100)
f_of_x = (x**3 + x**2 + x)[:, None]
integrator = levin.LevinSingle(0, x, f_of_x, logx=True, logy=True)

k = np.geomspace(1e-3, 1e4, 1000)
ell = np.full_like(k, 5, dtype=int)
integrator.set_levin(n_col=8, max_subintervals=32, rel_acc=1e-4)
result = integrator.integrate(x[0] * np.ones_like(k), x[-1] * np.ones_like(k), k, ell)  # (1000, 1)

# Repeat calls reuse the subintervals and cost one weighted sum per integral.
integrator.update_integrand(x, (x**2.5)[:, None], recheck=True)
result = integrator.integrate(x[0] * np.ones_like(k), x[-1] * np.ones_like(k), k, ell)

# Two Bessel functions: int f(x) j_10(k x) j_5(k x) dx
double = levin.LevinDouble(0, x, f_of_x)
result = double.integrate(x[0] * np.ones_like(k), x[-1] * np.ones_like(k), k, k,
                          np.full_like(k, 10, dtype=int), np.full_like(k, 5, dtype=int))
```

`bessel_type` is 0 (spherical) or 1 (cylindrical) for both classes; this
corresponds to `pylevin` types 0/1 and 2/3. The defaults include two accuracy
fixes (see [docs/design.md](docs/design.md)): a ratio cap against false
convergence (`max_ratio`) and interpolatory quadrature on non-oscillatory
pieces (`non_oscillatory`). Both are also in the C++ `pylevin` from
`levin_bessel`'s working tree. `set_levin(max_ratio=None, non_oscillatory=None)`
reproduces the original `pylevin` algorithm exactly.

## Testing

```shell
pytest
```

If the C++ `pylevin` is installed, the tests also compare against it.
[notebooks/levin_single_tests.ipynb](notebooks/levin_single_tests.ipynb) runs
end-to-end checks of the integrator, with plots.

## Citing

If you use this code, please cite [arXiv:2502.12142](https://arxiv.org/abs/2502.12142).
