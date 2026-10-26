"""GPU-accelerated Levin integration of oscillatory Bessel integrals in JAX.

Computes integrals of the type

    I(k_1, ...) = int_a^b dx f(x) J_ell_1(k_1 x) [J_ell_2(k_2 x)]

where J_ell is a spherical (``bessel_type=0``) or cylindrical
(``bessel_type=1``) Bessel function of the first kind. This is a JAX port of
``pylevin`` (https://github.com/rreischke/levin_bessel); integrals over one
(:class:`LevinSingle`) and two (:class:`LevinDouble`) Bessel functions are
implemented.

Levin collocation is ill-conditioned in single precision, so importing this
package enables JAX's 64-bit mode.
"""

import jax

jax.config.update("jax_enable_x64", True)

from levin.integrator import LevinDouble, LevinSingle  # noqa: E402

__all__ = ["LevinSingle", "LevinDouble"]
__version__ = "0.1.0.dev0"
