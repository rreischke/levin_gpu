import numpy as np
import pytest

from levin import chebyshev


@pytest.mark.parametrize("n_col", [4, 8, 16, 64])
def test_nodes_match_pylevin(n_col):
    j = np.arange(n_col)
    norm = -1.0 / np.cos(np.pi * (1.0 + 2 * n_col) / (2 * n_col))
    expected = norm * np.cos((2.0 * (j + 1) - 1) / (2.0 * n_col) * np.pi + np.pi)
    nodes = np.asarray(chebyshev.nodes(n_col))
    np.testing.assert_allclose(nodes, expected, rtol=1e-14, atol=1e-15)
    assert nodes[0] == pytest.approx(-1.0) and nodes[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("n_col", [2, 8, 33])
def test_basis_matches_numpy(n_col):
    y = np.linspace(-1, 1, 17)
    values, derivatives = chebyshev.basis(y, n_col)
    for m in range(n_col):
        T_m = np.polynomial.chebyshev.Chebyshev.basis(m)
        np.testing.assert_allclose(values[:, m], T_m(y), atol=1e-12)
        np.testing.assert_allclose(derivatives[:, m], T_m.deriv()(y), rtol=1e-12, atol=1e-10)


def test_map_to_interval():
    assert float(chebyshev.map_to_interval(-1.0, 2.0, 5.0)) == 2.0
    assert float(chebyshev.map_to_interval(1.0, 2.0, 5.0)) == 5.0
