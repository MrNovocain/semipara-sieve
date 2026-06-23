import numpy as np

from pseel.basis import PolynomialBasis
from pseel.residualize import Residualizer


def test_residual_is_orthogonal_to_basis():
    w = np.linspace(-1.0, 1.0, 40)
    P = PolynomialBasis(K=4).make(w)
    r = Residualizer(P).residualize(np.sin(w) + w**3)
    assert np.linalg.norm(P.T @ r) < 1e-9


def test_residualizing_basis_column_gives_zero():
    w = np.linspace(-1.0, 1.0, 40)
    P = PolynomialBasis(K=4).make(w)
    r = Residualizer(P).residualize(P[:, 2])
    assert np.linalg.norm(r) < 1e-9


def test_residualizing_constant_gives_zero_with_intercept():
    w = np.linspace(-1.0, 1.0, 40)
    P = PolynomialBasis(K=3).make(w)
    r = Residualizer(P).residualize(np.ones_like(w))
    assert np.linalg.norm(r) < 1e-9
