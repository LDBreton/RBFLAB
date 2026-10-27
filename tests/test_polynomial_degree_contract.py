import warnings

import numpy as np
import pytest

import rbflab as r
from rbflab.kernels import resolve_polynomial_degree
from rbflab.nodal import Arithmetic, NodalBasis


def test_polynomial_degree_defaults_are_unaugmented():
    assert r.ScalarSpace(r.PHS(5)).polynomial_degree is None
    assert r.RBFFD(r.PHS(5)).polynomial_degree is None
    assert r.GlobalCollocation(r.PHS(5)).polynomial_degree is None


@pytest.mark.parametrize("kernel, expected", [(r.PHS(3), 1), (r.PHS(5), 2), (r.IMQ(), None)])
def test_auto_uses_kernel_minimum(kernel, expected):
    assert resolve_polynomial_degree(kernel, "auto") == expected
    assert r.ScalarSpace(kernel, "auto").degree() == expected


@pytest.mark.parametrize("degree", [None, 0, 1])
def test_explicit_degree_below_cpd_minimum_is_preserved_with_warning(degree):
    with pytest.warns(UserWarning, match="CPD minimum"):
        assert resolve_polynomial_degree(r.PHS(5), degree) == degree
    with pytest.warns(UserWarning, match="CPD minimum"):
        assert r.ScalarSpace(r.PHS(5), degree).degree() == degree


def test_nodal_basis_honors_unaugmented_and_low_explicit_degrees():
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    none = NodalBasis(Arithmetic(r.PHS(5)), points, None)
    low = NodalBasis(Arithmetic(r.PHS(5)), points, 0)
    assert none.powers == []
    assert len(low.powers) == 1


def test_invalid_degree_is_rejected_but_auto_is_supported():
    with pytest.raises(ValueError, match="nonnegative integer"):
        resolve_polynomial_degree(r.PHS(5), -1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        resolve_polynomial_degree(r.PHS(5), 1.0)
