"""Control optimization contracts and reuse of the fixed Stokes factor."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import root
from scipy.sparse import csr_matrix, eye

from research.navier_stokes_schemes.stokes_control import StokesControl


def problem():
    matrix = lambda a: csr_matrix(np.asarray(a, dtype=float))
    return SimpleNamespace(
        A=matrix([[1.2, .3], [-.1, .8]]), M=eye(2, format="csr"),
        B=matrix([[.2, -.1], [.1, .3]]),
        DxU=matrix([[.2, .1], [.03, -.1]]),
        DyU=matrix([[.04, -.02], [.07, .13]]),
        DxL=matrix([[.03, -.01], [.02, .015]]),
        DyL=matrix([[-.01, .04], [.025, -.02]]),
        DxB=matrix([[.3, -.1], [.1, .2]]),
        DyB=matrix([[.1, .2], [-.2, .15]]),
        boundary=lambda t: np.array([.1+.2*t, -.03*t]),
        force=lambda t: np.array([.4+.1*t, -.1]),
        initial=lambda: (np.array([.05, -.02]), np.array([.06, -.025])))


def test_schur_forward_and_transpose_match_full_block():
    model = StokesControl(problem(), .025, 1)
    rhs = np.array([.1, -.4, .3, .8])
    for trans in ("N", "T"):
        matrix = model.matrix.toarray()
        if trans == "T":
            matrix = matrix.T
        expected = np.linalg.solve(matrix, rhs)
        actual = model.lu.solve(rhs, trans=trans)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(matrix@actual, rhs, atol=1e-14)


def test_reset_window_reuses_factor_and_refreshes_time_dependent_walls():
    p = problem()
    model = StokesControl(p, .02, 2)
    first = model.forward(np.zeros((2, 2)))
    initial = (first[-1, :2], first[-1, 2:])
    saved_factor = model.factorization
    model.reset_window(initial, .04)
    new = model.forward(np.zeros((2, 2)))
    independent = StokesControl(p, .02, 2, initial=initial, start_time=.04)
    np.testing.assert_allclose(new, independent.forward(np.zeros((2, 2))), atol=1e-15)
    assert model.factorization is saved_factor
    assert model.factorizations == 1
    shared = StokesControl(p, .02, 2, initial=initial, start_time=.04,
                           factorization=saved_factor)
    assert shared.factorizations == 0
    np.testing.assert_allclose(shared.forward(np.zeros((2, 2))), new, atol=1e-15)


@pytest.mark.parametrize("method", ["lbfgs", "gauss_newton", "newton_krylov"])
def test_optimized_control_solves_same_implicit_navier_stokes(method):
    p = problem()
    model = StokesControl(p, .025, 3, node_weights=np.array([.7]))
    result = model.solve(method=method, maxiter=100, gtol=1e-13,
                         residual_tolerance=1e-9)
    assert result["converged"]
    assert result["factorizations"] == 1
    assert result["adjoint_sweeps"] > 0
    assert result["cost"] <= result["history"][0]["cost"]
    # Independent all-at-once root of BE momentum and mixed constraints.
    def nonlinear_equations(flat):
        states = flat.reshape(3, 4)
        convection = model.convection(states)
        previous = model.initial_U
        equations = []
        for k in range(3):
            rhs = np.r_[model.boundary_rhs[k], model.forces[k]+previous/model.dt]
            equations.append(model.matrix@states[k]-rhs+np.r_[np.zeros(2), convection[k]])
            previous = states[k, :2]
        return np.asarray(equations).ravel()
    reference = root(nonlinear_equations, model.forward(np.zeros((3, 2))).ravel(), tol=1e-11)
    assert reference.success
    np.testing.assert_allclose(result["states"].ravel(), reference.x, atol=2e-10, rtol=1e-9)


def test_newton_krylov_uses_adjoint_descent_and_armijo_cost_decrease():
    model = StokesControl(problem(), .025, 3, node_weights=np.array([.7]))
    result = model.solve(np.full((3, 2), .4), method="newton_krylov",
                         residual_tolerance=1e-10)
    assert result['converged']
    assert len(result['history']) == result['optimizer_iterations']+1
    for previous, current in zip(result['history'], result['history'][1:]):
        assert current['directional_derivative'] < 0
        assert current['linear_relative_residual'] <= 1e-8
        assert current['cost'] <= previous['cost']+1e-4*current['line_search_step']*current['directional_derivative']
    # One adjoint at initialization and each accepted iterate, never a
    # transpose substituted by the nonsymmetric forward map.
    assert result['adjoint_sweeps'] == result['optimizer_iterations']+1


def test_zero_convection_initial_control_is_exact():
    p = problem()
    for name in ("DxU", "DxL", "DxB", "DyU", "DyL", "DyB"):
        setattr(p, name, csr_matrix((2, 2)))
    result = StokesControl(p, .01, 2).solve()
    assert result["converged"]
    assert result["optimizer_iterations"] == 0
    assert result["cost"] == 0
    np.testing.assert_array_equal(result["gradient"], 0)


def test_actual_residual_overrides_optimizer_success():
    result = StokesControl(problem(), .025, 3).solve(
        gtol=1, maxiter=1, residual_tolerance=1e-14)
    assert not result["converged"]
    assert result["residual_inf"] > 1e-14


@pytest.mark.parametrize("weights", [[0.], [-1.], [float("nan")], [.2, .3]])
def test_invalid_nodal_quadrature_is_rejected(weights):
    with pytest.raises(ValueError, match="node_weights"):
        StokesControl(problem(), .01, 1, node_weights=weights)


def test_control_and_window_dimension_guards():
    model = StokesControl(problem(), .01, 2)
    with pytest.raises(ValueError, match="wrong size"):
        model.evaluate(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        model.evaluate(np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="initial"):
        model.reset_window((np.zeros(3), np.zeros(3)), .02)
