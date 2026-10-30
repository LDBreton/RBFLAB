"""Small algebraic checks; no native weight assembly or MPI required."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from research.navier_stokes_schemes.coupled import convection, evolve


def problem(seed=8):
    rng = np.random.default_rng(seed)
    n = 4
    matrix = lambda: csr_matrix(.04*rng.normal(size=(n, n)))
    zero = csr_matrix((n, n))
    return SimpleNamespace(
        A=eye(n), B=zero, M=eye(n), SL=zero, PY=eye(n), PB=zero, PL=zero,
        DxU=matrix(), DxB=matrix(), DxL=matrix(),
        DyU=matrix(), DyB=matrix(), DyL=matrix(),
        force=lambda t: np.ones(n), boundary=lambda t: np.zeros(n),
        initial=lambda: (np.zeros(n), np.zeros(n)))


@pytest.mark.parametrize("rotational", [False, True])
def test_exact_mixed_convection_jacobian(rotational):
    p = problem()
    rng = np.random.default_rng(9)
    U, L, g, du, dl = rng.normal(size=(5, 4))
    _, JU, JL = convection(p, U, L, g, rotational)
    h = 1e-6
    plus = convection(p, U+h*du, L+h*dl, g, rotational)[0]
    minus = convection(p, U-h*du, L-h*dl, g, rotational)[0]
    np.testing.assert_allclose((plus-minus)/(2*h), JU @ du+JL @ dl,
                               rtol=1e-8, atol=1e-9)
    value, ju, jl = convection(p, U, L, g, rotational, jacobian=False)
    np.testing.assert_allclose(value, convection(p, U, L, g, rotational)[0],
                               rtol=1e-14, atol=1e-14)
    assert ju is None and jl is None


@pytest.mark.parametrize("scheme", ["oseen", "implicit", "oseen_newton", "rotational"])
def test_coupled_linear_limit_and_bdf2_startup(scheme):
    p = problem()
    for key in ("DxU", "DxB", "DxL", "DyU", "DyB", "DyL"):
        setattr(p, key, csr_matrix((4, 4)))
    dt = .1
    out = evolve(p, scheme, dt, 2)
    u1 = dt/(1+dt)
    u2 = (2*u1/dt+1)/(1.5/dt+1)
    np.testing.assert_allclose(out["U"], u2)
    np.testing.assert_allclose(out["L"], out["U"])
    np.testing.assert_allclose(out["gradient"], out["U"])


@pytest.mark.parametrize("scheme", ["implicit", "oseen_newton", "rotational"])
def test_nonlinear_solve_controls_full_residual(scheme):
    out = evolve(problem(), scheme, .1, 3)
    assert all(d["nonlinear_residual_inf"] <= d["tolerance"] for d in out["diagnostics"])
    assert all(d["constraint_residual_inf"] < 1e-12 for d in out["diagnostics"])


def test_rotational_recovers_physical_gradient():
    p = problem()
    out = evolve(p, "rotational", .1, 1)
    U, L = out["U"], out["L"]
    dx = (p.DxU @ U+p.DxL @ L).reshape(-1, 2)
    dy = (p.DyU @ U+p.DyL @ L).reshape(-1, 2)
    v = U.reshape(-1, 2)
    kinetic = np.column_stack((np.sum(v*dx, axis=1), np.sum(v*dy, axis=1))).ravel()
    np.testing.assert_allclose(out["gradient"], U-kinetic)


def test_unresolved_nonlinearity_is_reported():
    with pytest.raises(RuntimeError, match="failed at step"):
        evolve(problem(), "implicit", 1., 1, rtol=0, atol=0, max_iterations=1)


def test_invalid_step():
    with pytest.raises(ValueError):
        evolve(problem(), "implicit", 0, 1)


@pytest.mark.parametrize('method,ordering', [('gmres', 'COLAMD'),
                                          ('gmres_scaled', 'COLAMD'),
                                          ('gmres_scaled', 'MMD_AT_PLUS_A')])
def test_krylov_oseen_matches_direct_fields(method, ordering):
    p = problem()
    direct = evolve(p, 'oseen', .01, 8)
    p.linear_solver = method
    p.global_ordering = ordering
    iterative = evolve(p, 'oseen', .01, 8)
    for name in ('U', 'L', 'gradient'):
        np.testing.assert_allclose(iterative[name], direct[name], atol=1e-12, rtol=0)
    assert iterative['diagnostics'][-1]['direct_check_gradient_max'] < 1e-12
