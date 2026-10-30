"""Small DAE tests; no native weight assembly required."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from research.navier_stokes_schemes.transport import evolve


def problem():
    zero = csr_matrix((2, 2))
    # Singular M mimics pressure observations with an algebraic component.
    return SimpleNamespace(
        M=csr_matrix(np.diag([1., 0.])), A=eye(2), B=zero,
        PY=zero, PB=zero, PL=eye(2),
        DxU=zero, DxB=zero, DxL=zero, DyU=zero, DyB=zero, DyL=zero,
        points=np.array([[.5, .5]]), bounds=np.array([[0., 1.], [0., 1.]]),
        boundary=lambda t: np.zeros(2), force=lambda t: np.zeros(2),
        initial=lambda: (np.array([.1, 0.]), np.array([.1, 0.])),
        evaluate=lambda u, ell, t, x: np.tile(u, (len(x), 1)),
    )


@pytest.mark.parametrize("scheme", ["imex", "imex_ars222"])
def test_imex_second_order_with_singular_mass(scheme):
    errors = []
    for dt in (.05, .025):
        result = evolve(problem(), scheme, dt, round(.2/dt))
        errors.append(abs(result["U"][0] - .1*np.exp(-.2)))
        assert result["diagnostics"]["momentum_residual_max"] < 1e-12
    assert 3.9 < errors[0]/errors[1] < 4.2


def test_ars222_second_order_with_momentum_dependent_convection():
    errors = []
    # U' + U + U^2 = 0 with L=U and C=U*(DxL L).
    for dt in (.02, .01):
        p = problem()
        p.DxL = eye(2)
        result = evolve(p, "imex_ars222", dt, round(.2/dt))
        exact = .1*np.exp(-.2)/(1 + .1*(1-np.exp(-.2)))
        errors.append(abs(result["U"][0]-exact))
    assert 3.8 < errors[0]/errors[1] < 4.2


@pytest.mark.parametrize("scheme", ["characteristic", "characteristic_bdf2"])
def test_characteristics_use_mixed_evaluator_and_preserve_constraint(scheme):
    p = problem()
    calls = []
    original = p.evaluate
    def evaluate(u, ell, t, points):
        calls.append((t, ell.copy()))
        return original(u, ell, t, points)
    p.evaluate = evaluate
    result = evolve(p, scheme, .02, 3)
    assert calls
    np.testing.assert_allclose(result["L"], result["U"], atol=1e-13)
    assert result["U"][0] < .1


def test_departure_outside_requires_explicit_boundary_policy():
    p = problem()
    p.points = np.array([[.001, .5]])
    with pytest.raises(ValueError, match="left domain"):
        evolve(p, "characteristic", .1, 1)


def test_fixed_mixed_reconstruction_retains_original_exclusion():
    import rbflab as r
    from research.navier_stokes_schemes.reconstruction import FixedStencilReconstruction

    cloud = r.geometry.unit_box_grid(3)
    x, b = cloud.interior, cloud.points[cloud.boundary_indices]
    space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(r.Gaussian(1), 1),
        "p": r.PressureSpace(r.Gaussian(1), 1),
    })
    u, pressure = space.fields
    momentum = -r.laplacian(u) + r.gradient(pressure)
    source = {"velocity": r.Samples(x, operator=u),
              "boundary": r.Samples(b, operator=u),
              "momentum": r.Samples(x, operator=momentum, target="exclude")}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
                                 precision=r.Precision(local_digits=35))
    maps = local.operators(targets=x, operators={"L": momentum})
    fit = FixedStencilReconstruction(local, maps.L, u)
    for i, memberships in enumerate(fit.memberships):
        assert i not in memberships["momentum"]
        assert len(memberships["momentum"]) == len(x)-1
    queries = x + [.01, -.02]
    # Constant velocity with pressure p=x-y: nonzero momentum observations.
    values = lambda points: np.tile([1., -1.], (len(points), 1))
    answer = fit.evaluate(values(x), values(x), values(b), queries)
    np.testing.assert_allclose(answer, values(x), atol=1e-11)
    fit.evaluate(values(x), values(x), values(b), queries)
    assert fit.builds == 1 and fit.cache_hits == 1
