"""Independent characteristic BE algebra checks; synthetic fields, no native build.

These checks establish time-integrator identities only. They do not certify the
37-site corrected reconstruction or cavity convergence.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from research.navier_stokes_schemes.transport import evolve


@pytest.fixture(params=['transport', 'characteristic_cavity'])
def advance(request):
    if request.param == 'transport':
        return lambda p, dt, steps: evolve(p, 'characteristic', dt, steps)
    from research.navier_stokes_schemes.characteristic_cavity import evolve as cavity_evolve
    def run(p, dt, steps):
        evaluator = SimpleNamespace(evaluate=lambda u, ell, g, points:
                                    p.evaluate(u, ell, 0., points))
        return cavity_evolve(p, evaluator, dt, steps)
    return run


def shear_problem(curvature=0.):
    """A frozen divergence-free mixed field with momentum-dependent shear."""
    zero = csr_matrix((2, 2))
    origin = np.array([[.5, .5]])
    decay, shear = .7, 1.3
    initial = np.array([.3, -.2])
    calls = []

    def evaluate(u, ell, t, points):
        dy = np.asarray(points)[:, 1] - origin[0, 1]
        calls.append(np.asarray(points).copy())
        result = np.tile(u, (len(points), 1))
        result[:, 0] += ell[0]*(shear*dy + curvature*dy**2)
        return result

    p = SimpleNamespace(
        M=eye(2), A=decay*eye(2), B=zero,
        PY=zero, PB=zero, PL=eye(2),
        DxU=zero, DxB=zero, DxL=zero,
        DyU=zero, DyB=zero, DyL=csr_matrix(np.diag([shear, 0.])),
        points=origin, bounds=np.array([[0., 1.], [0., 1.]]),
        initial=lambda: (initial.copy(), decay*initial.copy()),
        boundary=lambda t: np.zeros(2), force=lambda t: np.zeros(2),
        evaluate=evaluate,
    )
    return p, initial, decay, shear, calls


def test_characteristic_be_evaluates_curved_mixed_field_at_departure(advance):
    p, u, decay, shear, calls = shear_problem(curvature=17.)
    dt = .08
    result = advance(p, dt, 1)
    displacement = -dt*u[1]
    history = u.copy()
    history[0] += decay*u[0]*(shear*displacement + 17.*displacement**2)
    np.testing.assert_allclose(result['U'], history/(1+dt*decay), atol=2e-15)
    assert any(np.max(abs(points-p.points)) > 0 for points in calls)


def test_characteristic_be_small_step_matches_frozen_oseen_to_first_order(advance):
    errors = []
    for dt in (.004, .002, .001):
        p, u, decay, shear, _ = shear_problem()
        result = advance(p, dt, 1)
        # Frozen advector u and implicit same-field gradient: u_y*a*L_x.
        oseen = u/np.array([1+dt*decay*(1+shear*u[1]), 1+dt*decay])
        errors.append(np.linalg.norm(result['U']-oseen, np.inf))
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=.01)


def test_characteristic_be_does_not_clip_a_departure_point(advance):
    p, _, _, _, _ = shear_problem()
    p.points = np.array([[.001, .5]])
    with pytest.raises(ValueError, match='left (physical )?domain'):
        advance(p, .1, 1)


def test_characteristic_be_pressure_gradient_fixed_point(advance):
    p, _, _, _, _ = shear_problem()
    force = np.array([.5, .5])  # grad(x*y) at this arrival point.
    p.M = csr_matrix((2, 2))
    p.A = eye(2)
    p.initial = lambda: (np.zeros(2), force.copy())
    p.force = lambda t: force.copy()
    p.evaluate = lambda u, ell, t, points: np.tile(u, (len(points), 1))
    result = advance(p, .03, 7)
    np.testing.assert_allclose(result['U'], 0., atol=1e-15)
    np.testing.assert_allclose(result['L'], force, atol=1e-15)
