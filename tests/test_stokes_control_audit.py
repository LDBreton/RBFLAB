"""Independent discrete-adjoint checks, with no MPI/native dependencies.

The dense all-time reference is deliberately assembled independently of the
reverse sweep. Nonzero momentum-gradient maps and fixed, time-varying wall
values ensure these checks exercise the mixed LHI dependencies.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.optimize import root
from scipy.sparse import csr_matrix

from research.navier_stokes_schemes.stokes_control import StokesControl


def fixture_problem(seed=629):
    rng = np.random.default_rng(seed)
    n, nb = 6, 4

    def matrix(rows=n, cols=n, scale=.15):
        return csr_matrix(scale*rng.normal(size=(rows, cols)))

    return SimpleNamespace(
        A=csr_matrix(1.7*np.eye(n)+.1*rng.normal(size=(n, n))),
        M=csr_matrix(np.eye(n)+.08*rng.normal(size=(n, n))),
        B=matrix(n, nb),
        DxU=matrix(), DyU=matrix(),
        DxL=matrix(scale=.45), DyL=matrix(scale=.35),
        DxB=matrix(n, nb), DyB=matrix(n, nb),
        boundary=lambda t: np.array([.8+.2*t, -.3, .4*t, .6-.1*t]),
        force=lambda t: np.linspace(-.2, .35, n)*(1+t),
        initial=lambda: (np.linspace(.06, .16, n), np.zeros(n)),
    )


def dense_reference(problem, controls, dt, start=0.):
    """Solve one all-time block system, then differentiate its residual."""
    steps, n = controls.shape
    identity = np.eye(n)
    zero = np.zeros((n, n))
    K = np.block([[problem.A.toarray(), -problem.M.toarray()],
                  [identity/dt, identity]])
    H = np.block([[zero, zero], [identity/dt, zero]])
    system = np.kron(np.eye(steps), K)
    system -= np.kron(np.diag(np.ones(steps-1), -1), H)
    E = np.vstack((zero, identity))
    forcing = []
    for k in range(steps):
        t = start+(k+1)*dt
        rhs = np.r_[-problem.B@problem.boundary(t), problem.force(t)]
        if k == 0:
            rhs[n:] += problem.initial()[0]/dt
        forcing.append(rhs)
    selector = np.kron(np.eye(steps), E)
    states = np.linalg.solve(system, np.ravel(forcing)-selector@controls.ravel())
    states = states.reshape(steps, 2*n)
    state_derivative = -np.linalg.solve(system, selector)
    residual = np.empty_like(controls)
    convection_derivatives = []
    for k, state in enumerate(states):
        U, L = np.split(state, 2)
        g = problem.boundary(start+(k+1)*dt)
        dx = problem.DxU@U+problem.DxL@L+problem.DxB@g
        dy = problem.DyU@U+problem.DyL@L+problem.DyB@g
        vx, vy = np.repeat(U[::2], 2), np.repeat(U[1::2], 2)
        convection = vx*dx+vy*dy
        residual[k] = controls[k]-convection
        direct_advector = np.zeros((n, n))
        for node in range(n//2):
            direct_advector[2*node:2*node+2, 2*node] = dx[2*node:2*node+2]
            direct_advector[2*node:2*node+2, 2*node+1] = dy[2*node:2*node+2]
        JU = direct_advector+vx[:, None]*problem.DxU.toarray()+vy[:, None]*problem.DyU.toarray()
        JL = vx[:, None]*problem.DxL.toarray()+vy[:, None]*problem.DyL.toarray()
        convection_derivatives.append(np.hstack((JU, JL)))
    jacobian = np.eye(steps*n)-block_diag(*convection_derivatives)@state_derivative
    return states, residual, jacobian


@pytest.mark.parametrize("steps", [1, 5])
def test_adjoint_matches_independent_all_time_dense_system(steps):
    problem = fixture_problem()
    dt, start = .037, .12
    node_weights = np.array([.13, .27, .18])
    rng = np.random.default_rng(82)
    controls = .3*rng.normal(size=(steps, 6))
    model = StokesControl(problem, dt, steps, start_time=start, node_weights=node_weights)
    actual = model.evaluate(controls)
    states, residual, jacobian = dense_reference(problem, controls, dt, start)
    weights = np.tile(np.repeat(node_weights, 2), steps)*dt
    expected_cost = .5*np.dot(weights*residual.ravel(), residual.ravel())
    expected_gradient = jacobian.T@(weights*residual.ravel())
    np.testing.assert_allclose(actual["states"], states, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual["residual"], residual, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual["cost"], expected_cost, rtol=2e-13)
    np.testing.assert_allclose(actual["gradient"].ravel(), expected_gradient, rtol=3e-12, atol=1e-15)
    np.testing.assert_allclose(actual["l2_gradient"].ravel(), expected_gradient/weights,
                               rtol=3e-12, atol=1e-13)


def test_directional_finite_differences_over_multiple_scales():
    problem = fixture_problem()
    model = StokesControl(problem, .04, 6, node_weights=np.array([.2, .3, .15]))
    rng = np.random.default_rng(921)
    controls = .4*rng.normal(size=(6, 6))
    direction = rng.normal(size=controls.shape)
    direction /= np.linalg.norm(direction)
    value = model.evaluate(controls)
    expected = np.sum(value["gradient"]*direction)
    errors = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        plus = model.evaluate(controls+h*direction, gradient=False)["cost"]
        minus = model.evaluate(controls-h*direction, gradient=False)["cost"]
        fd = (plus-minus)/(2*h)
        errors.append(abs(fd-expected)/max(abs(expected), 1e-30))
    assert min(errors) < 2e-8
    assert errors[1] < errors[0]/30


@pytest.mark.parametrize("weighted", [False, True])
def test_residual_linear_operator_has_correct_transpose(weighted):
    problem = fixture_problem()
    model = StokesControl(problem, .025, 4, node_weights=np.array([.3, .1, .25]))
    rng = np.random.default_rng(196)
    controls = .5*rng.normal(size=(4, 6))
    _, _, independent = dense_reference(problem, controls, .025)
    if weighted:
        independent = np.sqrt(np.tile(np.repeat([.3, .1, .25], 2), 4)*.025)[:, None]*independent
    operator = model.residual_jacobian(controls, weighted=weighted)
    v, w = rng.normal(size=(2, 24))
    np.testing.assert_allclose(operator@v, independent@v, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(operator.rmatvec(w), independent.T@w, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(np.dot(w, operator@v), np.dot(operator.rmatvec(w), v), rtol=3e-13)


def test_each_control_time_influences_its_future_states_only():
    problem = fixture_problem()
    model = StokesControl(problem, .03, 4)
    controls = np.zeros((4, 6))
    reference = model.forward(controls)
    shifted = controls.copy()
    shifted[2, 1] = .4
    changed = model.forward(shifted)
    np.testing.assert_array_equal(changed[:2], reference[:2])
    assert np.linalg.norm(changed[2:]-reference[2:]) > 1e-4


def test_window_reset_and_shared_factor_match_fresh_nonzero_wall_solve():
    problem = fixture_problem()
    model = StokesControl(problem, .03, 3)
    controls = np.full((3, 6), .15)
    initial = tuple(np.split(model.forward(controls)[-1], 2))
    retained = model.factorization
    model.reset_window(initial, .09)
    fresh = StokesControl(problem, .03, 3, initial=initial, start_time=.09)
    shared = StokesControl(problem, .03, 3, initial=initial, start_time=.09,
                           factorization=retained)
    assert model.factorization is retained
    assert shared.factorization is retained
    assert model.factorizations == 1 and shared.factorizations == 0
    for candidate in (model, shared):
        actual, expected = candidate.evaluate(controls), fresh.evaluate(controls)
        np.testing.assert_allclose(actual["states"], expected["states"], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(actual["gradient"], expected["gradient"], rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("method", ["lbfgs", "gauss_newton", "newton_krylov"])
def test_control_zero_matches_independent_implicit_newton_solution(method):
    problem = fixture_problem()
    dt, steps, n = .03, 3, 6
    model = StokesControl(problem, dt, steps)
    control = model.solve(method=method, gtol=1e-13, residual_tolerance=1e-8)
    assert control["converged"]

    def equations(flat):
        states = flat.reshape(steps, 2*n)
        previous = problem.initial()[0]
        values = []
        for k, state in enumerate(states):
            U, L = np.split(state, 2)
            t = (k+1)*dt
            g = problem.boundary(t)
            dx = problem.DxU@U+problem.DxL@L+problem.DxB@g
            dy = problem.DyU@U+problem.DyL@L+problem.DyB@g
            convection = np.repeat(U[::2], 2)*dx+np.repeat(U[1::2], 2)*dy
            values.append(np.r_[problem.A@U-problem.M@L+problem.B@g,
                                 (U-previous)/dt+L+convection-problem.force(t)])
            previous = U
        return np.ravel(values)

    # Start this solve from the Stokes trajectory, not the optimized answer.
    direct = root(equations, model.forward(np.zeros((steps, n))).ravel(), tol=1e-11)
    assert direct.success
    assert np.max(abs(equations(direct.x))) < 2e-12
    np.testing.assert_allclose(control["states"].ravel(), direct.x, rtol=0, atol=2e-8)
