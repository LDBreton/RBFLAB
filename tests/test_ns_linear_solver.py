import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research.navier_stokes_schemes.linear import LinearSolver


def test_reused_preconditioner_solves_changing_equations():
    rng = np.random.default_rng(37)
    base = 10*np.eye(30)+rng.normal(size=(30, 30))*.1
    exact = rng.normal(size=30)
    solver = LinearSolver('gmres')
    for change in (0., .02, .1):
        matrix = csr_matrix(base+change*rng.normal(size=(30, 30)))
        answer = solver.solve(matrix, matrix @ exact)
        np.testing.assert_allclose(answer, exact, atol=1e-12, rtol=0)
    assert solver.factorizations == 1
    assert solver.iterations > 0


@pytest.mark.parametrize('ordering', ['COLAMD', 'MMD_AT_PLUS_A'])
def test_equilibration_preserves_equations_and_refreshes_only_on_key_change(ordering):
    rng = np.random.default_rng(37)
    base = 10*np.eye(30)+rng.normal(size=(30, 30))*.1
    exact = rng.normal(size=30)
    rows = np.geomspace(1e-7, 1e7, 30)
    columns = np.geomspace(1e3, 1e-3, 30)
    solver = LinearSolver('gmres_scaled', ordering=ordering)
    for key, change in ((1., 0.), (1.5, .02), (1.5, .03)):
        matrix = csr_matrix(rows[:, None]*(base+change)*columns[None, :])
        rhs = matrix @ exact
        answer = solver.solve(matrix, rhs, key=key)
        np.testing.assert_allclose(answer, exact, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose((matrix @ answer)/rows, rhs/rows, rtol=1e-12, atol=1e-10)
    assert solver.factorizations == 2
    assert not any(solve['fallback'] for solve in solver.solves)
    assert solver.solves[-1]['iterations'] > 0
