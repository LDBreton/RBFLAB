"""Global closure checks independent of local interpolation residuals."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research.shared_pressure_lhi.solver import GaugedSystem, experiment


def test_gauge_preserves_compatible_equations_and_exposes_incompatible_rhs():
    # Velocity determined; pressure difference determined; constant free.
    matrix = csr_matrix([[2., 0., 0.], [0., 1., -1.], [0., -1., 1.]])
    system = GaugedSystem(matrix, pressure_count=2)
    solution, info = system.solve(np.array([4., 3., -3.]))
    np.testing.assert_allclose(solution, [2., 1.5, -1.5], atol=1e-14)
    assert info['numerical_nullity'] == 1
    assert info['max_original_residual'] < 1e-14
    _, incompatible = system.solve(np.array([4., 3., 0.]))
    assert abs(incompatible['compatibility_multiplier']) > .1
    assert incompatible['max_original_residual'] > .1


def test_additional_nullspace_is_not_silently_regularized():
    with pytest.raises(ValueError, match='Additional unresolved null modes'):
        GaugedSystem(csr_matrix(np.diag([1., 0., 0.])), pressure_count=2)


def test_global_closure_reports_rank_failure_instead_of_fabricating_solution():
    result = experiment(backend='python', pressure_degree=2, steps=2)
    assert len(result['results']) == 6
    for row in result['results']:
        assert row['status'] == 'rank_deficient'
        assert row['numerical_nullity'] > 1
        assert row['constant_pressure_defect'] < 1e-12
        assert 'velocity_max' not in row
