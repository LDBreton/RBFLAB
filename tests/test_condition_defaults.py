import os
import numpy as np
import pytest
import rbflab as r


def test_backend_condition_estimation_is_opt_in():
    for backend in (r.PythonBackend, r.CppBackend, r.CppGlobalBackend, r.TorchBackend):
        assert backend().compute_condition is False
        assert backend(compute_condition=True).compute_condition is True


def test_scalar_lhi_accepts_default_backend_without_estimates():
    problem = r.LinearPDE(-r.Laplacian(), 0, [r.Dirichlet(1)])
    system = r.LHI(r.Gaussian(1), stencil_size=9, polynomial_degree=1,
                   local_backend=r.PythonBackend()).assemble(problem, r.geometry.unit_box_grid(3))
    solution = system.solve()
    assert solution.diagnostics['max_scaled_local_condition'] is None
    assert solution.diagnostics['relative_residual'] < 1e-10


@pytest.mark.parametrize('native', [False, True])
def test_default_and_opt_in_weights_match(native):
    if native and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native integration')
    backend = r.CppBackend if native else r.PythonBackend
    points = r.geometry.unit_box_grid(2).points
    source = {'u': r.Samples(points)}
    space = r.ScalarSpace(r.Gaussian(1), polynomial_degree=1)
    results = []
    for option in ({}, {'compute_condition': True}):
        maps = r.LocalApproximation(source=source, trial=space.representers(source),
            backend=backend(**option)).operators(targets=points[[4]], operators={'lap': r.Laplacian()})
        results.append(maps.lap)
    assert results[0].local(0).diagnostics['scaled_condition'] is None
    assert results[1].local(0).diagnostics['scaled_condition'] > 0
    np.testing.assert_array_equal(results[0].to_scipy().toarray(), results[1].to_scipy().toarray())
