"""Local radius conventions are explicit and retain physical derivative units."""
import numpy as np
import pytest

import rbflab as r


@pytest.mark.parametrize('mixed', [False, True])
def test_local_radius_and_derivative_units_at_mesh_node(mixed):
    points = np.array([[0., 0.], [2., 0.], [0., 1.]])
    scalar = r.ScalarSpace(r.Gaussian(2), 1)
    value, derivative = r.Identity(), r.Derivative(0)
    space = scalar
    if mixed:
        space = r.MixedSpace({'a': scalar})
        value = r.FieldOperator({'a': [[value]]})
        derivative = r.FieldOperator({'a': [[derivative]]})
    source = {'values': r.Samples(points, value)}
    approximation = r.LocalApproximation(
        source=source, trial=space.representers(source),
        stencil_policy=r.StencilPolicy(scaling='local'),
        backend=r.PythonBackend(compute_condition=False))
    maps = approximation.operators(targets=points[[2]], operators={'dx': derivative})
    algebra = maps.dx.reconstruct_local(0)
    # Single-field scaling is centered on the first source; mixed scaling is
    # centered on the target. Neither convention is changed by this audit fix.
    assert float(algebra.kernel_scale) == pytest.approx(np.sqrt(5) if mixed else 2)
    values = 3*points[:, 0] - 2*points[:, 1] + 7
    np.testing.assert_allclose(np.asarray(maps.dx @ {'values': values}).reshape(-1), [3], atol=1e-12)
