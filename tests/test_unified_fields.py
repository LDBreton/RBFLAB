import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r


def test_symbolic_scalar_and_system_bind_generically():
    model = r.SymbolicSystem(2, fields=('temperature', 'concentration'))
    a, b = model.fields
    space = r.MixedSpace({a: r.ScalarSpace(r.Gaussian(1), 2),
                          b: r.ScalarSpace(r.Gaussian(2), 2)})
    assert space.fields == model.fields
    points = r.geometry.unit_box_grid(2).points
    source = {'values': r.Samples(points, operator=r.vector(a, b))}
    maps = r.LocalApproximation(source=source, trial=space.representers(source)).operators(
        targets=points, operators={'L': r.laplacian(a)+r.dx(b),
                                  'symbolic': model.laplacian(a)+sp.diff(b, model.coordinates[0])})
    np.testing.assert_array_equal(maps.L.to_scipy().toarray(), maps.symbolic.to_scipy().toarray())
    x, y = points.T
    np.testing.assert_allclose(maps.L @ {'values': np.c_[x*x+y, 3*x-y*y]}, 5, atol=1e-12)
    scalar = r.SymbolicScalar(2)
    one = r.MixedSpace({scalar.field: r.ScalarSpace(r.Gaussian(1), 1)})
    assert one.fields == (scalar.field,)


def test_symbolic_local_binding_rejects_invalid_functionals():
    model = r.SymbolicStokes(2, transient=True)
    u, p = model.fields
    space = r.MixedSpace({u: r.DivergenceFreeSpace(r.Gaussian(1), 2),
                          p: r.PressureSpace(r.Gaussian(1), 1)})
    points = r.geometry.unit_box_grid(2).points
    source = {'u': r.Samples(points, operator=u),
              'p': r.Samples(points, operator=r.gradient(p))}
    local = r.LocalApproximation(source=source, trial=space.representers(source))
    for bad in (p*p, sp.diff(p, model.time), p+1, sp.Function('unknown')(*model.coordinates)):
        with pytest.raises((ValueError, TypeError)):
            local.operators(targets=points[[4]], operators={'bad': bad})
    with pytest.raises(ValueError, match='dimensions'):
        space._symbolic_operator(p, 3)
    with pytest.raises(ValueError, match='shape'):
        r.MixedSpace({u: r.ScalarSpace(r.Gaussian(1), 1)})


def test_unified_three_dimensional_components_and_symbolic_addition():
    model = r.SymbolicStokes(3, transient=True)
    u, p = model.fields
    space = r.MixedSpace({u: r.DivergenceFreeSpace(r.Gaussian(1), 2),
                          p: r.PressureSpace(r.Gaussian(1), 1)})
    assert space.fields[0] is u
    expression = u + r.laplacian(u) + r.gradient(p)
    symbolic = u + model.laplacian(u) + model.gradient(p)
    assert expression.compile(3, space._symbolic_operator) == space._symbolic_operator(symbolic, 3)
    assert r.vector(u[0], u[1], u[2]).compile(3, space._symbolic_operator).arity == 3


@pytest.mark.parametrize('backend', ['python', 'cpp', 'mpi'])
def test_unified_37_site_weights_and_values_match_old_fields(backend):
    if backend != 'python' and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native integration')
    if backend == 'mpi' and os.environ.get('RBFLAB_MPI_TESTS') != '1':
        pytest.skip('Opt-in real MPI integration')
    from test_shared_stencil_stokes import build
    old, old_maps = build(backend, True)
    new, new_maps = build(backend, True, symbolic=True)
    for actual, expected in zip(new, old):
        np.testing.assert_array_equal(actual, expected)
    data = {name: np.tile([1., -1.], (len(group['points']), 1))
            for name, group in old_maps.L.input_layout.items()}
    for name in ('L', 'P'):
        np.testing.assert_array_equal(np.asarray((new_maps[name] @ data).tolist(), float),
                                      np.asarray((old_maps[name] @ data).tolist(), float))
