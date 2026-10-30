"""Actual 37-site (U,P,g,L) reconstruction and backend parity."""
import os
import numpy as np
import pytest
from examples.stokes_pressure_samples import build, pressure_test


@pytest.fixture(scope='module', params=[1, 2])
def python_maps(request):
    return request.param, build(pressure_degree=request.param, backend='python')


def test_pressure_gauge_and_quadratic_patch(python_maps):
    degree, (maps, inside, boundary, targets) = python_maps
    error = pressure_test(maps, inside, boundary, targets)
    shifted = pressure_test(maps, inside, boundary, targets, shift=3.)
    tolerance = 2e-6 if degree == 1 else 1e-10
    assert max(error.values()) < tolerance
    assert max(shifted.values()) < tolerance
    # The output pressure must not simply copy the target's pressure sample.
    matrix = maps.pressure['pressure'].to_scipy()
    for row, target in enumerate(targets):
        col = np.flatnonzero(np.all(inside == target, axis=1))[0]
        assert matrix[row, col] == 0


def test_nonzero_velocity_and_boundary_polynomial(python_maps):
    degree, (maps, inside, boundary, targets) = python_maps
    def velocity(nodes):
        x,y=nodes.T
        return np.column_stack((x*x, -2*x*y))
    x,y=inside.T
    data={'velocity':velocity(inside), 'pressure':x*y+3,
          'boundary':velocity(boundary), 'momentum':np.column_stack((y-.02,x))}
    x,y=targets.T
    exact={'velocity':velocity(targets),'pressure':(x*y+3)[:,None],
           'momentum':np.column_stack((y-.02,x)), 'gradient':np.column_stack((y,x)),
           'dx':np.column_stack((2*x,-2*y)), 'dy':np.column_stack((0*x,-2*x))}
    for name, expected in exact.items():
        actual=np.asarray((maps[name]@data).tolist(),dtype=float)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-6 if degree==1 else 1e-10)


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1' or
                    os.environ.get('RBFLAB_MPI_TESTS')!='1',reason='Opt-in C++/MPI weights')
@pytest.mark.parametrize('ranks',[0,2])
def test_all_weight_blocks_native_parity(python_maps,ranks):
    degree,(expected,_,_,_)=python_maps
    actual,inside,boundary,targets=build(pressure_degree=degree,backend='cpp',ranks=ranks)
    for name in ('velocity','pressure','momentum','gradient','dx','dy'):
        for group in ('velocity','pressure','boundary','momentum'):
            np.testing.assert_allclose(actual[name][group].to_scipy().toarray(),
                expected[name][group].to_scipy().toarray(),rtol=2e-12,atol=2e-11)
    assert max(pressure_test(actual,inside,boundary,targets).values()) < (2e-6 if degree==1 else 1e-10)
    if ranks:
        assert actual.diagnostics['execution']['scheduler']=='mpi'
