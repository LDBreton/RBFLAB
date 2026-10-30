"""Public examples reproduce fields and use the documented local maps."""
import os

import numpy as np
import pytest

from examples import mixed_local_maps, stokes_lhi_weights


def test_mixed_example_reconstructs_fields_and_derivatives():
    errors = mixed_local_maps.run()
    assert errors['field_max_absolute_nodal_error'] < 1e-10
    assert errors['equation_max_absolute_nodal_error'] < 1e-10


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1', reason='Opt-in real MPI example')
def test_stokes_weights_example_reproduces_affine_pressure():
    maps = stokes_lhi_weights.run(backend='cpp', digits=80, ranks=2)
    points = maps.gradient.input_layout['velocity']['points']
    data = {'velocity': np.c_[points[:, 1], -points[:, 0]],
            'momentum': np.tile([1., -1.], (len(points), 1))}
    np.testing.assert_allclose(maps.gradient @ data, [[1., -1.]], atol=1e-12)
    np.testing.assert_allclose(maps.momentum @ data, [[1., -1.]], atol=1e-12)
    assert maps.diagnostics['execution']['scheduler'] == 'mpi'


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1', reason='Opt-in real MPI example')
def test_navier_stokes_assembly_pressure_and_local_divergence():
    from examples.navier_stokes_lhi import assemble, evolve
    points, boundary, M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb = assemble(cells=6, nu=1., ranks=2)
    gradient = np.tile([1., -1.], len(points))
    # u=0, p=x-y: constant pressure forcing should not generate velocity.
    np.testing.assert_allclose(M @ gradient, 0, atol=1e-11)
    np.testing.assert_allclose(PL @ gradient, gradient, atol=1e-11)
    # Each local reconstruction cancels its own divergence at its center.
    divergence = (Dx[0::2] + Dy[1::2]).toarray()
    np.testing.assert_allclose(divergence, 0, atol=1e-11)
    for dt in (0.00025, 0.000125):
        U, G = evolve(M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb, np.zeros_like(gradient),
                      lambda t: gradient, lambda t: np.zeros(2*len(boundary)), dt, 4)
        np.testing.assert_allclose(U, 0, atol=1e-9)
        np.testing.assert_allclose(G, gradient, atol=1e-8)
        # Spatially constant u=(1+t)*(1,-1) has zero convection and viscosity.
        # This exercises boundary contributions in momentum and derivatives.
        g0 = np.tile([1., -1.], len(boundary))
        np.testing.assert_allclose(Dx @ gradient + Dxb @ g0, 0, atol=1e-11)
        np.testing.assert_allclose(Dy @ gradient + Dyb @ g0, 0, atol=1e-11)
        U, G = evolve(M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb, gradient,
                      lambda t: 2*gradient, lambda t: (1+t)*g0, dt, 4)
        np.testing.assert_allclose(U, (1+4*dt)*gradient, atol=1e-9)
        np.testing.assert_allclose(G, gradient, atol=1e-8)
