"""Independent algebra checks for the bounded cavity candidate driver."""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, diags, eye

from research.navier_stokes_schemes.cavity_candidates import diagnose, steady
from research.divfree_lhi_energy.cavity_corner_field import evaluate


def test_steady_stokes_observation_sign_and_nonzero_wall_term():
    # With zero derivatives, N=0 and L=0. The first equation is A U=-B g.
    n = 4
    zero = csr_matrix((n, n))
    p = SimpleNamespace(
        U0=np.zeros(n), initial=lambda: (np.zeros(n), np.zeros(n)),
        boundary=lambda t: np.array([1., -2., 3., -4.]),
        A=diags([2., 3., 4., 5.]), M=eye(n), B=eye(n),
        DxU=zero, DxL=zero, DxB=zero, DyU=zero, DyL=zero, DyB=zero)
    report, u, ell = steady(p)
    assert report['status'] == 'converged'
    np.testing.assert_allclose(u, -p.boundary(0)/[2., 3., 4., 5.])
    np.testing.assert_allclose(ell, 0, atol=1e-14)


def test_generalized_stokes_growth_uses_negative_a_and_mass():
    p = SimpleNamespace(points=np.array([[.2, .3], [.4, .5]]),
                        A=diags([-6., 2., 3., 4.]),
                        M=diags([2., 1., 1., 1.]), PL=eye(4), PY=eye(4))
    mode = diagnose(p)['rest_stokes_spectrum']
    np.testing.assert_allclose(mode['real'], 3.)
    assert mode['positive_count'] == 1
    assert mode['relative_residual'] < 1e-14


def test_corner_jets_and_stokes_force_in_core_and_transition():
    # Check core and cutoff annulus at both corners using independent FD jets.
    points = np.array([[.03, .96], [.14, .91], [.97, .96], [.86, .91]])
    h = 2e-6
    field = evaluate(points)
    numerical = []
    for axis in (0, 1):
        shift = np.zeros_like(points)
        shift[:, axis] = h
        numerical.append((evaluate(points+shift)['velocity']-
                          evaluate(points-shift)['velocity'])/(2*h))
    np.testing.assert_allclose(field['dx'], numerical[0], rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(field['dy'], numerical[1], rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(field['dx'][:, 0]+field['dy'][:, 1], 0, atol=1e-14)
    np.testing.assert_allclose(field['stokes_force'],
                               -.01*field['lap']+field['gradpressure'], atol=1e-11)
