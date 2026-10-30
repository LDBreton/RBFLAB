"""Small algebra checks independent of the saved cavity experiment."""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix

from research.navier_stokes_schemes import nullspace_lifting as lifting


def test_scaled_common_nullspace_change_preserves_all_polynomials(monkeypatch):
    rng = np.random.default_rng(128)
    C = rng.standard_normal((75, 40))*np.geomspace(1., 1e4, 40)
    W = rng.standard_normal((8, 75))
    T = rng.standard_normal((8, 40))
    data = dict(C=C, W=W, T=T, R=T-W@C,
                D=np.where(np.arange(75) % 2, 2., 1.),
                momentum=np.arange(75) % 2 == 1, r2=np.zeros(75))
    monkeypatch.setattr(lifting, 'prepare', lambda p: ([data], {}))
    model = lifting.NullLifting(SimpleNamespace(U0=np.zeros(2)))
    parameters = rng.uniform(-.1, .1, model.size)
    V, E = model.local(parameters, 0)
    # The new field retains every old observation moment. Checking only the
    # five pressure controls would miss high-degree velocity contamination.
    np.testing.assert_allclose(E@C, np.eye(40), atol=2e-9, rtol=0)
    np.testing.assert_allclose((V@C-T)/np.linalg.norm(C, axis=0), 0., atol=2e-10, rtol=0)
    assert np.linalg.norm(E-data['E0']) > 1e-3


def test_index_one_reduction_retains_finite_modes_and_mass_condition():
    p = SimpleNamespace(M=csr_matrix(np.diag([2., 3., 0., 0., 0., 0., 0.])),
                        A=csr_matrix(np.diag([4., 9., 1., 1., 1., 1., 1.])))
    result = lifting.descriptor_audit(p)
    assert result['consistent_dimension'] == 2
    assert result['reduced_positive_count'] == 0
    np.testing.assert_allclose(result['reduced_spectral_abscissa'], -2.)
    np.testing.assert_allclose(result['min_nonzero_mass_singular_value'], 2.)
    np.testing.assert_allclose(result['reduced_mass_condition'], 1.5)


def test_full_coupled_pencil_matches_rest_descriptor_at_zero_flow():
    from research.navier_stokes_schemes.cavity_nullspace_lifting import coupled_spectrum
    p = SimpleNamespace(M=csr_matrix(np.diag([2., 3., 4., 0., 0., 0., 0., 0.])),
                        A=csr_matrix(np.diag([4., 9., 16., 1., 1., 1., 1., 1.])),
                        boundary=lambda t: np.zeros(2))
    for name in ('DxU', 'DxL', 'DyU', 'DyL'):
        setattr(p, name, csr_matrix((8, 8)))
    for name in ('DxB', 'DyB'):
        setattr(p, name, csr_matrix((8, 2)))
    report = coupled_spectrum(p, np.zeros(8), np.zeros(8))
    assert report['resolved_count'] == 3
    assert report['positive_count'] == 0
    np.testing.assert_allclose(report['real'], -2.)
