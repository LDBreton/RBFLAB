"""Independent known descriptor check for the coupled cavity linearization."""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import diags,csr_matrix
from research.navier_stokes_schemes import nullspace_dynamics as module


def test_coupled_linearization_sign_and_descriptor_constraint(monkeypatch,tmp_path):
    # A u=M ell, u_t+JU u+ell=0. First component has ell=2u,
    # hence u_t=2u for JU=-4. Second component enforces u=ell=0.
    p=SimpleNamespace(A=diags([2.,3.]),M=diags([1.,0.]),boundary=lambda t:np.zeros(0))
    monkeypatch.setattr(module,'convection',lambda *args:(np.zeros(2),diags([-4.,5.]),csr_matrix((2,2))))
    result,mode,lam=module.linearized(p,np.zeros(2),np.zeros(2),tmp_path)
    assert result['resolved_count']==1
    assert result['positive_count']==1
    np.testing.assert_allclose(lam,2.,atol=1e-12)
    np.testing.assert_allclose(p.A@mode[:2]-p.M@mode[2:],0.,atol=1e-15)
    np.testing.assert_allclose(np.max(abs(mode[:2].real)),1e-7)
