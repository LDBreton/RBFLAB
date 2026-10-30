"""Driver-level persistence, affine pressure control and complex-mode checks."""
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,diags,eye
from research.navier_stokes_schemes import cavity_accuracy_verify as driver


def fixture_problem(n=4,affine=True):
    rng=np.random.default_rng(20261029)
    g=np.array([.2,-.1,1.]) if affine else np.zeros(2)
    sparse=lambda shape,scale:csr_matrix(scale*rng.normal(size=shape))
    p=SimpleNamespace(nu=.01,U0=np.zeros(n),g0=g,points=np.column_stack((np.linspace(.2,.8,n//2),np.full(n//2,.5))),
        A=diags(np.linspace(1.,1.4,n)),M=eye(n),B=sparse((n,len(g)),.12),
        PY=sparse((n,n),.2),PB=sparse((n,len(g)),.2),PL=eye(n),
        DxU=sparse((n,n),.2),DxL=sparse((n,n),.2),DxB=sparse((n,len(g)),.2),
        DyU=sparse((n,n),.2),DyL=sparse((n,n),.2),DyB=sparse((n,len(g)),.2),
        linear_solver='direct',global_ordering='COLAMD')
    p.boundary=lambda t:p.g0.copy();p.force=lambda t:np.zeros(n)
    return p


def test_driver_checkpoint_files_preserve_nonlinear_bdf_history(monkeypatch,tmp_path):
    monkeypatch.setattr(driver,'problem',lambda *args:fixture_problem())
    monkeypatch.setattr(driver,'reference_at_nodal_lines',lambda *args:{})
    U=np.array([.2,-.1,.3,.15]);p=fixture_problem();L=p.A@U+p.B@p.g0
    mode=np.r_[np.array([1.,-.3,.2,.4]),np.array([.6,-.1,.2,.3])].astype(complex)
    for name in ('whole','split'):
        out=tmp_path/name;out.mkdir();np.savez(out/'initial.npz',U=U,L=L,mode=mode,eigenvalue=-1+2j)
    options=dict(source=Path('unused'),cells=2,scheme='oseen_newton',dt=.02,control='perturb',end=.14,amplitude=.001)
    driver.advance(output=tmp_path/'whole',steps=7,**options)
    driver.advance(output=tmp_path/'split',steps=2,**options)
    driver.advance(output=tmp_path/'split',steps=5,**options)
    case='perturb_oseen_newton_0.02'
    a=np.load(tmp_path/'whole'/case/'checkpoint.npz');b=np.load(tmp_path/'split'/case/'checkpoint.npz')
    for key in ('U','L','Uold','Lold'):np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-14)
    assert int(b['next_step'])==8
    ds=json.loads((tmp_path/'split'/case/'diagnostics.json').read_text())
    assert [d['step'] for d in ds]==list(range(1,8))
    assert sum(d['newton_iterations'] for d in ds)>=7
    assert max(d['nonlinear_residual_inf'] for d in ds)<=2e-11


def test_driver_pressure_control_zeros_affine_datum_and_applies_force(monkeypatch,tmp_path):
    def pressure_problem(*args):
        p=fixture_problem();n=len(p.U0);p.M=csr_matrix((n,n));p.PY=eye(n)
        p.PL=eye(n)
        for d in ('Dx','Dy'):
            for group in ('U','L'):setattr(p,d+group,csr_matrix((n,n)))
            setattr(p,d+'B',csr_matrix((n,len(p.g0))))
        return p
    monkeypatch.setattr(driver,'problem',pressure_problem)
    n=4;np.savez(tmp_path/'initial.npz',U=np.zeros(n),L=np.zeros(n),mode=np.zeros(2*n),eigenvalue=-1.)
    driver.advance(Path('unused'),2,tmp_path,'oseen_newton',.01,'pressure',3,.03,.1)
    case=tmp_path/'pressure_oseen_newton_0.01'
    r=json.loads((case/'result.json').read_text());z=np.load(case/'fields.npz')
    expected=np.column_stack((pressure_problem().points[:,1],pressure_problem().points[:,0])).ravel()
    np.testing.assert_allclose(z['U'],0,atol=1e-14)
    np.testing.assert_allclose(z['L'],expected,atol=1e-14)
    assert r['pressure_gradient_error_inf']<1e-14


def test_prepare_preserves_complex_mode_phase_and_descriptor_eigenvector(monkeypatch,tmp_path):
    p=fixture_problem(n=8,affine=False)
    A=np.eye(8);A[:3,:3]=[[1,-2,0],[2,1,0],[0,0,2]];p.A=csr_matrix(A)
    p.M=diags([1.,1.,1.,0.,0.,0.,0.,0.])
    for d in ('Dx','Dy'):
        for group in ('U','L'):setattr(p,d+group,csr_matrix((8,8)))
        setattr(p,d+'B',csr_matrix((8,2)))
    monkeypatch.setattr(driver,'problem',lambda *args:p)
    monkeypatch.setattr(driver,'reference_at_nodal_lines',lambda *args:{})
    source=tmp_path/'source';source.mkdir();np.savez(source/'steady.npz',U=np.zeros(8),L=np.zeros(8))
    output=tmp_path/'verify';driver.prepare(source,2,output)
    z=np.load(output/'initial.npz');v=z['mode'];lam=complex(z['eigenvalue'])
    np.testing.assert_allclose(lam.real,-1.,atol=1e-13);np.testing.assert_allclose(abs(lam.imag),2.,atol=1e-13)
    np.testing.assert_allclose(np.max(abs(v[:8].real)),1.,atol=1e-14)
    pivot=np.argmax(abs(v[:8]));assert abs(v[pivot].imag)<1e-14 and v[pivot].real>0
    np.testing.assert_allclose(p.A@v[:8]-p.M@v[8:],0,atol=1e-13)
    np.testing.assert_allclose(v[8:]+lam*v[:8],0,atol=1e-13)
