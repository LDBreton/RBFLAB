"""Cross-backend functional algebra, not a duplicate of frontend API tests."""
from types import SimpleNamespace
import os
import numpy as np
import pytest
import rbflab as r
from rbflab.discrete_operators import _Basis
from rbflab.functional_backends import execute, reconstruct


def backend(name):
    if name=='python':
        return r.PythonBackend(compute_condition=False)
    if name=='torch':
        pytest.importorskip('torch')
        return r.TorchBackend(threads=2,batch_size=2,compute_condition=False)
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':
        pytest.skip('Opt-in native integration')
    return r.CppBackend(threads=2,compute_condition=False,cache_dir='outputs/functional_test_cache')


def owner(name='python', digits=None, vector=False, dimension=2, independent=False):
    rng=np.random.default_rng(941)
    n=12 if dimension==2 else 16
    X=rng.uniform(-1,1,(n,dimension))
    Z=X+rng.normal(0,.015,X.shape) if independent else X.copy()
    kernel=r.IMQ(2)
    basis=_Basis(kernel,digits,X,1,vector,'local')
    I=r.Identity(dimension);dx=r.Derivative(0,dimension);L=r.Laplacian(dimension)
    source=[I]*(n-4)+[dx]*2+[L]*2
    trial=([I]*n if independent else source.copy())
    Y=rng.uniform(-.2,.2,(3,dimension));target=[I,dx,L]
    job=dict(source_points=X,source_ops=source,trial_points=Z,trial_ops=trial,
        target_points=Y,target_ops=target,top=basis.polynomial(X,source),
        bottom=basis.polynomial(Z,trial),target_poly=basis.polynomial(Y,target),
        scale=basis.h,size=basis.n,polynomials=basis.m)
    return SimpleNamespace(kernel=kernel,precision=r.Precision(local_digits=digits),
        backend=backend(name),arithmetic=basis.a,dimension=dimension,
        components=dimension if vector else 1,vector=vector,jobs=[job])


def numpy(value):
    if hasattr(value,'detach'):return value.detach().numpy()
    return np.asarray(value.tolist() if hasattr(value,'rows') else value,float)


@pytest.mark.parametrize('name',['python','torch','cpp'])
@pytest.mark.parametrize('vector',[False,True])
@pytest.mark.parametrize('independent',[False,True])
def test_functional_reproduction_and_two_sided_sign(name,vector,independent):
    o=owner(name,vector=vector,independent=independent)
    answer=execute(o)[0];G,Q=reconstruct(o,0);W=answer['weights'];j=o.jobs[0]
    G,Q,W=map(numpy,(G,Q,W))
    np.testing.assert_allclose(G.T@W,Q,atol=2e-9,rtol=2e-9)
    # Every polynomial in the constrained space is reproduced from FUNCTIONAL
    # data, not by mistakenly treating derivative data as point values.
    np.testing.assert_allclose(W[:j['size']].T@numpy(j['top']),numpy(j['target_poly']),atol=2e-9)
    ref=owner('python',vector=vector,independent=independent)
    G0,Q0=reconstruct(ref,0)
    np.testing.assert_allclose(G,numpy(G0),atol=2e-9,rtol=2e-12)
    np.testing.assert_allclose(Q,numpy(Q0),atol=2e-9,rtol=2e-12)
    if not independent:
        np.testing.assert_allclose(G,G.T,atol=2e-10)
    else:
        assert np.max(np.abs(G-G.T))>1e-3


@pytest.mark.parametrize('name',['python','cpp'])
@pytest.mark.parametrize('vector',[False,True])
def test_functional_extended_precision(name,vector):
    o=owner(name,digits=45,vector=vector)
    result=execute(o)[0];G,Q=reconstruct(o,0);W=result['weights'];ctx=o.arithmetic.ctx
    assert ctx.mnorm(G.T*W-Q,'inf')<ctx.mpf('1e-34')
    j=o.jobs[0]
    assert ctx.mnorm(W[:j['size'],:].T*j['top']-j['target_poly'],'inf')<ctx.mpf('1e-34')


def test_torch_weights_retain_graph_and_multi_rhs():
    torch=pytest.importorskip('torch')
    o=owner('torch',independent=True)
    targets=torch.tensor(o.jobs[0]['target_points'],dtype=torch.float64,requires_grad=True)
    o.jobs[0]['target_points']=targets
    result=execute(o)[0]
    assert isinstance(result['weights'],torch.Tensor)
    assert result['weights'].shape[1]==3
    loss=result['weights'].square().sum()
    loss.backward()
    assert torch.isfinite(targets.grad).all()
    assert targets.grad.abs().max()>0
    G,Q=reconstruct(o,0)
    assert isinstance(G,torch.Tensor) and isinstance(Q,torch.Tensor)


def test_three_dimensional_functionals():
    o=owner(dimension=3,vector=True)
    result=execute(o)[0];G,Q=reconstruct(o,0)
    np.testing.assert_allclose(G.T@result['weights'],Q,atol=2e-9)


def test_python_worker_parity():
    o=owner()
    expected=execute(o)[0]['weights']
    o.backend=r.PythonBackend(workers=2,compute_condition=False)
    np.testing.assert_allclose(execute(o)[0]['weights'],expected,atol=1e-12)

@pytest.mark.parametrize('name,digits', [('python',None),('torch',None),('cpp',None),('cpp',45)])
def test_frontend_mixed_vector_functional_blocks(name,digits):
    """The full public API retains the scalar/vector native job semantics."""
    rng=np.random.default_rng(413)
    X=rng.uniform(-1,1,(16,2));Z=rng.uniform(-.8,.8,(5,2));F=rng.uniform(-.7,.7,(4,2))
    Y=rng.uniform(-.4,.4,(3,2))
    source={
        'u':r.Samples(X,size=10),
        'dx':r.Samples(Z,operator=r.Derivative(0,2),size=3),
        'pde':r.Samples(F,operator=r.Laplacian(2),size=3),
    }
    space=r.DivergenceFreeSpace(r.IMQ(2),polynomial_degree=1)
    method=r.LocalApproximation(source=source,trial=space.representers(source),
        backend=backend(name),precision=r.Precision(local_digits=digits,global_dtype='mpmath' if digits else 'float64'),
        stencil_policy=r.StencilPolicy(scaling='local'))
    ops=method.operators(targets=Y,operators={'value':r.Identity(2),'dx':r.Derivative(0,2),'lap':r.Laplacian(2)})
    data={'u':np.column_stack([X[:,0],-X[:,1]]),
          'dx':np.tile([1.,0.],(len(Z),1)),'pde':np.zeros_like(F)}
    actual=ops.value@data
    expected=np.column_stack([Y[:,0],-Y[:,1]])
    if digits:
        ctx=actual.ctx
        assert max(abs(actual[i,j]-ctx.mpf(float(expected[i,j]))) for i in range(len(Y)) for j in range(2))<ctx.mpf('1e-32')
    else:
        np.testing.assert_allclose(numpy(actual),expected,atol=3e-9)
    direct=sum((ops.value[key]@values for key,values in data.items()))
    np.testing.assert_allclose(numpy(direct),numpy(actual),atol=1e-12)
    local=ops.dx.local(0);dense=ops.dx.reconstruct_local(0)
    assert set(local.groups)==set(source)
    if digits:
        ctx=actual.ctx
        W=ctx.matrix(local.weights.rows+local.multipliers.rows,2)
        W[:local.weights.rows,:]=local.weights;W[local.weights.rows:,:]=local.multipliers
        assert ctx.mnorm(dense.matrix.T*W-dense.rhs,'inf')<ctx.mpf('1e-31')
    else:
        np.testing.assert_allclose(numpy(dense.matrix).T@np.vstack([numpy(local.weights),numpy(local.multipliers)]),numpy(dense.rhs),atol=3e-9)
    if name=='torch':
        import torch
        assert isinstance(ops.value.matrix,torch.Tensor)
        assert ops.value.matrix.is_sparse
    assert ops.value['u'].to_scipy().shape==(len(Y)*2,len(X)*2)
