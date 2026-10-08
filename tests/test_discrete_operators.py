"""Reusable rectangular RBF-FD and explicit approximation spaces."""
import os
import numpy as np
import pytest
import rbflab as r


def backend(name):
    if name=='python':return r.PythonBackend(compute_condition=False)
    if name=='torch':
        pytest.importorskip('torch')
        return r.TorchBackend(threads=2,batch_size=3,compute_condition=False)
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':pytest.skip('Opt-in native integration')
    return r.CppBackend(threads=2,compute_condition=False,cache_dir='outputs/operators_test_cache')


def points(d):
    rng=np.random.default_rng(37)
    return rng.uniform(-1,1,(24 if d==2 else 35,d)),rng.uniform(-.4,.4,(4,d))


@pytest.mark.parametrize('name',['python','torch','cpp'])
@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('vector',[False,True])
def test_polynomial_rectangular_and_local(name,dimension,vector):
    p,q=points(dimension)
    descriptor=(r.DivergenceFreeSpace if vector else r.ScalarSpace)(r.IMQ(2),2)
    method=r.RBFFD(spaces={'u':descriptor},stencil_size=18 if dimension==2 else 25,local_backend=backend(name))
    ops=method.operators(source=p,targets=q,operators={'value':r.Identity(dimension),'dx':r.Derivative(0,dimension),'lap':r.Laplacian(dimension)})
    assert ops.dx is ops['dx'];assert ops.diagnostics['factorizations']==len(q)
    assert ops.diagnostics['rhs_per_stencil']==3*(dimension if vector else 1)
    if vector:
        u=np.zeros_like(p);u[:,0]=p[:,1]**2;u[:,1]=p[:,0]**2
        exact=np.zeros_like(q);exact[:,0]=q[:,1]**2;exact[:,1]=q[:,0]**2
        dx=np.zeros_like(q);dx[:,1]=2*q[:,0]
        lap=np.zeros_like(q);lap[:,:2]=2
    else:u=(p*p).sum(1);exact=(q*q).sum(1);dx=2*q[:,0];lap=np.full(len(q),2*dimension)
    np.testing.assert_allclose(ops.value@u,exact,atol=2e-9)
    np.testing.assert_allclose(ops.dx@u,dx,atol=2e-9)
    np.testing.assert_allclose(ops.lap@u,lap,atol=2e-8)
    local=ops.dx.local(0);rebuild=ops.dx.reconstruct_local(0)
    w=np.vstack((local.weights,local.multipliers))
    np.testing.assert_allclose(rebuild.matrix.T@w,rebuild.rhs,atol=2e-9)
    # No saved factor/local matrix; mutating user geometry does not change reconstruction.
    p[:]=99;q[:]=88
    np.testing.assert_array_equal(rebuild.matrix,ops.dx.reconstruct_local(0).matrix)


@pytest.mark.parametrize('name',['python','torch','cpp'])
@pytest.mark.parametrize('dimension',[2,3])
def test_solenoidal_for_arbitrary_data(name,dimension):
    p,q=points(dimension)
    method=r.RBFFD(spaces={'U':r.DivergenceFreeSpace(r.IMQ(2),1)},stencil_size=12,local_backend=backend(name),stencil_policy=r.StencilPolicy(scaling='local'))
    ops=method.operators(source=p,targets=q,operators={f'd{j}':r.Derivative(j,dimension) for j in range(dimension)})
    data=np.random.default_rng(12).normal(size=p.shape)
    divergence=sum((ops[f'd{j}']@data)[:,j] for j in range(dimension))
    np.testing.assert_allclose(divergence,0,atol=2e-10)


@pytest.mark.parametrize('name',['python','cpp'])
@pytest.mark.parametrize('vector',[False,True])
def test_extended_weights_and_reconstruction(name,vector):
    p,q=points(2)
    descriptor=(r.DivergenceFreeSpace if vector else r.PressureSpace)(r.IMQ(2),1)
    ops=r.RBFFD(spaces={'field':descriptor},stencil_size=9,precision=r.Precision(local_digits=40,global_dtype='mpmath'),local_backend=backend(name)).operators(source=p,targets=q,operators={'dx':r.Derivative(0)})
    matrix=ops.dx.matrix;ctx=matrix.ctx
    values=ctx.matrix([[ctx.mpf(float(v)) for v in row] for row in p]) if vector else ctx.matrix([ctx.mpf(float(v)) for v in p[:,0]])
    if vector:
        # u=(x,-y), divergence-free linear polynomial
        values[:,1]=-values[:,1]
        result=ops.dx@values
        assert max(abs(result[i,0]-1) for i in range(len(q)))<ctx.mpf('1e-30')
        assert max(abs(result[i,1]) for i in range(len(q)))<ctx.mpf('1e-30')
    else:assert max(abs(v-1) for v in ops.dx@values)<ctx.mpf('1e-30')
    local=ops.dx.local(0);state=ops.dx.reconstruct_local(0)
    W=ctx.matrix(local.weights.rows+local.multipliers.rows,local.weights.cols)
    W[:local.weights.rows,:]=local.weights;W[local.weights.rows:,:]=local.multipliers
    assert ctx.mnorm(state.matrix.T*W-state.rhs,'inf')<ctx.mpf('1e-30')
    assert matrix.shape==(len(q)*(2 if vector else 1),len(p)*(2 if vector else 1))


@pytest.mark.parametrize('name',['torch','cpp'])
@pytest.mark.parametrize('vector',[False,True])
def test_nonpolynomial_backend_parity(name,vector):
    p,q=points(2);space=(r.DivergenceFreeSpace if vector else r.ScalarSpace)(r.IMQ(3),1)
    kwargs=dict(source=p,targets=q,operators={'dx':r.Derivative(0),'lap':r.Laplacian()})
    reference=r.RBFFD(spaces={'u':space},stencil_size=10,local_backend=backend('python')).operators(**kwargs)
    other=r.RBFFD(spaces={'u':space},stencil_size=10,local_backend=backend(name)).operators(**kwargs)
    for key in reference:np.testing.assert_allclose(reference[key].matrix.toarray(),other[key].matrix.toarray(),atol=3e-9,rtol=3e-9)


def test_pressure_constants_selection_and_rejection():
    p,q=points(2)
    method=r.RBFFD(spaces={'p':r.PressureSpace(r.IMQ(2)),'U':r.DivergenceFreeSpace(r.IMQ(2),1)},stencil_size=10)
    with pytest.raises(ValueError,match='Select a field'):method.operators(source=p,operators={'dx':r.Derivative(0)})
    ops=method.operators(source=p,targets=q,operators={'dx':r.Derivative(0)},space='p')
    np.testing.assert_allclose(ops.dx@np.ones(len(p)),0,atol=1e-12)
    with pytest.raises(ValueError,match='dimension'):method.operators(source=p,operators={'dx':r.Derivative(0,3)},space='p')
    with pytest.raises(ValueError,match='match source'):ops.dx@np.ones(5)
    with pytest.raises(NotImplementedError):r.RBFFD(r.IMQ(),scheme='symmetric').operators(source=p,operators={'dx':r.Derivative(0)})


def test_shared_factorization(monkeypatch):
    from rbflab.nodal import Arithmetic
    original=Arithmetic.factor;calls=[]
    def factor(self,*args,**kwargs):calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(Arithmetic,'factor',factor)
    p,q=points(2)
    r.RBFFD(r.IMQ(2),12).operators(source=p,targets=q,operators={'value':r.Identity(),'dx':r.Derivative(0),'dy':r.Derivative(1)})
    assert len(calls)==len(q)


def test_scalar_space_assemble_matches_kernel_api():
    cloud=r.unit_box_grid(2)
    problem=r.LinearPDE(-r.Laplacian(),1,[r.Dirichlet(0)])
    reference=r.RBFFD(r.IMQ(2),stencil_size=9,polynomial_degree=1).assemble(problem,cloud)
    mapped=r.RBFFD(spaces={'u':r.ScalarSpace(r.IMQ(2),1)},stencil_size=9).assemble(problem,cloud)
    np.testing.assert_array_equal(reference.matrix.toarray(),mapped.matrix.toarray())
    np.testing.assert_array_equal(reference.rhs,mapped.rhs)


def test_existing_weights_and_scaled_equation():
    from rbflab.nodal import rbf_fd_weights
    p,q=points(2)
    ops=r.RBFFD(r.IMQ(2),12,stencil_policy=r.StencilPolicy(scaling='local')).operators(source=p,targets=q,operators={'dx':r.Derivative(0)})
    row=ops.dx.local(0);state=ops.dx.reconstruct_local(0)
    w=rbf_fd_weights(r.IMQ(2),p[row.indices],q[0],r.Derivative(0),kernel_scaling='local')
    np.testing.assert_allclose(row.weights[:,0],w,atol=1e-12)
    augmented=np.vstack((row.weights,row.multipliers))
    np.testing.assert_allclose(state.scaled_matrix.T@(augmented/state.scale[:,None]),state.scaled_rhs,atol=2e-12)


@pytest.mark.parametrize('name',['python','cpp'])
def test_three_dimensional_extended_vector(name):
    p,q=points(3)
    ops=r.RBFFD(spaces={'u':r.DivergenceFreeSpace(r.IMQ(2),1)},stencil_size=9,precision=r.Precision(local_digits=35,global_dtype='mpmath'),local_backend=backend(name)).operators(source=p,targets=q[:1],operators={'dx':r.Derivative(0,3),'dy':r.Derivative(1,3),'dz':r.Derivative(2,3)})
    data=np.random.default_rng(8).normal(size=p.shape)
    div=(ops.dx@data)[0,0]+(ops.dy@data)[0,1]+(ops.dz@data)[0,2]
    assert abs(div)<ops.dx.matrix.ctx.mpf('1e-28')


def test_mp_rectangular_sparse_matmul_and_lu_guard():
    from rbflab.sparse_precision import MPSparseMatrix,MPSparseLU
    import mpmath
    ctx=mpmath.mp.clone();ctx.dps=40
    A=MPSparseMatrix(ctx,[{0:'1',2:'2'},{1:'3'}],ncols=3)
    assert list(A@['1','2','3'])==[7,6]
    assert (A@[['1','2'],['3','4'],['5','6']]).tolist()==[[11,14],[9,12]]
    with pytest.raises(ValueError,match='square'):MPSparseLU(A)
    with pytest.raises(ValueError,match='dimension'):A@[1,2]


def test_python_workers():
    p,q=points(2);kwargs=dict(source=p,targets=q[:2],operators={'dx':r.Derivative(0),'lap':r.Laplacian()})
    serial=r.RBFFD(r.IMQ(2),10,local_backend=r.PythonBackend(compute_condition=False)).operators(**kwargs)
    parallel=r.RBFFD(r.IMQ(2),10,local_backend=r.PythonBackend(workers=2,compute_condition=False)).operators(**kwargs)
    for name in serial:np.testing.assert_allclose(serial[name].matrix.toarray(),parallel[name].matrix.toarray(),atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('name',['python','torch','cpp'])
@pytest.mark.parametrize('kind',['phs','symbolic'])
def test_kernel_families_and_coincident_targets(name,kind):
    import sympy as sp
    p,_=points(2);p=p[:12]
    if kind=='phs':space=r.ScalarSpace(r.PHS(7),3)
    else:
        s=sp.Symbol('s',nonnegative=True);c=sp.Symbol('c',positive=True)
        space=r.DivergenceFreeSpace(r.Kernel(sp.exp(-c*s),s,(c,))(c='2'),1)
    method=r.RBFFD(spaces={'u':space},stencil_size=12,local_backend=backend(name))
    ops=method.operators(source=p,targets=p[:2],operators={'value':r.Identity(),'dx':r.Derivative(0)})
    data=np.random.default_rng(2).normal(size=p.shape if kind=='symbolic' else len(p))
    np.testing.assert_allclose(ops.value@data,data[:2],atol=2e-10)
    reference=r.RBFFD(spaces={'u':space},stencil_size=12,local_backend=backend('python')).operators(source=p,targets=p[:2],operators={'dx':r.Derivative(0)})
    np.testing.assert_allclose(ops.dx.matrix.toarray(),reference.dx.matrix.toarray(),atol=2e-9,rtol=2e-9)
