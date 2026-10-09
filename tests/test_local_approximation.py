"""Algebraic equivalence and researcher-facing contracts of the shared core."""
import numpy as np
import pytest
from scipy.sparse import eye
from scipy.sparse.linalg import splu
import rbflab as r


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('vector', [False, True])
def test_standard_matches_existing_weights(dimension, vector):
    rng=np.random.default_rng(173)
    x=rng.uniform(-1,1,(22,dimension)); y=rng.uniform(-.5,.5,(3,dimension))
    cls=r.DivergenceFreeSpace if vector else r.ScalarSpace
    v=cls(r.IMQ(2), 1)
    policy=r.StencilPolicy(scaling='local')
    args=dict(targets=y,operators={'value':r.Identity(dimension),'dx':r.Derivative(0,dimension),'lap':r.Laplacian(dimension)})
    old=r.RBFFD(spaces={'u':v},stencil_size=12,stencil_policy=policy).operators(source=x,**args)
    new=r.LocalApproximation(source=x,space=v,stencil_size=12,stencil_policy=policy).operators(**args)
    for name in old:
        np.testing.assert_allclose(new[name].matrix.toarray(), old[name].matrix.toarray(),atol=2e-9,rtol=2e-9)
        state=new[name].reconstruct_local(0); local=new[name].local(0)
        w=np.vstack((local.weights,local.multipliers))
        np.testing.assert_allclose(state.matrix.T@w,state.rhs,atol=1e-9)


def legacy_recipe(kernel=None, degree=2, precision=None):
    cloud=r.geometry.unit_box_grid(4)
    L=-r.Laplacian()
    problem=r.LinearPDE(L,-4,[r.Dirichlet(lambda x:1+(x*x).sum(1))])
    legacy=r.LHI(kernel or r.PHS(5),stencil_size=12,pde_stencil_size=5,
                 polynomial_degree=degree,precision=precision or r.Precision()).assemble(problem,cloud)
    ii=cloud.interior_indices; bi=cloud.boundary_indices
    umap={j:i for i,j in enumerate(ii)}; bmap={j:i for i,j in enumerate(bi)}
    source={
        'u':r.Samples(cloud.interior,indices=[[umap[j] for j in s.solution_indices] for s in legacy.stencils]),
        'boundary':r.Samples(cloud.points[bi],indices=[[bmap[j] for j in s.boundary_indices] for s in legacy.stencils]),
        'pde':r.Samples(cloud.interior,L,target='exclude',indices=[[umap[j] for j in s.pde_indices] for s in legacy.stencils]),
    }
    v=r.ScalarSpace(kernel or r.PHS(5),degree)
    local=r.LocalApproximation(source=source,trial=v.representers(source),precision=precision or r.Precision())
    return cloud,problem,legacy,local.operators(targets=cloud.interior,operators={'L':L})


def test_lhi_matrix_and_reconstruction_agree_with_legacy():
    cloud,problem,legacy,ops=legacy_recipe()
    np.testing.assert_allclose(ops.L['u'].matrix.toarray(),legacy.matrix.toarray(),atol=2e-10,rtol=2e-11)
    data={'u':1+(cloud.interior**2).sum(1),'boundary':1+(cloud.points[cloud.boundary_indices]**2).sum(1),'pde':np.full(len(cloud.interior),-4.)}
    np.testing.assert_allclose(ops.L@data,-4,atol=3e-11)
    rhs=-4-ops.L['pde']@data['pde']-ops.L['boundary']@data['boundary']
    np.testing.assert_allclose(rhs,legacy.rhs,atol=2e-10)
    for i,s in enumerate(legacy.stencils):
        np.testing.assert_allclose(ops.L.local(i).weights[:,0],s.weights,atol=2e-10,rtol=2e-11)
        A=ops.L.reconstruct_local(i).matrix
        np.testing.assert_allclose(A,A.T,atol=1e-12)


def test_manual_heat_mass_matches_evolution_assembly():
    cloud,problem,legacy,ops=legacy_recipe()
    evolution=r.EvolutionPDE(problem.operator,1,initial=0,boundary=[r.Dirichlet(0)])
    before=r.LHI(r.PHS(5),stencil_size=12,pde_stencil_size=5,polynomial_degree=2).assemble(evolution,cloud)
    M=eye(len(cloud.interior))-ops.L['pde'].matrix
    A=ops.L['u'].matrix
    np.testing.assert_allclose(M.toarray(),before.mass.toarray(),atol=2e-12)
    np.testing.assert_allclose(A.toarray(),before.spatial.toarray(),atol=2e-10)
    dt=.001; u=np.zeros(len(cloud.interior)); forcing=ops.L['pde']@np.ones(len(u))
    solve=splu((M/dt+A).tocsc()).solve
    for _ in range(3):u=solve(M@u/dt+1-forcing)
    trajectory=before.solve(dt,3,scheme='backward_euler')
    np.testing.assert_allclose(u,trajectory.final.unknowns,atol=2e-10)


@pytest.mark.parametrize('digits',[None,40])
def test_independent_functional_centers_and_transformed_trial(digits):
    rng=np.random.default_rng(53)
    x=rng.uniform(-1,1,(10,2)); z=rng.uniform(-1,1,(6,2)); b=rng.uniform(-1,1,(5,2))
    source={'u':r.Samples(x),'dx':r.Samples(z,r.Derivative(0)),'boundary':r.Samples(b,r.Identity())}
    v=r.ScalarSpace(r.IMQ(2),2)
    prec=r.Precision(local_digits=digits,global_dtype='mpmath' if digits else 'float64')
    ops=r.LocalApproximation(source=source,trial=v.representers(source),precision=prec).operators(targets=np.array([[.13,.24]]),operators={'L':r.Laplacian(),'value':r.Identity()})
    data={'u':1+(x*x).sum(1),'dx':2*z[:,0],'boundary':1+(b*b).sum(1)}
    assert abs(float((ops.L@data)[0])-4)<2e-9
    # Data and trial need not coincide. A first-order shifted functional
    # preserves the constant polynomial rather than annihilating it.
    trial={'u':r.Samples(x,r.Identity()+r.Derivative(0))}
    op=r.LocalApproximation(source={'u':r.Samples(x)},trial=v.representers(trial),precision=prec).operators(targets=x[:2],operators={'value':r.Identity()}).value
    np.testing.assert_allclose(np.asarray(op@(1+(x*x).sum(1)),float).reshape(-1),1+(x[:2]*x[:2]).sum(1),atol=1e-8)
    state=op.reconstruct_local(0)
    A=np.array(state.matrix.tolist() if digits else state.matrix,float)
    assert np.max(abs(A-A.T))>.01


def test_torch_native_sparse_blocks_and_value_gradients():
    torch=pytest.importorskip('torch')
    x=r.geometry.unit_box_grid(3).points
    ops=r.LocalApproximation(source=x,space=r.ScalarSpace(r.PHS(5),2),backend=r.TorchBackend()).operators(targets=x[:3],operators={'value':r.Identity(),'L':r.Laplacian()})
    assert isinstance(ops.value.matrix,torch.Tensor)
    u=torch.tensor((x*x).sum(1),dtype=torch.float64,requires_grad=True)
    result=ops.value@{'u':u}
    result.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    np.testing.assert_allclose(ops.L['u'].to_scipy()@u.detach().numpy(),4,atol=2e-9)


def test_explicit_membership_normals_and_input_snapshot():
    x=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    samples=r.Samples(x,indices=[[0,1,2,3]])
    local=r.LocalApproximation(source={'u':samples},trial=r.ScalarSpace(r.IMQ(2),1).representers({'u':samples}))
    x[:]=99
    op=local.operators(targets=[[.3,.4]],operators={'dx':r.Derivative(0)}).dx
    np.testing.assert_allclose(op@{'u':np.array([0.,1.,0.,1.])},1,atol=1e-12)
    with pytest.raises(ValueError,match='exactly'):op@{'missing':np.zeros(4)}
    with pytest.raises(ValueError,match='explicit trial'):
        r.LocalApproximation(source={'d':r.Samples([[0,0],[1,1]],r.Derivative(0))},space=r.ScalarSpace(r.IMQ()))
    s=r.Samples([[0,0],[1,1]],size=1,target='exclude')
    np.testing.assert_array_equal(s.select(np.array([0,0]),0,r.StencilPolicy(),0),[1])
    with pytest.raises(ValueError,match='normals'):
        r.Samples([[0,0]],r.NormalDerivative())
    s=r.Samples([[0,0]],r.NormalDerivative(),normals=[[1,0]])
    assert s._operators[0]==r.Derivative(0)


def test_duplicate_functionals_and_rank_rejected():
    x=np.array([[0.,0.],[1.,0.],[0.,1.]])
    source={'u':r.Samples(x),'repeat':r.Samples(x)}
    with pytest.raises(ValueError,match='Duplicate'):
        r.LocalApproximation(source=source,trial=r.ScalarSpace(r.IMQ()).representers(source)).operators(targets=x[:1],operators={'value':r.Identity()})
    source={'du':r.Samples(x,r.Derivative(0))}
    with pytest.raises(np.linalg.LinAlgError,match='unisolvent'):
        r.LocalApproximation(source=source,trial=r.ScalarSpace(r.IMQ(),1).representers(source)).operators(targets=x[:1],operators={'value':r.Identity()})


def test_polynomial_smoothness_check_uses_both_arguments():
    x=r.geometry.unit_box_grid(3).points
    source={'u':r.Samples(x),'L':r.Samples(x,r.Laplacian(),target='exclude',size=3)}
    with pytest.raises(ValueError,match='order 4'):
        r.LocalApproximation(source=source,trial=r.ScalarSpace(r.PHS(3),1).representers(source)).operators(targets=x[:1],operators={'L':r.Laplacian()})



def test_extended_matrix_algebra_for_manual_time_step():
    from rbflab.sparse_precision import MPSparseMatrix, MPSparseLU
    import mpmath
    ctx=mpmath.mp.clone();ctx.dps=45
    Sf=MPSparseMatrix(ctx,[{0:'0.1',1:'0.02'},{0:'0.03',1:'0.2'}])
    M=MPSparseMatrix.eye(ctx,2)-Sf
    A=MPSparseMatrix(ctx,[{0:'2',1:'-1'},{0:'-1',1:'2'}])
    dt=ctx.mpf('0.001')
    K=M/dt+A
    u=ctx.matrix(['1','2']);rhs=M@u/dt
    nxt=MPSparseLU(K).solve(rhs)
    assert ctx.norm(K@nxt-rhs,'inf')<ctx.mpf('1e-38')
    product=A.T@M
    dense=lambda matrix:ctx.matrix([[row.get(j,ctx.zero) for j in range(matrix.shape[1])] for row in matrix.rows])
    assert ctx.mnorm(dense(product)-dense(A).T*dense(M),'inf')<ctx.mpf('1e-40')



def test_reused_neighbor_index_and_explicit_trial_membership(monkeypatch):
    import rbflab.samples as sampling
    x=r.geometry.unit_box_grid(3).points
    allowed=r.Samples(x,size=4)
    excluded=r.Samples(x,size=4,target='exclude')
    def unexpected(*args,**kwargs):
        raise AssertionError('Nearest sampling must reuse its spatial index')
    monkeypatch.setattr(sampling,'cKDTree',unexpected)
    for i in range(4):
        assert len(allowed.select(x[i],i,r.StencilPolicy(),0))==4
        assert i not in excluded.select(x[i],i,r.StencilPolicy(),0)
    monkeypatch.undo()
    source=r.Samples(x,size=4)
    trial=r.ScalarSpace(r.IMQ(2)).translates(r.Samples(x,indices=[[0,1,2]]))
    with pytest.raises(ValueError,match='equal nonzero'):
        r.LocalApproximation(source=source,trial=trial).operators(targets=x[:1],operators={'value':r.Identity()})


def test_normal_lists_inherit_cloud_geometry():
    from types import SimpleNamespace
    cloud=SimpleNamespace(points=np.array([[0.,0.],[1.,1.]]),normals=np.array([[1.,0.],[0.,1.]]))
    samples=r.Samples(cloud,operator=[r.Identity(),r.NormalDerivative()])
    assert samples._operators[1]==r.Derivative(1)
