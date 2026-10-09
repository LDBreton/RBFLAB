"""Research API contracts, independent Hermite groups, and lifecycle regressions."""
import numpy as np
import pytest
import sympy as sp
import rbflab as r
from scipy.spatial import cKDTree


def problem_cloud(cells=3):
    cloud=r.geometry.unit_box_grid(cells)
    m=r.SymbolicScalar();u=m.field;x,y=m.coordinates
    exact=1+x*x+y*y
    problem=m.stationary(sp.Eq(-m.laplacian(u),-4),boundary=[m.bc('boundary',sp.Eq(u,exact))])
    return cloud,problem,m.data(exact)


def test_explicit_weights_order_and_shared_operator_algebra():
    rng=np.random.default_rng(76);p=rng.random((17,2));q=np.array([.4,.5])
    method=r.RBFFD(r.PHS(5),polynomial_degree=2,stencil_size=17,stencil_policy=r.StencilPolicy(scaling='local'))
    local=method.weights(centers=p,target=q,operator=r.Laplacian())
    np.testing.assert_array_equal(local.indices,np.arange(17))
    np.testing.assert_allclose(local.weights[:,0]@(p*p).sum(1),4,atol=1e-10)
    op=method.operators(source=p,targets=q[None],operators={'lap':r.Laplacian()}).lap
    selected=op.local(0).indices
    direct=method.weights(centers=p[selected],target=q,operator=r.Laplacian())
    np.testing.assert_allclose(direct.weights,op.local(0).weights,atol=1e-11)
    local_system=local.reconstruct_local()
    np.testing.assert_allclose(local_system.matrix.T@np.vstack((local.weights,local.multipliers)),local_system.rhs,atol=1e-10)


def test_explicit_groups_reproduce_default_matrix():
    cloud,problem,_=problem_cloud(4)
    method=r.LHI(r.PHS(5),stencil_size=15,polynomial_degree=2)
    original=method.assemble(problem,cloud)
    ii,bi=cloud.interior_indices,cloud.boundary_indices
    imap={n:i for i,n in enumerate(ii)};bmap={n:i for i,n in enumerate(bi)}
    groups={
      'values':r.CenterGroup(cloud.interior,role='solution',indices=[[imap[n] for n in s.solution_indices] for s in original.stencils],target='require'),
      'wall':r.CenterGroup(cloud.points[bi],role='boundary',operator=r.Identity(),data=problem.boundary[0].rhs,indices=[[bmap[n] for n in s.boundary_indices] for s in original.stencils]),
      'equation':r.CenterGroup(cloud.interior,role='pde',indices=[[imap[n] for n in s.pde_indices] for s in original.stencils],target='exclude'),
    }
    explicit=r.LHI(r.PHS(5),polynomial_degree=2,centers=groups).assemble(problem,cloud)
    np.testing.assert_allclose(explicit.matrix.toarray(),original.matrix.toarray(),atol=1e-12)
    np.testing.assert_allclose(explicit.rhs,original.rhs,atol=1e-12)
    np.testing.assert_allclose(explicit.solve().interior_values,original.solve().interior_values,atol=1e-12)


@pytest.mark.parametrize('digits',[None,35])
def test_independent_pde_and_derivative_data_clouds(digits):
    cloud,problem,truth=problem_cloud()
    pde=np.array([[.21,.35],[.77,.35],[.3,.71],[.68,.71],[.5,.48]])
    groups={
      'u':r.CenterGroup(cloud.interior,role='solution',target='require'),
      'boundary':r.CenterGroup(cloud.points[cloud.boundary_indices],role='boundary',operator=r.Identity(),data=truth),
      'forcing':r.CenterGroup(pde,role='pde',size=3,target='exclude'),
      'sensor':r.CenterGroup(np.array([[.47,.61]]),role='data',operator=r.Derivative(0),data=.94),
    }
    precision=r.Precision(local_digits=digits,global_dtype='mpmath' if digits else 'float64')
    system=r.LHI(r.PHS(5),polynomial_degree=2,precision=precision,centers=groups).assemble(problem,cloud)
    q=np.array([[.31,.47],[.68,.57]])
    np.testing.assert_allclose(system.solve().evaluate(q),truth(q),atol=2e-10)
    assert len(system.stencils[0].groups['forcing'])==3
    assert system.dof_map['size']==len(cloud.interior)


def test_group_duplicate_functionals_and_target_rules():
    cloud,problem,truth=problem_cloud()
    group=r.CenterGroup(cloud.interior,role='pde',indices=[[0]]*len(cloud.interior),target='exclude')
    with pytest.raises(ValueError,match='exclusion'):group.select(cloud.interior[0],0,2)
    groups={'u':r.CenterGroup(cloud.interior,role='solution'),'duplicate':r.CenterGroup(cloud.interior,role='data',operator=r.Identity(),data=truth)}
    with pytest.raises(ValueError,match='Duplicate local functionals'):
        r.LHI(r.IMQ(),centers=groups).assemble(problem,cloud)


@pytest.mark.parametrize('method',[r.RBFFD(r.PHS(5),stencil_size=12,polynomial_degree=2),r.LHI(r.PHS(5),stencil_size=12,polynomial_degree=2),r.GlobalCollocation(r.PHS(5),polynomial_degree=2)])
def test_matrix_mutation_invalidates_factor(method):
    cloud,problem,_=problem_cloud()
    system=method.assemble(problem,cloud);system.solve()
    factor=system.factor
    system.matrix*=2
    solution=system.solve()
    assert system.factor is not factor
    assert solution.diagnostics['relative_residual']<1e-9


def test_capabilities_reject_before_assembly():
    cloud,problem,_=problem_cloud()
    with pytest.raises(NotImplementedError,match='PythonBackend'):
        r.LHI(r.IMQ(),local_backend=r.CppBackend()).preflight(problem,cloud)
    with pytest.raises(NotImplementedError,match='standard'):
        r.RBFFD(r.IMQ(),scheme='symmetric').weights(centers=cloud.points,target=[.5,.5],operator=r.Laplacian())
    with pytest.raises(ValueError,match='both'):
        r.RBFFD(r.IMQ(),spaces={'u':r.ScalarSpace(r.IMQ())}).preflight(operation='operators')
    with pytest.raises(NotImplementedError,match='SVD'):
        r.RBFFD(r.IMQ(),local_solver=r.LocalSolver('svd')).preflight(operation='operators')
    m=r.SymbolicScalar(transient=True);u=m.field
    transient=m.evolution(sp.Eq(sp.diff(u,m.time)-m.laplacian(u),0),initial=0,boundary=[m.bc('boundary',sp.Eq(u,0))])
    with pytest.raises(NotImplementedError,match='mass maps'):
        r.LHI(r.IMQ(),centers={'u':r.CenterGroup(cloud.interior,role='solution')}).preflight(transient,cloud)


def test_prepare_returns_new_object_and_namespace_is_explicit():
    cloud,problem,_=problem_cloud()
    method=r.GlobalCollocation(r.IMQ())
    prepared=method.prepare(problem)
    assert prepared is not method
    assert method.kernel == r.IMQ()
    assert isinstance(r.SymbolicStokes(),r.SymbolicStokes)
    with pytest.raises(TypeError):r.SymbolicSystem(vector_fields=('U',),scalar_fields=('p',))
    for name in ('rbf_fd_weights','GlobalStokes','BlockLHI','LegacyCppLHIBackend','from_rbfmeshgen','CudaLHIBackend','TorchKernel'):
        assert not hasattr(r,name)


def test_scalar_space_route_is_shared():
    cloud,problem,_=problem_cloud()
    for cls in (r.LHI,r.GlobalCollocation,r.RBFFD):
        kwargs={} if cls is r.GlobalCollocation else {'stencil_size':12}
        a=cls(r.PHS(5),polynomial_degree=2,**kwargs).assemble(problem,cloud)
        b=cls(spaces={'u':r.ScalarSpace(r.PHS(5),2)},**kwargs).assemble(problem,cloud)
        dense=lambda m:m.toarray() if hasattr(m,'toarray') else m
        np.testing.assert_allclose(dense(a.matrix),dense(b.matrix),atol=1e-12)


def test_transient_cache_invalidates_on_spatial_edit():
    cloud=r.geometry.unit_box_grid(3)
    m=r.SymbolicScalar(transient=True);u=m.field;x,y=m.coordinates
    exact=sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    problem=m.evolution(sp.Eq(sp.diff(u,m.time)-m.laplacian(u),0),initial=exact,boundary=[m.bc('boundary',sp.Eq(u,0))])
    system=r.RBFFD(r.PHS(5),stencil_size=12,polynomial_degree=2).assemble(problem,cloud)
    system.solve(.01,1)
    old=list(system._factors.values())[0]
    system.matrix*=1.01
    system.solve(.01,1)
    assert list(system._factors.values())[0] is not old


def test_full_precision_fingerprint_observes_sub_float64_change():
    from rbflab.api_contracts import fingerprint
    import mpmath
    ctx=mpmath.mp.clone();ctx.dps=50
    matrix=ctx.matrix([[1]])
    before=fingerprint(matrix);matrix[0,0]+=ctx.mpf('1e-40')
    assert before!=fingerprint(matrix)


def test_center_example_and_neumann_group():
    from examples.tutorials.lhi_centers import run
    assert run()<1e-10
    cloud,problem,truth=problem_cloud()
    q=np.array([[.22,.37]])
    group=r.CenterGroup(q,role='boundary',operator=r.NormalDerivative(),normals=[[1,0]],data=.44)
    groups={'u':r.CenterGroup(cloud.interior,role='solution'),
            'wall':r.CenterGroup(cloud.points[cloud.boundary_indices],role='boundary',operator=r.Identity(),data=truth),
            'flux':group,'pde':r.CenterGroup(cloud.interior,role='pde',target='exclude')}
    solution=r.LHI(r.PHS(5),centers=groups,polynomial_degree=2).assemble(problem,cloud).solve()
    np.testing.assert_allclose(solution.evaluate(q),truth(q),atol=1e-10)


def test_independent_groups_in_three_dimensions():
    cloud=r.geometry.unit_box_grid(2,3)
    m=r.SymbolicScalar(3);u=m.field;x,y,z=m.coordinates
    exact=1+x*x+y*y+z*z;truth=m.data(exact)
    problem=m.stationary(sp.Eq(-m.laplacian(u),-6),boundary=[m.bc('boundary',sp.Eq(u,exact))])
    groups={'u':r.CenterGroup(cloud.interior,role='solution'),
            'wall':r.CenterGroup(cloud.points[cloud.boundary_indices],role='boundary',operator=r.Identity(3),data=truth),
            'pde':r.CenterGroup([[.3,.4,.6],[.7,.6,.4]],role='pde')}
    solution=r.LHI(r.PHS(5),polynomial_degree=2,centers=groups).assemble(problem,cloud).solve()
    query=np.array([[.31,.42,.57]])
    np.testing.assert_allclose(solution.evaluate(query),truth(query),atol=1e-10)


@pytest.mark.parametrize('method_type',[r.GlobalCollocation,r.LHI])
def test_coupled_scalar_blocks_use_canonical_methods(method_type):
    cloud=r.geometry.unit_box_grid(3)
    m=r.SymbolicSystem(fields=('a','b'));a,b=m.fields;x,y=m.coordinates
    exact_a=1+x*x;exact_b=1+y*y
    problem=m.stationary([sp.Eq(-m.laplacian(a)+b,-2+exact_b),sp.Eq(-m.laplacian(b)+a,-2+exact_a)],
        boundary=[m.bc('boundary',[sp.Eq(a,exact_a),sp.Eq(b,exact_b)])])
    method=method_type(spaces={a:r.ScalarSpace(r.PHS(5),2),b:r.ScalarSpace(r.PHS(5),2)})
    system=method.assemble(problem,cloud)
    solution=system.solve()
    assert solution.diagnostics['relative_residual']<1e-9
    assert system.dof_map['fields']==(a,b)
