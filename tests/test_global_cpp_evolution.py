"""Native scalar evolution preserves the existing semidiscrete method."""
import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r
from rbflab.nodal import Arithmetic

native=pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')


def problem(dimension=2,neumann=False):
    model=r.SymbolicScalar(dimension,transient=True);u=model.field;t=model.time;x=model.coordinates
    exact=sp.exp(-t)*(1+sum(v*v for v in x))
    lhs=sp.diff(u,t)-model.laplacian(u)
    forcing=sp.diff(exact,t)-model.laplacian(exact)
    boundaries=[]
    if neumann:
        for j,name in enumerate(('x','y','z')[:dimension]):
            for suffix in ['min','max']:
                boundaries.append(model.bc(name+suffix,sp.Eq(sp.diff(u,x[j]),sp.diff(exact,x[j]))))
    else:boundaries=[model.bc('boundary',sp.Eq(u,exact))]
    return model,model.evolution(sp.Eq(lhs,forcing),initial=exact.subs(t,0),boundary=boundaries),exact


@native
@pytest.mark.parametrize('scheme',['symmetric','asymmetric'])
@pytest.mark.parametrize('digits',[None,60])
def test_scalar_evolution_matrices_trajectories_and_reuse(tmp_path,scheme,digits,monkeypatch):
    _,p,_=problem();cloud=r.geometry.unit_box_grid(3)
    precision=r.Precision(global_digits=digits)
    expected=r.GlobalCollocation(r.IMQ(),scheme=scheme,polynomial_degree=2,precision=precision).assemble(p,cloud)
    reference={s:expected.solve('.01',4,scheme=s) for s in ['backward_euler','bdf2']}
    q=[[.3,.4],[.62,.27]];operators=[None,r.Derivative(0),r.Laplacian()]
    fields={(s,i,j):(state.evaluate_mp(q,op) if digits else state.evaluate(q,op))
        for s,trajectory in reference.items() for i,state in enumerate(trajectory.states) for j,op in enumerate(operators)}
    def forbidden(*args,**kwargs):raise AssertionError('Python dense factor or kernel assembly used')
    monkeypatch.setattr(Arithmetic,'factor',forbidden)
    monkeypatch.setattr(Arithmetic,'matrix',forbidden)
    method=r.GlobalCollocation(spaces={'u':r.ScalarSpace(r.IMQ(),2)},scheme=scheme,precision=precision,
        backend=r.CppGlobalBackend(threads=4,compute_condition=False,cache_dir=tmp_path))
    with method.assemble(p,cloud) as actual:
        def equal(a,b,tol=None):
            if digits:
                if hasattr(a,'to_mpmath'):a=a.to_mpmath()
                assert actual.ctx.mnorm(a-actual.ctx.matrix(b.tolist()),'inf')<actual.ctx.mpf(tol or '1e-45')
            else:np.testing.assert_allclose(np.asarray(a),b,atol=2e-8,rtol=2e-8)
        equal(actual.spatial,expected.spatial);equal(actual.mass,expected.mass);equal(actual.initial_matrix,expected.initial_matrix)
        assert np.count_nonzero(actual.mass.to_numpy()[cloud.boundary_indices])==0
        for s in reference:
            trajectory=actual.solve('.01',4,scheme=s)
            assert trajectory.diagnostics['data_backend']=='cpp_symbolic'
            assert trajectory.diagnostics['initial_constraint_residual']<1e-10
            for i,(a,b) in enumerate(zip(trajectory.states,reference[s].states)):
                equal(a.unknowns,b.unknowns);equal(a.unknown_derivative,b.unknown_derivative,'1e-42')
                for j,op in enumerate(operators):
                    equal(a.evaluate_mp(q,op) if digits else a.evaluate(q,op),fields[s,i,j])
            assert actual.solve('.01',2,scheme=s).diagnostics['new_factorizations']==0
        equal(actual.matrix,expected.spatial)
        # Native stiffness edits invalidate shifted factors, without discarding
        # the separately owned initial coefficient projection.
        actual.matrix=actual.matrix.to_mpmath() if digits else actual.matrix.to_numpy()
        assert actual.solve('.01',2).diagnostics['new_factorizations']==2
    with pytest.raises(RuntimeError):trajectory.final.evaluate([[.3,.4]])


@native
def test_three_dimensional_neumann_constant_mode(tmp_path):
    model=r.SymbolicScalar(3,transient=True);u=model.field;t=model.time;x=model.coordinates
    p=model.evolution(sp.Eq(sp.diff(u,t)-model.laplacian(u),1),initial=1,
        boundary=[model.bc('boundary',sp.Eq(model.normal_derivative(u),0))])
    cloud=r.geometry.unit_box_grid(2,3)
    # Geometry binds the normal-dependent operator; prescribed zero data is
    # symbolic and remains native.
    method=r.GlobalCollocation(r.IMQ(),polynomial_degree=1,precision=r.Precision(global_digits=60),
        backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False))
    expected=r.GlobalCollocation(r.IMQ(),polynomial_degree=1,precision=r.Precision(global_digits=60)).assemble(p,cloud)
    reference=expected.solve('.01',3)
    with method.assemble(p,cloud) as actual:
        assert actual.shifted_trial_functionals
        assert actual.ctx.mnorm(actual.matrix.to_mpmath()-actual.ctx.matrix(expected.matrix.tolist()),'inf')<actual.ctx.mpf('1e-52')
        trajectory=actual.solve('.01',3)
        q=[[.2,.3,.4]]
        assert actual.ctx.mnorm(trajectory.final.evaluate_mp(q)-actual.ctx.matrix(reference.final.evaluate_mp(q).tolist()),'inf')<actual.ctx.mpf('1e-45')


@native
def test_mplapack_scalar_evolution(tmp_path):
    prefix=os.environ.get('RBFLAB_MPLAPACK_DIR')
    if not prefix:pytest.skip('Explicit MPLAPACK prefix required')
    _,p,_=problem();cloud=r.geometry.unit_box_grid(2)
    method=r.GlobalCollocation(r.PHS(5),polynomial_degree=2,precision=r.Precision(global_digits=60),
        backend=r.CppGlobalBackend(linear_solver='mplapack',mplapack_dir=prefix,cache_dir=tmp_path,compute_condition=False))
    with method.assemble(p,cloud) as system:
        result=system.solve('.01',3)
        assert result.diagnostics['linear_solver']=='mplapack'
        assert result.diagnostics['data_backend']=='cpp_symbolic'
        assert result.diagnostics['initial_compatibility_residual']<1e-40


@native
def test_mixed_derivative_boundaries_nonzero_start_and_fallback(tmp_path):
    model=r.SymbolicScalar(2,transient=True);u=model.field;x,y=model.coordinates;t=model.time
    exact=sp.exp(-t)*(1+x*x+y*y)
    equation=sp.Eq(sp.diff(u,t)-model.laplacian(u),sp.diff(exact,t)-model.laplacian(exact))
    cloud=r.geometry.unit_box_grid(3)
    label=next(name for name,ids in cloud.boundary.items() if np.all(cloud.points[ids,0]==0))
    p=model.evolution(equation,initial=exact.subs(t,sp.Rational(1,10)),boundary=[
        model.bc(label,sp.Eq(u,exact)),model.bc('boundary',sp.Eq(model.normal_derivative(u),model.normal_derivative(exact)))])
    common=dict(kernel=r.IMQ(),polynomial_degree=2,precision=r.Precision(global_digits=60))
    expected=r.GlobalCollocation(**common).assemble(p,cloud).solve('.01',4,start_time='.1')
    with r.GlobalCollocation(**common,backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False)).assemble(p,cloud) as system:
        actual=system.solve('.01',4,start_time='.1')
        assert actual.diagnostics['data_backend']=='python'  # normal-dependent data
        for a,b in zip(actual.states,expected.states):
            assert system.ctx.mnorm(a.evaluate_mp([[.3,.4]])-system.ctx.matrix(b.evaluate_mp([[.3,.4]]).tolist()),'inf')<system.ctx.mpf('1e-44')
        initial=system.initial_matrix.to_mpmath();system.initial_matrix=initial
        assert system.solve('.01',2,start_time='.1').diagnostics['initial_constraint_residual']<1e-45


@native
def test_supported_scalar_data_does_not_call_python_symbolic_evaluation(tmp_path,monkeypatch):
    from rbflab.symbolic import SymbolicData
    _,p,_=problem()
    def forbidden(*args,**kwargs):raise AssertionError('Python symbolic evaluation used')
    monkeypatch.setattr(SymbolicData,'mp_values',forbidden)
    with r.GlobalCollocation(r.IMQ(),polynomial_degree=2,precision=r.Precision(global_digits=60),
        backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False)).assemble(p,r.geometry.unit_box_grid(2)) as system:
        assert system.solve('.01',3).diagnostics['data_backend']=='cpp_symbolic'


@native
def test_scalar_shift_zero_diagonal_and_uninitialized_commands(tmp_path):
    model=r.SymbolicScalar(2,transient=True);u=model.field
    p=model.evolution(sp.Eq(sp.diff(u,model.time)-model.laplacian(u),0),initial=0,
        boundary=[model.bc('boundary',sp.Eq(u,0))])
    with r.GlobalCollocation(r.IMQ(),precision=r.Precision(global_digits=60),
        backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False)).assemble(p,r.geometry.unit_box_grid(2)) as system:
        ctx=system.ctx;K=ctx.eye(system.size);K[0,0]=K[1,1]=0;K[0,1]=K[1,0]=1
        system.matrix=K-100*system.mass.to_mpmath()
        assert system.solve('.01',2).diagnostics['new_factorizations']==2
        system.worker.request('RESET')
        for command in ['INITIAL_REPLACE 0','EVOLVE_INIT','INITIAL_EXPORT']:
            with pytest.raises(RuntimeError,match='BUILD|INITIAL_MATRIX'):system.worker.request(command)
        assert system.worker.request('HELLO')


@native
def test_nonfinite_initial_boundary_derivative_rejected(tmp_path):
    from rbflab.time_data import InitialData
    values=r.PrecisionData(lambda points:np.zeros(len(points)),lambda ctx,x,y:ctx.zero)
    bad=r.PrecisionData(lambda points:np.full(len(points),np.nan),lambda ctx,x,y:ctx.nan)
    initial=InitialData(values,{(1,0):bad,(0,1):bad})
    p=r.EvolutionPDE(-r.Laplacian(),0,initial,(r.BoundaryCondition('boundary',r.NormalDerivative(),0),))
    with r.GlobalCollocation(r.IMQ(),polynomial_degree=0,precision=r.Precision(global_digits=60),
        backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False)).assemble(p,r.geometry.unit_box_grid(2)) as system:
        with pytest.raises(ValueError,match='finite'):system.solve('.01',2)
