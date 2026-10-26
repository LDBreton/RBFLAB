"""Optional dense engine parity, ownership and pooled MPFR precision contracts."""
import os
import math
import numpy as np
import pytest
import sympy as sp
import rbflab as r

native=pytest.mark.skipif(
    os.environ.get('RBFLAB_CPP_TESTS')!='1' or not os.environ.get('RBFLAB_MPLAPACK_DIR'),
    reason='Opt-in native MPLAPACK integration with explicit dependency prefix')


def backend(tmp_path,threads=4,condition=True,solver='mplapack'):
    return r.CppGlobalBackend(threads=threads,compute_condition=condition,
        cache_dir=tmp_path,linear_solver=solver,
        mplapack_dir=os.environ['RBFLAB_MPLAPACK_DIR'] if solver=='mplapack' else None)


@native
@pytest.mark.parametrize('threads',[1,4])
def test_nonpolynomial_blocked_lu_matches_eigen(tmp_path,threads):
    model=r.SymbolicScalar();u=model.field;x,y=model.coordinates
    exact=sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    problem=model.stationary(sp.Eq(-model.laplacian(u),2*sp.pi**2*exact),
        boundary=[model.bc('boundary',sp.Eq(u,0))])
    query=np.random.default_rng(20261022).uniform(.05,.95,(64,2))
    reference=None
    for solver in ('eigen','mplapack'):
        method=r.GlobalCollocation(r.PHS(7),polynomial_degree=3,
            precision=r.Precision(global_digits=60),
            backend=backend(tmp_path,threads,False,solver))
        with method.assemble(problem,r.geometry.unit_box_grid(8)) as system:
            solution=system.solve();ctx=system.ctx
            assert solution.diagnostics['relative_residual_mp']<ctx.mpf('1e-45')
            values=solution.evaluate_mp(query)
            if solver=='eigen':
                reference=(solution.coefficients.tolist(),values.tolist())
            else:
                coefficients=ctx.matrix(reference[0]);expected=ctx.matrix(reference[1])
                scale=max(ctx.mpf(1),ctx.mnorm(coefficients,'inf'))
                assert ctx.mnorm(solution.coefficients-coefficients,'inf')/scale<ctx.mpf('1e-45')
                assert ctx.mnorm(values-expected,'inf')<ctx.mpf('1e-45')
                if os.name=='nt':
                    assert system.worker._worker.process.args[0].endswith('.exe')


@native
@pytest.mark.parametrize('scheme',['symmetric','asymmetric'])
def test_scalar_precision_pool_condition_and_mutation(tmp_path,scheme):
    model=r.SymbolicScalar();u=model.field;x,y=model.coordinates
    exact=1+x*x+2*y*y
    problem=model.stationary(sp.Eq(-model.laplacian(u)+u,-6+exact),
        boundary=[model.bc('boundary',sp.Eq(u,exact))])
    cloud=r.geometry.unit_box_grid(2)
    processes=[]
    for digits,threads in [(60,4),(80,1),(35,4)]:
        method=r.GlobalCollocation(r.PHS(5),scheme=scheme,polynomial_degree=2,
            precision=r.Precision(global_digits=digits),backend=backend(tmp_path,threads))
        with method.assemble(problem,cloud) as system:
            processes.append(system.worker._worker.process.pid)
            bits=math.ceil(digits*math.log2(10))+8
            assert system.worker.request('SOLVER_INFO')==['mplapack',str(threads),str(bits),str(bits)]
            solution=system.solve();ctx=system.ctx;tol=ctx.mpf(10)**(-digits+9)
            assert abs(solution.evaluate_mp([[.25,.5]])[0]-ctx.mpf('1.5625'))<tol
            assert solution.diagnostics['linear_solver']=='mplapack'
            assert ctx.isfinite(solution.diagnostics['scaled_condition_mp'])
            system.rhs=[v*2 for v in system.rhs]
            assert abs(system.solve().evaluate_mp([[.25,.5]])[0]-ctx.mpf('3.125'))<tol
            assert abs(solution.evaluate_mp([[.25,.5]])[0]-ctx.mpf('1.5625'))<tol
            system.matrix=system.matrix.to_mpmath()*2
            assert abs(system.solve().evaluate_mp([[.25,.5]])[0]-ctx.mpf('1.5625'))<tol
    assert len(set(processes))==1


@native
def test_asymmetric_transpose_condition_and_singular_error(tmp_path):
    problem=r.LinearPDE(r.Identity(),1,(r.Dirichlet(1),))
    method=r.GlobalCollocation(r.IMQ(),scheme='asymmetric',precision=r.Precision(global_digits=60),backend=backend(tmp_path))
    with method.assemble(problem,r.geometry.unit_box_grid(2)) as system:
        ctx=system.ctx;n=system.size
        # Triangular nonsymmetric matrix requires correct transposed pivot solve
        # in the 1-norm condition estimator, independently of symmetric kernels.
        A=ctx.eye(n)
        for i in range(n-1):A[i,i+1]=ctx.mpf('1.3')
        A[0,0]=ctx.mpf('.2');A[n-1,0]=ctx.mpf('.7')
        system.matrix=A
        solution=system.solve()
        expected=ctx.lu_solve(A,ctx.matrix(system.rhs))
        assert ctx.mnorm(solution.coefficients-expected,'inf')<ctx.mpf('1e-55')
        scales=[1/ctx.sqrt(max(abs(A[i,j]) for j in range(n))) for i in range(n)]
        B=ctx.matrix([[scales[i]*A[i,j]*scales[j] for j in range(n)] for i in range(n)])
        condition=ctx.mnorm(B,1)*ctx.mnorm(B**-1,1)
        estimate=solution.diagnostics['scaled_condition_mp']
        assert 1<=estimate<=condition*(1+ctx.mpf('1e-54'))
        eigen_method=r.GlobalCollocation(r.IMQ(),scheme='asymmetric',
            precision=r.Precision(global_digits=60),backend=backend(tmp_path,solver='eigen'))
        with eigen_method.assemble(problem,r.geometry.unit_box_grid(2)) as eigen:
            eigen.matrix=A
            reference=eigen.solve().diagnostics['scaled_condition_mp']
            assert abs(estimate-reference)<ctx.mpf('1e-54')
        A=ctx.eye(n)
        for j in range(n):A[n-1,j]=A[0,j]
        system.matrix=A
        with pytest.raises(RuntimeError,match='Singular'):system.solve()


@native
@pytest.mark.parametrize('dimension',[2,3])
def test_stokes_shift_and_full_field_parity(tmp_path,dimension):
    from test_global_cpp_unsteady import problem_and_spaces
    problem,spaces=problem_and_spaces(dimension,hybrid=True)
    cloud=r.geometry.unit_box_grid(2,dimension);precision=r.Precision(global_digits=60)
    queries=np.full((2,dimension),.37);queries[1,0]=.71
    reference={}
    for solver in ['eigen','mplapack']:
        method=r.GlobalCollocation(spaces=spaces,precision=precision,
            backend=backend(tmp_path,condition=True,solver=solver))
        with method.assemble(problem,cloud) as system:
            for scheme in ['backward_euler','bdf2']:
                trajectory=system.solve('.01',4,scheme=scheme)
                assert trajectory.diagnostics['linear_solver']==solver
                for i,state in enumerate(trajectory.states):
                    for field in ['velocity','pressure_gradient','divergence','residuals']:
                        actual=getattr(state,field)(queries,extended=True);key=(scheme,i,field)
                        if solver=='eigen':reference[key]=actual
                        else:assert system.ctx.mnorm(actual-system.ctx.matrix(reference[key].tolist()),'inf')<system.ctx.mpf('1e-45')
                    pressure=state.pressure(queries,reference=(np.zeros(dimension),0),extended=True)
                    key=(scheme,i,'pressure')
                    if solver=='eigen':reference[key]=pressure
                    else:assert system.ctx.mnorm(pressure-system.ctx.matrix(reference[key].tolist()),'inf')<system.ctx.mpf('1e-45')
                assert system.solve('.01',4,scheme=scheme).diagnostics['new_factorizations']==0
            system.matrix=system.matrix.to_mpmath()
            assert system.solve('.01',2).diagnostics['new_factorizations']==2
