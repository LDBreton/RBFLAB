"""Native method-of-lines Stokes matrices and time-step parity."""
import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r

native=pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')


def problem_and_spaces(dimension=2,hybrid=False):
    model=r.SymbolicStokes(dimension,transient=True)
    u,p=model.fields;t=model.time;x=model.coordinates
    velocity=sp.exp(-t)*sp.ImmutableMatrix([x[(j+1)%dimension]**2 for j in range(dimension)])
    pressure=sp.exp(-t)*sum((j+1)*v for j,v in enumerate(x))
    lhs=sp.diff(u,t)-model.laplacian(u)+model.gradient(p)
    forcing=sp.diff(velocity,t)-model.laplacian(velocity)+model.gradient(pressure)
    problem=model.evolution([sp.Eq(lhs,forcing)],initial={u:velocity.subs(t,0)},
        boundary=[model.bc('boundary',sp.Eq(u,velocity))])
    vk=r.Hybrid(r.IMQ(1),r.PHS(7),1) if hybrid else r.IMQ(1)
    pk=r.Hybrid(r.IMQ(1),r.PHS(3),1) if hybrid else r.IMQ(1)
    return problem,{u:r.DivergenceFreeSpace(vk,2),p:r.PressureSpace(pk,1)}


@native
@pytest.mark.parametrize('digits,dimension,hybrid',[(None,2,True),(40,2,True),(None,3,False),(40,3,False)])
def test_native_mol_parity_and_ownership(digits,dimension,hybrid,tmp_path,monkeypatch):
    problem,spaces=problem_and_spaces(dimension,hybrid)
    cloud=r.geometry.unit_box_grid(2,dimension)
    precision=r.Precision(global_digits=digits)
    expected=r.GlobalCollocation(spaces=spaces,precision=precision).assemble(problem,cloud)
    trajectories={scheme:expected.solve('.01',4,scheme=scheme) for scheme in ['backward_euler','bdf2']}
    q=np.full((2,dimension),.37);q[1,0]=.71
    reference={(scheme,i,name):getattr(state,name)(q,extended=bool(digits))
        for scheme,tr in trajectories.items() for i,state in enumerate(tr.states)
        for name in ['velocity','pressure_gradient','residuals','divergence']}
    reference_pressure={(scheme,i):state.pressure(q,reference=(np.zeros(dimension),0),extended=bool(digits))
        for scheme,tr in trajectories.items() for i,state in enumerate(tr.states)}
    from rbflab.nodal import Arithmetic
    import rbflab.space_stokes as flow
    def forbidden(*args,**kwargs):raise AssertionError('Python dense kernel assembly/factorization used')
    monkeypatch.setattr(Arithmetic,'factor',forbidden)
    monkeypatch.setattr(flow,'_blocks',forbidden)
    method=r.GlobalCollocation(spaces=spaces,precision=precision,
        backend=r.CppGlobalBackend(cache_dir=tmp_path,threads=2,compute_condition=not digits and dimension==2))
    with method.assemble(problem,cloud) as actual:
        def equal(a,b):
            if digits:
                assert actual.ctx.mnorm(a-actual.ctx.matrix(b.tolist()),'inf')<actual.ctx.mpf('1e-27')
            else:np.testing.assert_allclose(a,b,atol=2e-8,rtol=2e-8)
        equal(actual.matrix.to_mpmath() if digits else actual.matrix.to_numpy(),expected.matrix)
        equal(actual.mass.to_mpmath() if digits else actual.mass.to_numpy(),expected.mass)
        equal(actual.data(actual.number('.02')),expected.data(expected.arithmetic.number('.02')))
        count=dimension*len(cloud.interior)
        assert np.count_nonzero(actual.mass.to_numpy()[count:])==0
        with pytest.raises(AttributeError):actual.mass=np.eye(actual.size)
        for scheme in ['backward_euler','bdf2']:
            trajectory=actual.solve('.01',4,scheme=scheme)
            for i,a in enumerate(trajectory.states):
                for name in ['velocity','pressure_gradient','residuals','divergence']:
                    equal(getattr(a,name)(q,extended=bool(digits)),reference[scheme,i,name])
                equal(a.pressure(q,reference=(np.zeros(dimension),0),extended=bool(digits)),
                      reference_pressure[scheme,i])
            # Inspecting A after stepping must still return the stiffness matrix.
            equal(actual.matrix.to_mpmath() if digits else actual.matrix.to_numpy(),expected.matrix)
            assert trajectory.diagnostics['backend']=='cpp_global'
        assert actual.solve('.01',4).diagnostics['new_factorizations']==0
        actual.matrix=actual.matrix.to_mpmath() if digits else actual.matrix.to_numpy()
        assert actual.solve('.01',2).diagnostics['new_factorizations']==2
        with pytest.raises(ValueError,match='requires dt'):actual.solve()
        with pytest.raises(ValueError,match='Positive'):actual.solve('-1',2)
        with pytest.raises(ValueError,match='Unknown'):actual.solve('.01',2,scheme='rk4')
    with pytest.raises(RuntimeError,match='closed'):actual.mass.to_numpy()


@native
def test_incompatible_initial_velocity(tmp_path):
    from dataclasses import replace
    problem,spaces=problem_and_spaces()
    problem=replace(problem,initial=(0,0))
    method=r.GlobalCollocation(spaces=spaces,backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False))
    with method.assemble(problem,r.geometry.unit_box_grid(2)) as system:
        with pytest.raises(ValueError,match='incompatible'):system.solve('.01',2)


@native
def test_parallel_precision_reuse_and_evaluation_cache(tmp_path):
    """Reused workers must reset precision/cache; query caches exclude coefficients."""
    problem,spaces=problem_and_spaces(hybrid=True)
    cloud=r.geometry.unit_box_grid(8)
    query=np.random.default_rng(413).uniform(.1,.9,(130,2))
    reference=None
    for digits,threads in [(60,4),(80,4),(60,1)]:
        method=r.GlobalCollocation(spaces=spaces,precision=r.Precision(global_digits=digits),
            backend=r.CppGlobalBackend(cache_dir=tmp_path,threads=threads,compute_condition=False))
        with method.assemble(problem,cloud) as system:
            configured,bits=map(int,system.worker.request('RUNTIME_INFO'))
            assert configured==threads and bits==int(np.ceil(digits*np.log2(10)))+8
            info=lambda:tuple(map(int,system.worker.request('EVAL_CACHE_INFO')))
            assert info()==(0,0,0,0)
            tr=system.solve('.01',2,scheme='backward_euler')
            tr.states[0].velocity(cloud.interior,extended=True)
            assert info()[0]==1 and info()[3]==0  # Reuse mass, no second matrix.
            first=tr.states[0].pressure_gradient(query,extended=True)
            assert info()[1:2]==(1,)
            second=tr.states[1].pressure_gradient(query,extended=True)
            assert info()[0]==2 and info()[1]==1
            assert system.ctx.mnorm(first-second,'inf')>system.ctx.mpf('1e-5')
            velocity=tr.states[1].velocity(query,extended=True)
            shifted=tr.states[1].pressure_gradient(query+.001,extended=True)
            assert info()[1]==3  # Operator and point changes must miss.
            current=(second,velocity,shifted)
            if reference is None:reference=[v.tolist() for v in current]
            else:
                for value,expected in zip(current,reference):
                    assert system.ctx.mnorm(value-system.ctx.matrix(expected),'inf')<system.ctx.mpf('1e-42')
            assert system.solve('.01',2,scheme='backward_euler').diagnostics['new_factorizations']==0
