"""Native symbolic data parity, boundary precedence and Python fallback."""
import os
from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest
import sympy as sp
import rbflab as r
from rbflab.symbolic import SymbolicData
from rbflab.time_data import TimeData

native=pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')


def method(spaces,digits,tmp_path):
    return r.GlobalCollocation(spaces=spaces,precision=r.Precision(global_digits=digits),
        backend=r.CppGlobalBackend(threads=4,compute_condition=False,cache_dir=tmp_path))


@native
@pytest.mark.parametrize('dimension',[2,3])
def test_native_data_and_trajectory_parity(tmp_path,dimension):
    from test_global_cpp_unsteady import problem_and_spaces
    problem,spaces=problem_and_spaces(dimension,hybrid=True)
    coordinates=problem.model.coordinates
    problem=replace(problem,initial=tuple(problem.model.data(
        sp.exp(-sp.Rational(1,10))*coordinates[(j+1)%dimension]**2) for j in range(dimension)))
    with method(spaces,60,tmp_path).assemble(problem,r.geometry.unit_box_grid(2,dimension)) as system:
        assert system.data_backend=='cpp_symbolic'
        ctx=system.ctx
        for time in ['0','.001','.137']:
            assert ctx.mnorm(system.data(time)-system._python_data(ctx.mpf(time)),'inf')<ctx.mpf('1e-55')
        for scheme in ['backward_euler','bdf2']:
            actual=system.solve('.01',4,scheme=scheme,start_time='.1')
            assert actual.diagnostics['data_backend']=='cpp_symbolic'
            original=system.data;system.data=system._python_data
            try:expected=system.solve('.01',4,scheme=scheme,start_time='.1')
            finally:system.data=original
            assert expected.diagnostics['new_factorizations']==0
            for a,b in zip(actual.states,expected.states):
                for name in ['velocity','pressure_gradient','residuals','divergence']:
                    assert ctx.mnorm(getattr(a,name)([[.31]*dimension],extended=True)-
                        getattr(b,name)([[.31]*dimension],extended=True),'inf')<ctx.mpf('1e-44')
        # Replacing the problem must not retain compiled data silently.
        system.problem=replace(problem,forcing=tuple(2*v for v in [1]*dimension))
        assert ctx.mnorm(system.data('.2')-system._python_data(ctx.mpf('.2')),'inf')==0
        assert system.data_backend=='python'


@native
def test_boundaries_constants_and_pooled_precision(tmp_path):
    from test_global_cpp_unsteady import problem_and_spaces
    problem,spaces=problem_and_spaces(hybrid=True);model=problem.model
    x,y=model.coordinates;t=model.time
    value=sp.Float('0.123456789012345678901234567890123456789',45)
    data=lambda expression:SymbolicData(expression,model.coordinates,time_symbol=t)
    forcing=(data(sp.pi*sp.sin(t)+value*x+sp.E),data(sp.Rational(1,3)*y*sp.cos(t)))
    problem=replace(problem,forcing=forcing,boundary=(('left',(data(x+t),data(y-t))),
        ('rest',(data(x+t+1),data(y-t+2)))))
    original=r.geometry.unit_box_grid(8)
    left=np.flatnonzero(original.points[:,0]==0)
    cloud=r.geometry.PointCloud(original.points,{'left':left,
        'rest':np.setdiff1d(original.boundary_indices,left)},{})
    process=None
    for digits in [60,80,35]:
        with method(spaces,digits,tmp_path).assemble(problem,cloud) as system:
            assert system.data_backend=='cpp_symbolic'
            pid=system.worker._worker.process.pid
            if process is not None:assert pid==process
            process=pid;ctx=system.ctx
            for time in [Fraction(1,10),Fraction(7,31)]:
                actual=system.data(time);expected=system._python_data(system.number(time))
                assert ctx.mnorm(actual-expected,'inf')<ctx.mpf(10)**(-digits+5)
            assert all(v==0 for v in list(actual)[system.n:])
            with pytest.raises(RuntimeError,match='row count'):system.worker.request('DATA_INIT 0')
            # Mutating cloud geometry must use the original Python semantics.
            saved=cloud.points[0,0];cloud.points[0,0]=.01
            try:
                assert ctx.mnorm(system.data('.2')-system._python_data(ctx.mpf('.2')),'inf')==0
                assert system.data_backend=='python'
            finally:cloud.points[0,0]=saved


@native
def test_callback_fallback_retains_execution(tmp_path):
    from test_global_cpp_unsteady import problem_and_spaces
    problem,spaces=problem_and_spaces();calls=[]
    callback=TimeData(lambda points,time:np.full(len(points),time),
        lambda ctx,x,y,time:calls.append(time) or time+x-y)
    problem=replace(problem,forcing=(callback,problem.forcing[1]))
    with method(spaces,60,tmp_path).assemble(problem,r.geometry.unit_box_grid(2)) as system:
        assert system.data_backend=='python'
        before=len(calls);system.data('.1');assert len(calls)>before
        system.data('.1');assert len(calls)>2*before
        assert system.data_fallback_reason


@native
def test_conflicting_overlap_retains_python_precedence_and_errors(tmp_path):
    from test_global_cpp_unsteady import problem_and_spaces
    problem,spaces=problem_and_spaces();model=problem.model;x,y=model.coordinates
    problem=replace(problem,boundary=(('boundary',(model.data(x),model.data(y))),
        ('boundary',(model.data(x+1),model.data(y+1)))))
    with method(spaces,60,tmp_path).assemble(problem,r.geometry.unit_box_grid(2)) as system:
        assert system.data_backend=='python'
        assert 'Overlapping' in system.data_fallback_reason
        assert system.ctx.mnorm(system.data('.1')-system._python_data(system.number('.1')),'inf')==0


@native
def test_stationary_symbolic_data_double(tmp_path):
    model=r.SymbolicStokes(2);u,p=model.fields;x,y=model.coordinates
    velocity=sp.ImmutableMatrix([y*y,x*x]);pressure=x+2*y
    problem=model.stationary(sp.Eq(-model.laplacian(u)+model.gradient(p),
        -model.laplacian(velocity)+model.gradient(pressure)),
        boundary=[model.bc('boundary',sp.Eq(u,velocity))])
    spaces={u:r.DivergenceFreeSpace(r.IMQ(),2),p:r.PressureSpace(r.IMQ(),1)}
    with method(spaces,None,tmp_path).assemble(problem,r.geometry.unit_box_grid(2)) as system:
        assert system.data_backend=='cpp_symbolic'
        np.testing.assert_allclose(system.data(0),system._python_data(0),atol=1e-13)
