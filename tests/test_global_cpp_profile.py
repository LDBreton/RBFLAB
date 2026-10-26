"""Opt-in native phase timing retains numerical and cached-factor contracts."""
import os
import pytest
import rbflab as r

native=pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')


def timings(worker):
    tokens=worker.request('GLOBAL_TIMING');count=int(tokens[0]);records={}
    assert len(tokens)==1+12*count
    for i in range(count):
        row=tokens[1+12*i:1+12*(i+1)]
        records[row[0]]=(int(row[1]),list(map(float,row[2:])))
    return records


@native
@pytest.mark.parametrize('digits',[None,60])
def test_global_phase_timers_and_reset(tmp_path,digits):
    from test_global_cpp_unsteady import problem_and_spaces
    problem,spaces=problem_and_spaces(hybrid=True)
    method=r.GlobalCollocation(spaces=spaces,precision=r.Precision(global_digits=digits),
        backend=r.CppGlobalBackend(cache_dir=tmp_path,compute_condition=False))
    with method.assemble(problem,r.geometry.unit_box_grid(2)) as system:
        worker=system.worker
        assert timings(worker)=={}
        with pytest.raises(RuntimeError,match='0 or 1'):worker.request('GLOBAL_PROFILE 2')
        worker.request('GLOBAL_PROFILE 1')
        first=system.solve('.01',2,scheme='backward_euler')
        records=timings(worker)
        assert set(records)=={'SHIFT','SOLVE','MASS_MV','RHS_DATA'}
        assert all(count==2 for count,_ in records.values())
        for _,seconds in records.values():
            assert all(v>=0 for v in seconds)
            assert sum(seconds[:9])<=seconds[9]*(1+1e-6)
        assert records['SHIFT'][1][2]>0
        before=first.states[-1].velocity([[.3,.4]],extended=bool(digits))
        worker.request('GLOBAL_PROFILE 1')
        second=system.solve('.01',2,scheme='backward_euler')
        assert second.diagnostics['new_factorizations']==0
        assert timings(worker)['SHIFT'][1][2]==0
        assert second.states[-1].velocity([[.3,.4]],extended=bool(digits)).tolist()==before.tolist()
        worker.request('GLOBAL_PROFILE 0')
        assert timings(worker)=={}
        system.solve('.01',1,scheme='backward_euler')
        assert timings(worker)=={}
        worker.request('GLOBAL_PROFILE 1');worker.request('RESET')
        assert timings(worker)=={}
