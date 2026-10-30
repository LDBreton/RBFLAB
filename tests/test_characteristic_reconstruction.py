"""Contracts for the persistent exact mixed characteristic evaluator."""
from io import StringIO
from types import SimpleNamespace
import os
from pathlib import Path
import numpy as np
import pytest
from research.navier_stokes_schemes.characteristic_reconstruction import (
    CorrectedCharacteristicReconstruction, build_evaluator,
)


def protocol_evaluator(reply='RESULT 1 3.25 -4.5\n'):
    evaluator=CorrectedCharacteristicReconstruction.__new__(CorrectedCharacteristicReconstruction)
    evaluator.records=[None,None]
    evaluator.p=SimpleNamespace(U0=np.zeros(4),boundary_points=np.zeros((3,2)))
    evaluator.process=SimpleNamespace(stdin=StringIO(),stdout=StringIO(reply),stderr=StringIO())
    evaluator.evaluations=0;evaluator.seconds=0.
    return evaluator


def test_subset_protocol_retains_owner_and_node_major_observations():
    e=protocol_evaluator()
    actual=e.evaluate(np.arange(4),np.arange(4)+20,np.arange(6)+10,[[.4,.5]],rows=[1])
    np.testing.assert_array_equal(actual,[[3.25,-4.5]])
    lines=e.process.stdin.getvalue().splitlines()
    assert lines==['EVAL 0 1','0 1 2 3','10 11 12 13 14 15','20 21 22 23','1 0.40000000000000002 0.5']
    assert e.evaluations==1


@pytest.mark.parametrize('rows,points',[(None,[[.4,.5]]),([2],[[.4,.5]]),([-1],[[.4,.5]]),([0,1],[[.4,.5]])])
def test_subset_cannot_silently_reselect_a_nearest_owner(rows,points):
    e=protocol_evaluator()
    with pytest.raises(ValueError):e.evaluate(np.zeros(4),np.zeros(4),np.zeros(6),points,rows=rows)
    assert e.process.stdin.getvalue()==''


@pytest.mark.parametrize('operator,mode',[('dx',1),('dy',2),('momentum',3),('gradient',4)])
def test_derivative_requests_evaluate_the_same_native_field(operator,mode):
    e=protocol_evaluator()
    e.evaluate(np.zeros(4),np.zeros(4),np.zeros(6),[[.4,.5]],rows=[0],operator=operator)
    assert e.process.stdin.getvalue().startswith(f'EVAL {mode} 1\n')


def test_rejects_nonfinite_source_before_writing_worker():
    e=protocol_evaluator()
    with pytest.raises(ValueError):e.evaluate([np.nan,0,0,0],np.zeros(4),np.zeros(6),[[.4,.5]],rows=[0])
    assert e.process.stdin.getvalue()==''


@pytest.mark.skipif(os.environ.get('RBFLAB_CHARACTERISTIC_NATIVE_TESTS')!='1',reason='opt-in saved-source native MPFR80/MPI8 field verification')
@pytest.mark.parametrize('cells,baseline,parameters',[
    (12,'ns_cavity_candidates_20261029_native','ns_accuracy_divergence_early4_20261029'),
    (16,'ns_cavity_candidates_20261029_cells16','ns_refined_rejected_unitlid_20261029'),
])
def test_actual_mixed_field_cardinality_and_nodal_derivative_parity(cells,baseline,parameters):
    root=Path(__file__).resolve().parents[1]
    with build_evaluator(cells,root/'outputs'/baseline/'nu2',root/'outputs'/parameters/'parameters.npy',
                         root/'outputs/ns_characteristic_reconstruction_20261030'/f'cells{cells}',ranks=8) as e:
        rng=np.random.default_rng(20261030)
        U=rng.standard_normal(len(e.p.U0));L=rng.standard_normal(len(U));g=rng.standard_normal(2*len(e.p.boundary_points))
        np.testing.assert_allclose(e.evaluate(U,L,g,e.p.points).ravel(),U,rtol=0,atol=3e-8)
        for operator,names in [('dx',('DxU','DxB','DxL')),('dy',('DyU','DyB','DyL')),
                               ('momentum',('A','B','SL')),('gradient',('PY','PB','PL'))]:
            expected=sum(getattr(e.p,name)@value for name,value in zip(names,(U,g,L)))
            np.testing.assert_allclose(e.evaluate(U,L,g,e.p.points,operator=operator).ravel(),expected,rtol=1e-10,atol=2e-7)


def test_cached_text_is_immutable_and_rejects_changed_payload(tmp_path,monkeypatch):
    from research.navier_stokes_schemes.characteristic_reconstruction import _cache_text
    path=tmp_path/'jobs.in'
    _cache_text(path,'fixed numerical request\n')
    assert path.read_text()=='fixed numerical request\n'
    assert not path.with_name(path.name+'.partial').exists()
    def no_write(*args,**kwargs):
        raise AssertionError('An existing immutable cache must not be rewritten')
    monkeypatch.setattr(Path,'write_text',no_write)
    _cache_text(path,'fixed numerical request\n')
    with pytest.raises(ValueError,match='failed validation'):
        _cache_text(path,'different numerical request\n')


def test_factor_commit_publishes_report_before_factor(tmp_path,monkeypatch):
    import json
    from research.navier_stokes_schemes.characteristic_reconstruction import _commit_factors,_factor_report
    factor,report=tmp_path/'factors.txt',tmp_path/'build.json'
    descriptor={'ranks':8,'local_digits':80}
    order=[]
    original=Path.replace
    def observed(source,target):
        order.append(Path(target).name)
        if Path(target)==factor:
            assert json.loads(report.read_text())==descriptor
        return original(source,target)
    monkeypatch.setattr(Path,'replace',observed)
    _commit_factors(factor,report,['0 scaled-LU'],descriptor)
    assert order==['build.json','factors.txt']
    assert factor.read_text()=='0 scaled-LU\n'
    assert _factor_report(factor,report)==descriptor
    with pytest.raises(FileExistsError):
        _commit_factors(factor,report,['replacement'],descriptor)
    assert factor.read_text()=='0 scaled-LU\n'


def test_factor_without_completion_report_is_rejected(tmp_path):
    from research.navier_stokes_schemes.characteristic_reconstruction import _factor_report
    factor,report=tmp_path/'factors.txt',tmp_path/'build.json'
    assert _factor_report(factor,report) is None
    factor.write_text('partial or externally damaged factor cache')
    with pytest.raises(ValueError,match='missing completion report'):
        _factor_report(factor,report)


def test_interrupted_factor_publish_is_not_treated_as_complete(tmp_path,monkeypatch):
    from research.navier_stokes_schemes.characteristic_reconstruction import _commit_factors,_factor_report
    factor,report=tmp_path/'factors.txt',tmp_path/'build.json'
    original=Path.replace
    def interrupted(source,target):
        if Path(target)==factor:
            raise OSError('simulated interruption before factor completion marker')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',interrupted)
    with pytest.raises(OSError,match='simulated interruption'):
        _commit_factors(factor,report,['0 staged-LU'],{'ranks':8})
    assert report.exists()
    assert not factor.exists()
    assert _factor_report(factor,report) is None
