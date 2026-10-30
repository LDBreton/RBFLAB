"""Native opt-in regression for geometry reuse and node-major map scatter."""
import os

import pytest
from threadpoolctl import threadpool_limits

from research.navier_stokes_schemes.validate_reuse import validate_case


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1',
                    reason='set RBFLAB_MPI_TESTS=1 for real MPI/MPFR map comparisons')
@pytest.mark.parametrize('cells', [6, 89, 126])
def test_exact_template_map_parity(cells):
    with threadpool_limits(limits=1):
        result = validate_case(cells, ranks=2)
    assert result['status'] == 'passed'
    assert result['reuse']['owners'] == len(result['owner_indices'])
    assert result['reuse']['scalar_target_rows'] == 8
    assert set(result['blocks']) == {'A', 'B', 'SL', 'PY', 'PB', 'PL',
                                     'DxU', 'DxB', 'DxL', 'DyU', 'DyB', 'DyL'}


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1',
                    reason='set RBFLAB_MPI_TESTS=1 for real MPI/MPFR map comparisons')
def test_two_owners_reuse_one_template():
    with threadpool_limits(limits=1):
        result = validate_case(89, ranks=2, duplicate_pair=True)
    assert result['reuse']['computed_templates'] == 1
    assert result['reuse']['owners'] == 2


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1',
                    reason='set RBFLAB_MPI_TESTS=1 for real MPI/MPFR checkpoints')
def test_weight_checkpoint_roundtrip_and_corruption(tmp_path):
    import numpy as np
    import rbflab as r
    from research.navier_stokes_schemes.problem import Problem
    from research.navier_stokes_schemes.reused import assemble

    p = Problem(cells=6, ranks=2, build_maps=False, weight_folder=tmp_path)
    cloud = r.geometry.unit_box_grid(6)
    with threadpool_limits(limits=1):
        original, first = assemble(p, cloud, owners=[0])
        loaded, second = assemble(p, cloud, owners=[0])
    assert first['computed_templates'] == 1
    assert second['computed_templates'] == 0 and second['reused_templates'] == 1
    for name in original:
        assert np.max(abs((original[name]-loaded[name]).data), initial=0.) == 0.
    checkpoint = tmp_path/'batch_00000.npz'
    with np.load(checkpoint, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    values[next(iter(values))][0, 0] = np.nan
    np.savez_compressed(checkpoint, **values)
    with pytest.raises(ValueError, match='Invalid weight checkpoint'):
        assemble(p, cloud, owners=[0])
