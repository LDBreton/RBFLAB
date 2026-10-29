"""Public examples reproduce fields and use the documented local maps."""
import os

import numpy as np
import pytest

from examples import mixed_local_maps, stokes_lhi_weights


def test_mixed_example_reconstructs_fields_and_derivatives():
    errors = mixed_local_maps.run()
    assert errors['field_max_absolute_nodal_error'] < 1e-10
    assert errors['equation_max_absolute_nodal_error'] < 1e-10


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1', reason='Opt-in real MPI example')
def test_stokes_weights_example_reproduces_affine_pressure():
    maps = stokes_lhi_weights.run(backend='cpp', digits=80, ranks=2)
    points = maps.gradient.input_layout['velocity']['points']
    data = {'velocity': np.c_[points[:, 1], -points[:, 0]],
            'momentum': np.tile([1., -1.], (len(points), 1))}
    np.testing.assert_allclose(maps.gradient @ data, [[1., -1.]], atol=1e-12)
    np.testing.assert_allclose(maps.momentum @ data, [[1., -1.]], atol=1e-12)
    assert maps.diagnostics['execution']['scheduler'] == 'mpi'
