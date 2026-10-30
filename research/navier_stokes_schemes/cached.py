"""Reload this manufactured study's fixed maps without rebuilding weights."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from .problem import Problem


def load_problem(root, cells):
    root = Path(root)
    config = json.loads((root/'configuration.json').read_text(encoding='utf-8'))
    if cells not in config['cells'] or config['nu'] != 1 or config['stencil'] != 37:
        raise ValueError('Expected this study\'s viscosity-1, 37-site cached maps')
    expected = dict(augmentation='6/1', scaling='q=cells/44',
                    pressure='pure Gaussian cbar=1', local_digits=80,
                    global_dtype='float64', domain='unit square', offset=0,
                    boundary='homogeneous')
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Cached map configuration differs from this manufactured study')
    problem = Problem(cells=cells, nu=config['nu'], build_maps=False)
    folder = root/f'maps_cells{cells}'
    with np.load(folder/'nodes.npz', allow_pickle=False) as nodes:
        np.testing.assert_array_equal(nodes['points'], problem.points)
        np.testing.assert_array_equal(nodes['boundary_points'], problem.boundary_points)
    for name in ('M', 'A', 'B', 'SL', 'PY', 'PB', 'PL',
                 'DxU', 'DxB', 'DxL', 'DyU', 'DyB', 'DyL'):
        matrix = load_npz(folder/f'{name}.npz')
        boundary_block = name in {'B', 'PB', 'DxB', 'DyB'}
        columns = len(problem.boundary_points) if boundary_block else len(problem.points)
        if matrix.shape != (2*len(problem.points), 2*columns) or not np.isfinite(matrix.data).all():
            raise ValueError(f'Invalid cached map: {name}')
        setattr(problem, name, matrix)
    problem.linear_solver = 'gmres'
    return problem

