"""Compare exact-template maps with original public maps on selected owners."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import eye
from threadpoolctl import threadpool_limits

import rbflab as r
from .problem import Problem
from .reused import assemble, geometry


BLOCKS = dict(L=('A', 'B', 'SL'), P=('PY', 'PB', 'PL'),
              Dx=('DxU', 'DxB', 'DxL'), Dy=('DyU', 'DyB', 'DyL'))
GROUPS = ('velocity', 'boundary', 'momentum')


def selected_owners(problem):
    """All small-grid owners; six geometrically different large-grid owners."""
    if problem.cells <= 6:
        return np.arange(len(problem.points))
    h = 1 / problem.cells
    probes = np.array([(h, h), (h, .5), (3*h, .5), (.5, .5),
                       (1-h, 1-h), (.75, 1-h)])
    owners = np.array([np.argmin(np.sum((problem.points-p)**2, axis=1)) for p in probes])
    assert len(np.unique(owners)) == len(probes)
    return owners


def validate_case(cells, ranks=2, duplicate_pair=False):
    before = time.perf_counter()
    problem = Problem(cells=cells, ranks=ranks, build_maps=False)
    cloud = r.geometry.unit_box_grid(cells)
    if duplicate_pair:
        _, records, _ = geometry(cloud)
        by_job = {}
        for owner, record in enumerate(records):
            by_job.setdefault(record['job'], []).append(owner)
        candidates = [ids for ids in by_job.values() if len(ids) >= 2]
        if not candidates:
            raise ValueError('This grid has no exactly repeated template')
        center_pair = min(candidates, key=lambda ids: np.linalg.norm(problem.points[ids[0]]-.5))
        owners = np.asarray(center_pair[:2])
        del records, by_job, candidates
    else:
        owners = selected_owners(problem)
    row_ids = (2*owners[:, None]+np.arange(2)).ravel()
    momentum = -problem.nu*r.laplacian(problem.u)+r.gradient(problem.p)
    targets = dict(L=momentum, P=r.gradient(problem.p),
                   Dx=r.dx(problem.u), Dy=r.dy(problem.u))
    print(f'REFERENCE cells={cells}, owners={len(owners)}', flush=True)
    local = r.LocalApproximation(source=problem.source,
                                trial=problem.space.representers(problem.source),
                                **problem.options)
    reference_start = time.perf_counter()
    maps = local.operators(targets=problem.points[owners], operators=targets)
    reference_seconds = time.perf_counter()-reference_start
    reference = {label: maps[name][group].to_scipy()
                 for name, labels in BLOCKS.items() for label, group in zip(labels, GROUPS)}
    reference_execution = maps.diagnostics['execution']
    del maps, local
    gc.collect()
    reuse_start = time.perf_counter()
    actual, info = assemble(problem, cloud, owners=owners)
    reuse_seconds = time.perf_counter()-reuse_start
    if duplicate_pair:
        assert info['computed_templates'] == 1 and info['owners'] == 2
    checks = {}
    for name, expected in reference.items():
        observed = actual[name][row_ids]
        assert observed.shape == expected.shape
        difference = (observed-expected).tocsr()
        absolute = float(np.max(np.abs(difference.data), initial=0.))
        scale = float(np.max(np.abs(expected.data), initial=0.))
        # Small selected row counts make this dense check bounded even on the
        # large grids, and enforce entrywise rather than normwise tolerance.
        np.testing.assert_allclose(observed.toarray(), expected.toarray(),
                                   rtol=1e-12, atol=1e-9,
                                   err_msg=f'cells={cells}, block={name}')
        checks[name] = dict(max_absolute_difference=absolute,
                           reference_max_absolute=scale,
                           relative_to_block_max=absolute/scale if scale else 0.,
                           reference_nnz=int(expected.nnz), actual_nnz=int(observed.nnz))
    mass_difference = actual['M']-(eye(2*len(problem.points), format='csr')-actual['SL'])
    assert np.max(np.abs(mass_difference.data), initial=0.) == 0.
    row = dict(cells=cells, nodes=len(cloud.points), interior_nodes=len(problem.points),
               owner_indices=owners.tolist(), owner_coordinates=problem.points[owners].tolist(),
               coverage=('two distinct owners sharing one exact template' if duplicate_pair else
                         'all interior owners' if cells <= 6 else 'six selected owners; not a whole-grid check'),
               local_digits=problem.digits, ranks=ranks, scalar_target_rows=8,
               kernels='current Problem.space shared unchanged by reference and reused requests',
               reference_seconds=reference_seconds, reuse_seconds=reuse_seconds,
               total_seconds=time.perf_counter()-before, reference_execution=reference_execution,
               reuse=info, blocks=checks, status='passed')
    print(f'PARITY cells={cells}: maxabs={max(c["max_absolute_difference"] for c in checks.values()):.3e}, '
          f'templates={info["computed_templates"]}, total={row["total_seconds"]:.1f}s', flush=True)
    return row


def run(output, cells=(6, 89, 126), ranks=2, duplicate_pair=False):
    path = Path(output)
    if path.exists():
        raise FileExistsError(f'Choose a new validation output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', scope='map parity only; no evolution/stability claim',
                  cells=list(cells), ranks=ranks, duplicate_pair=duplicate_pair, cases=[], source_sha256={})
    for filename in ('problem.py', 'reused.py', 'validate_reuse.py'):
        source = Path(__file__).with_name(filename)
        report['source_sha256'][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        for count in cells:
            report['cases'].append(validate_case(count, ranks, duplicate_pair=duplicate_pair))
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New JSON file, never overwritten')
    parser.add_argument('--cells', nargs='+', type=int, default=[6, 89, 126])
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--duplicate-pair', action='store_true',
                        help='Compare two owners that must share exactly one template')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.cells, args.ranks, args.duplicate_pair)
