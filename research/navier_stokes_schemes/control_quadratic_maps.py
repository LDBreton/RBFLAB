"""True quadratic-pressure mixed LHI maps, constructed through the public API.

This is a fresh polynomial-degree-2 kernel saddle solve, not a correction of
the degree-1 maps. Only exactly translated/permuted local problems are reused.
"""
import gc
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sympy as sp
from scipy.sparse import coo_matrix, eye

import rbflab as r
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits
from .cavity import Cavity
from .control_maps import NAMES


def make_problem(cells, ranks=8):
    """The degree-6/2 variant of the fixed viscosity-normalized baseline."""
    problem = Cavity(cells=cells, nu=.01, ranks=ranks, digits=80,
                     native_timeout=3600, build_maps=False)
    q = sp.Rational(cells, 44)
    viscosity = sp.Rational(1, 100)
    s = sp.Symbol('s', nonnegative=True)
    problem.space = r.MixedSpace({
        'u': r.DivergenceFreeSpace(r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), 6),
        'p': r.PressureSpace(r.Kernel(viscosity**2*q**4*sp.exp(-q*q*s), s)(), 2),
    })
    problem.u, problem.p = problem.space.fields
    momentum = -viscosity*r.laplacian(problem.u)+r.gradient(problem.p)
    problem.source = {
        'velocity': r.Samples(problem.points, operator=problem.u, within=problem.stencil),
        'boundary': r.Samples(problem.boundary_points, operator=problem.u, within=problem.stencil),
        'momentum': r.Samples(problem.points, operator=momentum,
                              within=problem.stencil, target='exclude'),
    }
    return problem


def assemble_quadratic(problem, cloud, owners=None):
    unique, records, selection = geometry(cloud)
    owners = list(range(len(records))) if owners is None else list(owners)
    needed = sorted({records[i]['job'] for i in owners})
    folder = Path(problem.weight_folder) if problem.weight_folder else None
    recipe = dict(version=2, cells=problem.cells, nu=problem.nu, digits=problem.digits,
                  stencil=37, augmentation=[6, 2], q=f'{problem.cells}/44',
                  pressure='nu**2*q**4*exp(-q*q*r*r)',
                  velocity='exp(-10*q*q*r*r)+q**7*r**7',
                  cloud_sha256=hashlib.sha256(cloud.points.tobytes()).hexdigest(),
                  jobs=needed, outputs=['L', 'P', 'Dx', 'Dy'])
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder/'recipe.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != recipe:
            raise ValueError('Weight checkpoint recipe differs from this problem')
        manifest.write_text(json.dumps(recipe, indent=2), encoding='utf-8')

    momentum = -sp.Rational(str(problem.nu))*r.laplacian(problem.u)+r.gradient(problem.p)
    observed = [problem.u[0], problem.u[1], momentum[0], momentum[1]]
    targets = dict(L=momentum, P=r.gradient(problem.p),
                   Dx=r.dx(problem.u), Dy=r.dy(problem.u))
    weights, batches = {}, []
    for start in range(0, len(needed), 64):
        ids = needed[start:start+64]
        checkpoint = folder/f'batch_{start:05d}.npz' if folder else None
        if checkpoint and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                for job in ids:
                    value = saved[f'w{job}']
                    if value.shape != (len(unique[job]['points']), 8) or not np.isfinite(value).all():
                        raise ValueError(f'Invalid weight checkpoint: {checkpoint}, job {job}')
                    weights[job] = value
            print(f'Reused checkpoint: {start+len(ids)}/{len(needed)} templates', flush=True)
            continue
        before = time.perf_counter()
        jobs = [unique[i] for i in ids]
        lengths = [len(job['points']) for job in jobs]
        offsets = np.r_[0, np.cumsum(lengths)]
        source = {'observations': r.Samples(
            np.concatenate([job['points'] for job in jobs]),
            operator=[observed[code] for job in jobs for code in job['codes']],
            indices=[np.arange(offsets[i], offsets[i+1]) for i in range(len(jobs))])}
        local = r.LocalApproximation(source=source,
                                     trial=problem.space.representers(source), **problem.options)
        maps = local.operators(targets=np.array([job['target'] for job in jobs]), operators=targets)
        for i, job in enumerate(ids):
            weights[job] = np.column_stack([
                np.asarray(maps[name].local(i).weights.tolist(), dtype=float) for name in targets])
            if not np.isfinite(weights[job]).all():
                raise FloatingPointError('Nonfinite template weight')
        report = dict(jobs=ids, seconds=time.perf_counter()-before,
                      execution=maps.diagnostics['execution'])
        batches.append(report)
        if checkpoint:
            pending = checkpoint.with_name(checkpoint.stem+'.pending.npz')
            np.savez_compressed(pending, **{f'w{job}': weights[job] for job in ids})
            pending.replace(checkpoint)
            checkpoint.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'WEIGHTS {start+len(ids)}/{len(needed)} templates, {report["seconds"]:.1f}s', flush=True)
        del maps, local, source
        gc.collect()

    ni, nb = len(cloud.interior), len(cloud.boundary_indices)
    names = dict(L=('A', 'B', 'SL'), P=('PY', 'PB', 'PL'),
                 Dx=('DxU', 'DxB', 'DxL'), Dy=('DyU', 'DyB', 'DyL'))
    entries = {(name, group): ([], [], []) for name in targets for group in range(3)}
    for owner in owners:
        record = records[owner]
        w = weights[record['job']]
        for group, size in enumerate((ni, nb, ni)):
            mask = record['groups'] == group
            if not np.any(mask):
                continue
            # Historical records use component-major DOFs. Export node-major.
            columns = record['columns'][mask]
            columns = 2*(columns % size)+columns//size
            for output, name in enumerate(targets):
                rr, cc, vv = entries[name, group]
                for component in range(2):
                    rr.append(np.full(len(columns), 2*owner+component))
                    cc.append(columns)
                    vv.append(w[mask, 2*output+component])
    blocks = {}
    for (name, group), (rr, cc, vv) in entries.items():
        shape = (2*ni, 2*(nb if group == 1 else ni))
        if vv:
            matrix = coo_matrix((np.concatenate(vv),
                                 (np.concatenate(rr), np.concatenate(cc))), shape=shape).tocsr()
        else:
            matrix = coo_matrix(shape).tocsr()
        blocks[names[name][group]] = matrix
    blocks['M'] = eye(2*ni, format='csr')-blocks['SL']
    computed = sum(len(batch['jobs']) for batch in batches)
    info = dict(selection=selection, requested_templates=len(needed),
                computed_templates=computed, reused_templates=len(needed)-computed, batches=batches,
                scalar_target_rows=8, owners=len(owners))
    if folder:
        (folder/'assembly.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return blocks, info


def pressure_checks(problem, folder):
    """Insert exact zero velocity and five represented pressure fields."""
    x, y = problem.points.T
    fields = {
        'x-y': np.column_stack((np.ones_like(x), -np.ones_like(y))).ravel(),
        'x2': np.column_stack((2*x, np.zeros_like(y))).ravel(),
        'xy': np.column_stack((y, x)).ravel(),
        'y2': np.column_stack((np.zeros_like(x), 2*y)).ravel(),
    }
    result = {}
    for name, gradient in fields.items():
        result[name] = dict(
            constraint_inf=float(np.max(abs(problem.M@gradient))),
            gradient_error_inf=float(np.max(abs(problem.PL@gradient-gradient))),
            dx_velocity_inf=float(np.max(abs(problem.DxL@gradient))),
            dy_velocity_inf=float(np.max(abs(problem.DyL@gradient))))
        if max(result[name].values()) > 1e-8:
            raise AssertionError(f'Quadratic pressure reproduction failed: {name}: {result[name]}')
    (Path(folder)/'pressure_patch.json').write_text(json.dumps(result, indent=2))
    return result


def build(cells, folder, ranks=8):
    """Build all actual local weights with MPI; retain resumable batches."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(cells=cells, total_nodes=(cells+1)**2, interior_nodes=(cells-1)**2,
        nu=.01, stencil=37, augmentation=[6, 2], q=f'{cells}/44',
        velocity='exp(-10*q^2*r^2)+q^7*r^7',
        pressure='nu^2*q^4*exp(-q^2*r^2)', pressure_amplitude='nu^2*q^4',
        pressure_cbar=1, precision_digits=80, ranks=ranks,
        build='fresh public LocalApproximation C++/MPFR80/MPI weights with exact translation reuse',
        corner_subtraction=False, pressure_correction=False,
        boundary='unit lid; corner nodes and other walls stationary')
    manifest = folder/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Existing map configuration differs')
    manifest.write_text(json.dumps(config, indent=2))
    p = make_problem(cells, ranks)
    p.weight_folder = folder/'weights'
    started = time.perf_counter()
    blocks, execution = assemble_quadratic(p, r.geometry.unit_box_grid(cells))
    for name, matrix in blocks.items():
        setattr(p, name, matrix)
        save_npz(folder/f'{name}.npz', matrix)
    np.savez(folder/'nodes.npz', points=p.points, boundary_points=p.boundary_points)
    hashes = {name: hashlib.sha256((folder/f'{name}.npz').read_bytes()).hexdigest()
              for name in NAMES}
    report = dict(seconds=time.perf_counter()-started, execution=execution, hashes=hashes,
                  augmentation=[6, 2], polynomial_columns=40)
    (folder/'assembly.json').write_text(json.dumps(report, indent=2))
    pressure = pressure_checks(p, folder)
    print(json.dumps(dict(event='maps_ready', cells=cells, seconds=report['seconds'],
                          pressure=pressure)), flush=True)
    if cells <= 20:
        from .control_maps import audit
        result = audit(p, folder)
        print(json.dumps(dict(event='small_spectrum', cells=cells,
                              spectrum=result['stokes_spectrum'])), flush=True)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        build(args.cells, args.output, args.ranks)
