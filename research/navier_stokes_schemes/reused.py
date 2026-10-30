"""Reuse only exactly translated/permuted Hermite problems; never round nodes.

The geometry routine is the one validated against the saved Stokes baseline.
All weight construction still uses public LocalApproximation APIs.
"""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix, eye

import rbflab as r
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry


def assemble(problem, cloud, owners=None):
    unique, records, selection = geometry(cloud)
    owners = list(range(len(records))) if owners is None else list(owners)
    needed = sorted({records[i]['job'] for i in owners})
    folder = Path(problem.weight_folder) if problem.weight_folder else None
    recipe = dict(version=1, cells=problem.cells, nu=problem.nu, digits=problem.digits,
                  stencil=37, augmentation=[6, 1], q=f'{problem.cells}/44',
                  pressure='q**4*exp(-q*q*r*r)',
                  velocity='exp(-10*q*q*r*r)+q**7*r**7',
                  cloud_sha256=hashlib.sha256(cloud.points.tobytes()).hexdigest(),
                  jobs=needed, outputs=['L', 'P', 'Dx', 'Dy'])
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder/'recipe.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != recipe:
            raise ValueError('Weight checkpoint recipe differs from this problem')
        manifest.write_text(json.dumps(recipe, indent=2), encoding='utf-8')

    momentum = -problem.nu*r.laplacian(problem.u)+r.gradient(problem.p)
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
