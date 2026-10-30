"""Independent PDE and overlap tests of a fixed 37-site mixed LHI ansatz.

All local maps use the public API. Auxiliary targets never become observations.
Small dense rank/least-squares audits are intentional; this is research code.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
import rbflab as r
from scipy.linalg import null_space, svd
from scipy.sparse import bmat, csr_matrix, eye, load_npz, save_npz, vstack
from scipy.spatial import cKDTree


GROUPS = ('velocity', 'pressure', 'boundary', 'momentum')
OPERATORS = ('velocity', 'pressure', 'momentum', 'gradient', 'dx', 'dy')


def build_maps(folder, cells=6, ranks=2):
    """Freeze each owner's source memberships while evaluating new locations."""
    folder = Path(folder)
    if (folder/'maps.json').exists():
        meta = json.loads((folder/'maps.json').read_text())
        if meta['cells'] != cells or meta['ranks'] != ranks:
            raise ValueError('Cached configuration differs')
        geometry = dict(np.load(folder/'geometry.npz'))
        maps = {op: {g: load_npz(folder/f'{op}_{g}.npz') for g in GROUPS}
                for op in OPERATORS}
        return maps, geometry, meta

    cloud = r.geometry.unit_box_grid(cells)
    nodes = cloud.interior
    boundary = cloud.points[cloud.boundary_indices]
    n, h = len(nodes), 1./cells
    stencil = r.Stencil(cloud.points, size=37)
    memberships = []
    for owner, point in enumerate(nodes):
        sites = {tuple(p) for p in cloud.points[stencil.select(point)]}
        inside = [i for i, p in enumerate(nodes) if tuple(p) in sites]
        wall = [i for i, p in enumerate(boundary) if tuple(p) in sites]
        without_owner = [i for i in inside if i != owner]
        memberships.append(dict(velocity=inside, pressure=without_owner,
                                boundary=wall, momentum=without_owner))

    targets, owners = list(nodes.copy()), list(range(n))
    auxiliary = []
    for owner, point in enumerate(nodes):
        for delta in ((.3, .2), (-.3, -.2), (.2, -.3), (-.2, .3)):
            auxiliary.append(len(targets))
            targets.append(point+h*np.array(delta))
            owners.append(owner)
    pairs = sorted(cKDTree(nodes).query_pairs(h*1.00001))
    overlap = []
    for first, second in pairs:
        point = (nodes[first]+nodes[second])/2
        overlap.append((len(targets), len(targets)+1))
        targets.extend((point, point))
        owners.extend((first, second))
    targets, owners = np.asarray(targets), np.asarray(owners)

    nu, q = sp.Rational(1, 100), sp.Rational(cells, 44)
    s = sp.Symbol('s', nonnegative=True)
    space = r.MixedSpace({
        'u': r.DivergenceFreeSpace(r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7),
                                  polynomial_degree=6),
        'p': r.ScalarSpace(r.Kernel(nu**2*q**4*sp.exp(-q*q*s), s)(), polynomial_degree=2),
    })
    u, p = space.fields
    momentum = -nu*r.laplacian(u)+r.gradient(p)
    fields = dict(velocity=u, pressure=p, boundary=u, momentum=momentum)
    source = {g: r.Samples(boundary if g == 'boundary' else nodes, operator=fields[g],
                          indices=[memberships[i][g] for i in owners]) for g in GROUPS}
    approximation = r.LocalApproximation(source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=80), backend=r.CppBackend(threads=1, timeout=3600),
        execution=r.LocalExecution(scheduler='mpi', ranks=ranks) if ranks else None)
    result = approximation.operators(targets=targets, operators=dict(
        velocity=u, pressure=p, momentum=momentum, gradient=r.gradient(p),
        dx=r.dx(u), dy=r.dy(u)))
    maps = {op: {g: result[op][g].to_scipy() for g in GROUPS} for op in OPERATORS}
    geometry = dict(nodes=nodes, boundary=boundary, targets=targets, owners=owners,
                    auxiliary=np.asarray(auxiliary), overlap=np.asarray(overlap))
    meta = dict(cells=cells, ranks=ranks, stencil=37, viscosity=.01, q=cells/44,
                augmentation=[6, 2], pressure_coefficient=1, local_digits=80,
                global_arithmetic='Float64', fixed_source_memberships=True,
                auxiliary_sites_are_observations=False,
                execution=result.diagnostics.get('execution'))
    folder.mkdir(parents=True, exist_ok=True)
    for op in OPERATORS:
        for g in GROUPS:
            save_npz(folder/f'{op}_{g}.npz', maps[op][g])
    np.savez(folder/'geometry.npz', **geometry)
    (folder/'maps.json').write_text(json.dumps(meta, indent=2))
    return maps, geometry, meta


def vector_rows(ids):
    return (2*np.asarray(ids)[:, None]+np.arange(2)).ravel()


def select(maps, op, ids):
    rows = np.asarray(ids) if op == 'pressure' else vector_rows(ids)
    return {g: maps[op][g][rows] for g in GROUPS}


def difference(maps, op, pairs):
    left, right = select(maps, op, pairs[:, 0]), select(maps, op, pairs[:, 1])
    return {g: left[g]-right[g] for g in GROUPS}


def evaluate(block, data):
    return sum(block[g]@np.asarray(data[g]).ravel() for g in GROUPS)


class Equations:
    def __init__(self, maps, geometry, dt, variant):
        self.maps, self.geo, self.dt = maps, geometry, dt
        self.n = len(geometry['nodes'])
        self.alpha = 0. if dt is None else 1./dt
        center = np.arange(self.n)
        self.L = select(maps, 'momentum', center)
        self.P = select(maps, 'pressure', center)
        self.M = eye(2*self.n)-self.L['momentum']
        a = self.alpha
        self.K = bmat([[self.L['velocity']+a*self.M, self.L['pressure']],
            [-self.P['velocity']+a*self.P['momentum'], eye(self.n)-self.P['pressure']]], format='csr')
        self.extras = []
        if variant in ('pde2', 'pde4', 'pde4_velocity', 'pde4_velocity_pressure', 'pde4_traction'):
            ids = geometry['auxiliary'].reshape(self.n, 4)
            ids = ids[:, :2].ravel() if variant == 'pde2' else ids.ravel()
            v, l = select(maps, 'velocity', ids), select(maps, 'momentum', ids)
            block = {g: l[g]+a*v[g] for g in GROUPS}
            self.extras.append(('pde', block, ids, v))
        if variant in ('velocity', 'pde4_velocity', 'pde4_velocity_pressure'):
            self.extras.append(('overlap', difference(maps, 'velocity', geometry['overlap']), None, None))
        if variant in ('pressure', 'pde4_velocity_pressure'):
            self.extras.append(('overlap', difference(maps, 'pressure', geometry['overlap']), None, None))
        if variant in ('traction', 'pde4_traction'):
            # Same normal on both patches. Interior grid neighbors determine it.
            pairs = geometry['overlap']
            owner = geometry['owners']
            delta = geometry['nodes'][owner[pairs[:, 1]]]-geometry['nodes'][owner[pairs[:, 0]]]
            normal = delta/np.linalg.norm(delta, axis=1)[:, None]
            dx, dy = difference(maps, 'dx', pairs), difference(maps, 'dy', pairs)
            pressure = difference(maps, 'pressure', pairs)
            block = {}
            for g in GROUPS:
                stress_xx = .02*dx[g][0::2]-pressure[g]
                stress_yy = .02*dy[g][1::2]-pressure[g]
                stress_xy = .01*(dy[g][0::2]+dx[g][1::2])
                tx = stress_xx.multiply(normal[:, 0, None])+stress_xy.multiply(normal[:, 1, None])
                ty = stress_xy.multiply(normal[:, 0, None])+stress_yy.multiply(normal[:, 1, None])
                stacked = vstack([tx, ty], format='csr')
                indices = np.column_stack((np.arange(len(pairs)), len(pairs)+np.arange(len(pairs)))).ravel()
                block[g] = stacked[indices]
            self.extras.append(('overlap', block, None, None))
        self.H = vstack([bmat([[b['velocity']-a*b['momentum'], b['pressure']]], format='csr')
                         for _, b, _, _ in self.extras], format='csr') if self.extras else csr_matrix((0, 3*self.n))
        self.matrix = vstack([self.K, self.H], format='csr')

    def rhs(self, case, time, previous):
        fields = exact_fields(case, self.geo['nodes'], time)
        wall = exact_fields(case, self.geo['boundary'], time)['velocity'].ravel()
        force = fields['momentum'].ravel() if self.dt is None else fields['force'].ravel()
        rvalue = force+self.alpha*previous['velocity'].ravel()
        rhs = [self.M@rvalue-self.L['boundary']@wall,
               self.P['boundary']@wall+self.P['momentum']@rvalue]
        for kind, block, ids, old_velocity in self.extras:
            value = -block['boundary']@wall-block['momentum']@rvalue
            if kind == 'pde':
                test = exact_fields(case, self.geo['targets'][ids], time)
                value += test['momentum'].ravel() if self.dt is None else test['force'].ravel()
                if self.dt is not None:
                    value += self.alpha*evaluate(old_velocity, previous)
            rhs.append(value)
        return np.concatenate(rhs), fields, wall, rvalue


class LeastSquares:
    """Mean-zero pressure, explicit row weights, column scaling, no regularizer."""
    def __init__(self, equations):
        n = equations.n
        self.matrix = equations.matrix.toarray()
        # Mean-zero pressure parameterization; no equation is silently discarded.
        pressure_basis = null_space(np.ones((1, n)))
        self.basis = np.zeros((3*n, 3*n-1))
        self.basis[:2*n, :2*n] = np.eye(2*n)
        self.basis[2*n:, 2*n:] = pressure_basis
        self.row_scale = 1./np.maximum(np.sum(abs(self.matrix), axis=1), np.finfo(float).tiny)
        reduced = (self.matrix*self.row_scale[:, None])@self.basis
        self.column_scale = 1./np.maximum(np.linalg.norm(reduced, axis=0), np.finfo(float).tiny)
        self.left, self.singular, self.right = svd(reduced*self.column_scale, full_matrices=False)
        threshold = self.singular[0]*max(reduced.shape)*np.finfo(float).eps
        self.rank = int(np.count_nonzero(self.singular > threshold))
        self.full_rank = self.rank == reduced.shape[1]
        base = equations.K.toarray()
        base /= np.maximum(np.sum(abs(base), axis=1), np.finfo(float).tiny)[:, None]
        _, values, vectors = svd(base@self.basis, full_matrices=False)
        weak = self.basis@vectors[values < values[0]*max(base.shape)*np.finfo(float).eps].T
        extra = equations.H.toarray()*self.row_scale[len(base):, None]
        detected = np.linalg.svd(extra@weak, compute_uv=False) if len(extra) else np.array([])
        self.diagnostics = dict(rows=reduced.shape[0], gauge_fixed_unknowns=reduced.shape[1],
            rank=self.rank, full_rank=self.full_rank, threshold=float(threshold),
            smallest_singular=float(self.singular[-1]),
            condition=float(self.singular[0]/self.singular[-1]) if self.singular[-1] else float('inf'),
            base_weak_dimension=weak.shape[1], extra_on_weak_singular_values=detected.tolist(),
            weighting='inverse full row 1-norm; inverse reduced column 2-norm variable scaling')

    def solve(self, rhs):
        if not self.full_rank:
            raise ValueError('Underresolved numerical rank; refusing pseudoinverse solution')
        coeff = self.right.T@((self.left.T@(rhs*self.row_scale))/self.singular)
        return self.basis@(self.column_scale*coeff)


_smooth = None


def exact_fields(case, nodes, time):
    global _smooth
    x, y = nodes.T
    if case == 'pressure_xy':
        velocity = np.zeros_like(nodes)
        momentum = nodes[:, ::-1].copy()
        return dict(velocity=velocity, pressure=x*y, momentum=momentum, force=momentum,
                    gradient=nodes[:, ::-1].copy())
    if case == 'polynomial':
        base = np.column_stack((x*x, -2*x*y))
        velocity = (1+time)*base
        momentum = np.column_stack((y-.02*(1+time), x))
        return dict(velocity=velocity, pressure=x*y, momentum=momentum,
                    force=momentum+base, gradient=nodes[:, ::-1].copy())
    if _smooth is None:
        sx, sy, t = sp.symbols('x y t')
        psi = sp.sin(sp.pi*sx)**2*sp.sin(sp.pi*sy)**2
        u = sp.Matrix([sp.diff(psi, sy), -sp.diff(psi, sx)])*(1+t)
        p = sp.sin(sp.pi*sx)*sp.cos(sp.pi*sy)
        grad = sp.Matrix([sp.diff(p, sx), sp.diff(p, sy)])
        ell = -sp.Rational(1,100)*(u.diff(sx,2)+u.diff(sy,2))+grad
        _smooth = {name: sp.lambdify((sx, sy, t), list(value), 'numpy')
                   for name, value in dict(velocity=u, pressure=[p], momentum=ell,
                                          force=ell+u.diff(t), gradient=grad).items()}
    result = {name: np.column_stack(fun(x, y, time)) for name, fun in _smooth.items()}
    result['pressure'] = result['pressure'].ravel()
    return result


def run(folder, cells=6, ranks=2, final_time=.01):
    folder = Path(folder)
    maps, geo, meta = build_maps(folder/'weights', cells, ranks)
    records = []
    variants = ('baseline', 'pde2', 'pde4', 'velocity', 'pressure', 'traction',
                'pde4_velocity', 'pde4_velocity_pressure', 'pde4_traction')
    n = len(geo['nodes'])
    velocity_jump = difference(maps, 'velocity', geo['overlap'])
    for dt in (None, .00025, .000125):
        for variant in variants:
            equations = Equations(maps, geo, dt, variant)
            factor = LeastSquares(equations)
            for case in ('pressure_xy', 'polynomial', 'smooth'):
                initial = exact_fields(case, geo['nodes'], 0.)
                previous = dict(velocity=initial['velocity'],
                    pressure=initial['pressure']-initial['pressure'].mean(),
                    boundary=exact_fields(case, geo['boundary'], 0.)['velocity'],
                    momentum=initial['momentum'])
                record = dict(factor.diagnostics, dt=dt, variant=variant, case=case)
                time = 0. if dt is None else dt
                rhs, fields, wall, rvalue = equations.rhs(case, time, previous)
                exact = np.r_[fields['velocity'].ravel(), fields['pressure']-fields['pressure'].mean()]
                defect = equations.matrix@exact-rhs
                record['exact_scaled_residual'] = float(np.max(abs(defect*factor.row_scale)))
                if not factor.full_rank:
                    record['status'] = 'rank_deficient'
                else:
                    count = 1 if dt is None else round(final_time/dt)
                    maximum_velocity = 0.
                    for step in range(1, count+1):
                        time = 0. if dt is None else step*dt
                        rhs, fields, wall, rvalue = equations.rhs(case, time, previous)
                        values = factor.solve(rhs)
                        finite = bool(np.isfinite(values).all())
                        velocity = values[:2*n].reshape(n, 2)
                        maximum_velocity = max(maximum_velocity, float(np.max(abs(velocity))))
                        previous = dict(velocity=velocity, pressure=values[2*n:], boundary=wall,
                                        momentum=(rvalue-equations.alpha*velocity.ravel()).reshape(n, 2))
                        if not finite or maximum_velocity > 1e6:
                            break
                    gradient = evaluate(select(maps, 'gradient', np.arange(n)), previous).reshape(n, 2)
                    residual = equations.matrix@values-rhs
                    status = ('nonfinite_abort' if not finite else
                              'solved' if maximum_velocity <= 1e6 else 'growth_abort')
                    record.update(status=status,
                        steps=step, final_time=time, maximum_velocity=maximum_velocity,
                        velocity_max=float(np.max(abs(velocity-fields['velocity']))),
                        velocity_rms=float(np.sqrt(np.mean(np.sum((velocity-fields['velocity'])**2, axis=1)))),
                        pressure_max=float(np.max(abs(values[2*n:]-(fields['pressure']-fields['pressure'].mean())))),
                        gradient_max=float(np.max(abs(gradient-fields['gradient']))),
                        scaled_residual=float(np.max(abs(residual*factor.row_scale))),
                        base_residual=float(np.max(abs(residual[:3*n]))),
                        extra_residual=float(np.max(abs(residual[3*n:]))) if len(residual)>3*n else 0.,
                        overlap_velocity_jump=float(np.max(abs(evaluate(velocity_jump, previous)))))
                records.append(record)
                print(cells, dt, variant, case, record['status'], record.get('velocity_max'), flush=True)
            (folder/'results.json').write_text(json.dumps(dict(configuration=meta,
                error_scope='interior mesh nodes; overlap jump is a separate diagnostic', results=records), indent=2))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=6)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output/'results.json').exists():
        parser.error('Results already exist; choose another output directory')
    run(args.output, args.cells, args.ranks)
