"""Small dense, differentiable steady Stokes reference on fixed stencils."""
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
from .torch_backend import _torch, _block, TorchKernel, TorchBackend, solve_weights


class DifferentiableLHI:
    """Kernel-parameter sensitivities; CPU Float64, steady 2D, <=500 nodes.

    Geometry, boundary/forcing data, viscosity and nearest-stencil ownership
    are fixed. The normal production backend remains unchanged.
    """
    def __init__(self, method, problem, cloud):
        from .spaces import SpaceStokesProblem
        from .space_stokes import select_spaces
        from .lhi_backends import _prepare
        if not isinstance(problem, SpaceStokesProblem):
            raise TypeError('Expected a symbolic Stokes problem with velocity/pressure spaces')
        if problem.transient or cloud.dimension != 2:
            raise NotImplementedError('Differentiable reference requires steady 2D Stokes')
        if len(cloud.points) > 500:
            raise ValueError('Dense differentiable reference is limited to 500 nodes')
        if not isinstance(method.local_backend, TorchBackend) or method.local_backend.shape_rule != 'fixed':
            raise ValueError('Use TorchBackend with fixed shape parameters')
        velocity, pressure, vd, pd = select_spaces(method, problem)
        if vd is not None or pd is not None:
            raise NotImplementedError('Differentiable reference requires no polynomial tails')
        # Reuse production validation, geometry and data evaluation once.
        self.reference = method.assemble(problem, cloud)
        base = self.reference.base
        native = SimpleNamespace(kernel=velocity.kernel, pressure_kernel=pressure.kernel,
            precision=method.precision, stencil_policy=method.stencil_policy,
            polynomial_degree=None, stencil_size=method.stencil_size,
            pde_stencil_size=method.pde_stencil_size, min_boundary_centers=method.min_boundary_centers)
        _, self.tasks, self.stencils = _prepare(native, base.problem, cloud, 'fixed')
        self.velocity_kernel = TorchKernel(velocity.kernel).prepare(2, 6)
        self.pressure_kernel = TorchKernel(pressure.kernel).prepare(2, 2)
        self.cloud = cloud
        self.ni, self.nb = len(cloud.interior), len(cloud.boundary_indices)
        torch = _torch()
        f, g = base._data(0)
        self.forcing = torch.tensor(np.asarray(f), dtype=torch.float64, device='cpu')
        self.boundary = torch.tensor(np.asarray(g), dtype=torch.float64, device='cpu')
        self.batch_size = method.local_backend.batch_size

    @property
    def parameter_names(self):
        return {name: tuple(p.name for p in kernel.bound.family.parameters)
                for name, kernel in [('velocity', self.velocity_kernel), ('pressure', self.pressure_kernel)]}

    def solve(self, velocity_parameters, pressure_parameters):
        torch = _torch()
        params = []
        for values, kernel in [(velocity_parameters, self.velocity_kernel), (pressure_parameters, self.pressure_kernel)]:
            if not isinstance(values, torch.Tensor) or values.dtype != torch.float64 or values.device.type != 'cpu':
                raise ValueError('Supply CPU Float64 parameter tensors')
            if values.shape != (len(kernel.bound.values),):
                raise ValueError('Supply one parameter vector per kernel family')
            params.append(values)
        vp, pp = params
        groups = defaultdict(list)
        for i, task in enumerate(self.tasks): groups[len(task['points'])].append(i)
        grams, weights = {}, {}
        residuals = []
        for ids in groups.values():
            for offset in range(0, len(ids), self.batch_size):
                selected = ids[offset:offset+self.batch_size]
                tasks = [self.tasks[i] for i in selected]
                points = torch.tensor(np.asarray([t['points'] for t in tasks]), dtype=torch.float64, device='cpu')
                codes = torch.tensor([t['codes'] for t in tasks],device='cpu')
                mu = torch.tensor([float(t['mu']) for t in tasks], dtype=torch.float64, device='cpu')[:,None,None]
                G = _block(points[:,:,None,:]-points[:,None,:,:], codes[:,None,:], codes[:,:,None],
                           self.velocity_kernel, self.pressure_kernel, vp, pp, mu)
                G = G.tril()+G.tril(-1).transpose(-1,-2)
                origin = torch.tensor(np.asarray([t['origin'] for t in tasks]), dtype=torch.float64, device='cpu')
                z = (origin[:,None,None,:]-points[:,None,:,:]).expand(-1,4,-1,-1)
                Q = _block(z, codes[:,None,:], torch.arange(5,9,device='cpu')[None,:,None],
                           self.velocity_kernel, self.pressure_kernel, vp, pp, mu)
                W, residual, _ = solve_weights(G, Q.transpose(-1,-2))
                residuals.append(residual)
                for j,i in enumerate(selected): grams[i], weights[i] = G[j], W[j]
        maps = {}
        for group, count in [('solution',self.ni), ('boundary',self.nb), ('pde',self.ni)]:
            row_indices, columns, values = [], [], []
            for i, stencil in enumerate(self.stencils):
                positions = [j for j,(g,_) in enumerate(stencil.slots) if g == group]
                if not positions: continue
                cols = [stencil.slots[j][1] for j in positions]
                for k in range(4):
                    row_indices.extend([k*self.ni+i]*len(cols)); columns.extend(cols)
                    values.append(weights[i][positions,k])
            matrix = torch.zeros((4*self.ni,2*count), dtype=torch.float64, device='cpu')
            if values:
                matrix = matrix.index_put((torch.tensor(row_indices,device='cpu'),torch.tensor(columns,device='cpu')), torch.cat(values), accumulate=True)
            maps[group] = matrix
        A = maps['solution'][:2*self.ni]
        rhs = self.forcing-maps['pde'][:2*self.ni]@self.forcing-maps['boundary'][:2*self.ni]@self.boundary
        u = torch.linalg.solve(A, rhs)
        if not torch.isfinite(u).all(): raise FloatingPointError('Nonfinite global solution')
        return DifferentiableFlow(self, vp, pp, grams, weights, maps, A, rhs, u, torch.cat(residuals))


class DifferentiableFlow:
    def __init__(self, owner, vp, pp, grams, weights, maps, matrix, rhs, u, residuals):
        self.owner, self.vp, self.pp = owner, vp, pp
        self.grams, self.weights, self.maps = grams, weights, maps
        self.matrix, self.rhs, self.nodal_velocity = matrix, rhs, u
        self.local_residuals = residuals

    def fields(self, points):
        """Return velocity and direct pressure gradient at fixed query points.

        Uses the existing nearest-stencil reconstruction, including the nodal
        velocity shortcut. Query coordinates are not differentiable variables.
        """
        from scipy.spatial import cKDTree
        torch = _torch(); owner = self.owner
        if isinstance(points, torch.Tensor):
            raise TypeError('Query points must be fixed NumPy data')
        q = np.asarray(points, dtype=float)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError('Expected finite (n,2) query points')
        if len(q)==0:
            empty=torch.zeros((0,2),dtype=torch.float64, device='cpu');return empty,empty
        owners = cKDTree(owner.cloud.interior).query(q)[1]
        result = {}
        for index in np.unique(owners):
            index = int(index); task=owner.tasks[index]; stencil=owner.stencils[index]
            values=torch.stack([{'solution':self.nodal_velocity,'boundary':owner.boundary,'pde':owner.forcing}[g][j]
                                for g,j in stencil.slots])
            # Symmetric solve uses the same equilibration as local weights.
            coefficients,_,_=solve_weights(self.grams[index],values[:,None])
            rows=np.flatnonzero(owners==index)
            z=torch.tensor(q[rows,None,:]-task['points'][None,:,:],dtype=torch.float64, device='cpu')
            z=z[:,None,:,:].expand(-1,4,-1,-1)
            evaluation=_block(z,torch.tensor(task['codes'],device='cpu')[None,None,:],torch.tensor([1,2,7,8],device='cpu')[None,:,None],
                              owner.velocity_kernel,owner.pressure_kernel,self.vp,self.pp,
                              torch.tensor(float(task['mu']),dtype=torch.float64, device='cpu'))
            fields=(evaluation@coefficients).squeeze(-1)
            for j,row in enumerate(rows):
                field=fields[j]
                if np.array_equal(q[row],task['origin']):
                    field=torch.cat([self.nodal_velocity[[index,index+owner.ni]],field[2:]])
                result[int(row)]=field
        out=torch.stack([result[i] for i in range(len(q))])
        return out[:,:2],out[:,2:]
