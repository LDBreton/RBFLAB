"""Optional PyTorch Float64 primitives and 2D Stokes LHI backend.

Importing RBFLAB does not import torch. Global sparse solves and off-node
reconstruction continue to use the existing CPU implementation.
"""
from .configuration import BackendSettings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
import sympy as sp
from .kernel_compiler import as_bound, indices
from .kernels import _COORDINATES
from .symbolic_kernel import expressions, compact_parts, near_expression


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install RBFLAB's optional torch extra: pip install 'rbflab[torch]'") from exc
    return torch


def _math_functions():
    torch = _torch()
    return {name: (lambda value, fn=getattr(torch, name):
                   fn(torch.as_tensor(value, dtype=torch.float64)))
            for name in ('sqrt', 'exp', 'sin', 'cos', 'log')}


@lru_cache(maxsize=512)
def _functions(family, alpha):
    torch = _torch()
    modules = [_math_functions(), 'math']
    args = (*_COORDINATES[:len(alpha)], *family.parameters)
    regular, origin = expressions(family, alpha)
    _, support = compact_parts(family.expression, family.radial_variable)
    exprs = (regular, origin, near_expression(family, alpha)) if support is not None else (regular, origin)
    return tuple(sp.lambdify(args, e, modules=modules, cse=True) for e in exprs)


class TorchKernel:
    """Tensor Cartesian derivatives in 2D/3D with certified origin/support branches.

    Parameter overrides have shape (..., number_of_parameters), broadcastable
    to displacement.shape[:-1]. Kernels and parameter values stay independent.
    """
    def __init__(self, kernel):
        self.bound = as_bound(kernel)

    def prepare(self, dimension=2, derivative_order=6):
        if dimension not in (2, 3) or type(derivative_order) is not int or not 0 <= derivative_order <= 6:
            raise ValueError('Expected dimension 2/3 and derivative_order 0..6')
        for alpha in indices(dimension, derivative_order):
            _functions(self.bound.family, alpha)
        return self

    def derivative(self, displacement, alpha=None, *, parameters=None):
        torch = _torch()
        z = torch.as_tensor(displacement, dtype=torch.float64)
        if z.ndim < 1 or z.shape[-1] not in (2, 3) or not torch.isfinite(z).all():
            raise ValueError('Expected finite 2D/3D displacements')
        alpha = tuple(alpha) if alpha is not None else (0,) * z.shape[-1]
        if len(alpha) != z.shape[-1]:
            raise ValueError('Derivative dimension mismatch')
        functions = _functions(self.bound.family, alpha)
        params = torch.as_tensor(tuple(map(float, self.bound.values)) if parameters is None else parameters,
                                 dtype=z.dtype, device=z.device)
        if params.ndim < 1 or params.shape[-1] != len(self.bound.values) or not torch.isfinite(params).all():
            raise ValueError('Invalid runtime parameter array')
        for j, symbol in enumerate(self.bound.family.parameters):
            if symbol.is_positive and (params[..., j] <= 0).any():
                raise ValueError(f'{symbol} must be positive')
            if symbol.is_nonnegative and (params[..., j] < 0).any():
                raise ValueError(f'{symbol} must be nonnegative')
        params = torch.broadcast_to(params, z.shape[:-1] + (params.shape[-1],))
        flat = z.reshape(-1, z.shape[-1])
        params = params.reshape(len(flat), params.shape[-1])
        zero = (flat == 0).all(dim=-1)
        squared = (flat * flat).sum(dim=-1)
        active = torch.ones_like(zero)
        near = torch.zeros_like(zero)
        _, support = compact_parts(self.bound.family.expression, self.bound.family.radial_variable)
        if support is not None:
            # Support expressions contain only parameters, never coordinates.
            fn = _support_function(self.bound.family)
            radius2 = fn(*(params[:, j] for j in range(params.shape[-1])))
            active = squared < radius2
            near = active & ~zero & (squared < radius2 / 16)
        out = torch.zeros(len(flat), dtype=z.dtype, device=z.device)
        masks = (active & ~zero & ~near, zero & active, near)
        # Evaluate only selected branches: torch.where would also evaluate
        # singular expressions at zero and can poison subsequent gradients.
        for fn, mask in zip(functions, masks):
            if mask.any():
                x = flat[mask]
                p = params[mask]
                value = fn(*(x[:, j] for j in range(x.shape[-1])),
                           *(p[:, j] for j in range(p.shape[-1])))
                out[mask] = torch.as_tensor(value, dtype=z.dtype, device=z.device)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite tensor kernel derivative')
        return out.reshape(z.shape[:-1])


@lru_cache(maxsize=64)
def _support_function(family):
    torch = _torch()
    _, support = compact_parts(family.expression, family.radial_variable)
    return sp.lambdify(family.parameters, support,
                      modules=[_math_functions(), 'math'], cse=True)


def solve_weights(gram, rhs, *, compute_condition=False):
    """Solve G.T W = B for batches, with symmetric row-max equilibration.

    Returns weights, relative max residual per batch, and optional scaled 2-norm
    condition number. This primitive is independent of the PDE and stencil type.
    """
    torch = _torch()
    if gram.dtype != torch.float64 or rhs.dtype != torch.float64:
        raise ValueError('Tensor solves require Float64')
    if gram.ndim < 2 or gram.shape[-1] != gram.shape[-2] or rhs.shape[:-1] != gram.shape[:-1]:
        raise ValueError('Expected matching square matrices and multiple RHS columns')
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise FloatingPointError('Nonfinite local system')
    maxima = gram.abs().amax(dim=-1)
    if (maxima <= 0).any():
        raise ValueError('Local system has a zero row')
    scale = maxima.rsqrt()
    matrix = gram * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    weights = torch.linalg.solve(matrix.transpose(-1, -2), rhs * scale.unsqueeze(-1)) * scale.unsqueeze(-1)
    defect = gram.transpose(-1, -2) @ weights - rhs
    denom = rhs.abs().amax(dim=(-2, -1))
    residual = defect.abs().amax(dim=(-2, -1)) / torch.where(denom > 0, denom, torch.ones_like(denom))
    if not torch.isfinite(weights).all() or not torch.isfinite(residual).all():
        raise FloatingPointError('Nonfinite local weights or residual')
    condition = torch.linalg.cond(matrix) if compute_condition else None
    return weights, residual, condition


@lru_cache(maxsize=64)
def _terms(source, target):
    # Tuples: component, Cartesian derivative, coefficient, viscosity power.
    def functional(code):
        if code in (1, 2, 3, 4):
            return [((code - 1) % 2, (0, 0), 1, 0)]
        if code in (5, 6):
            c = code - 5
            return [(c, (2, 0), -1, 1), (c, (0, 2), -1, 1),
                    (2, (1, 0) if c == 0 else (0, 1), 1, 0)]
        if code in (7, 8):
            return [(2, (1, 0) if code == 7 else (0, 1), 1, 0)]
        raise ValueError('Unknown Stokes functional')
    terms = defaultdict(int)
    for a, left, lc, lm in functional(target):
        for b, right, rc, rm in functional(source):
            if a == b == 2:
                base = [((0, 0), 1)]
            elif a < 2 and b < 2:
                d = [0, 0]; d[a] += 1; d[b] += 1
                base = [(tuple(d), 1)]
                if a == b:
                    base += [((2, 0), -1), ((0, 2), -1)]
            else:
                continue
            for d, coefficient in base:
                alpha = tuple(x + y + z for x, y, z in zip(left, right, d))
                terms[(a == 2, alpha, lm + rm)] += lc * rc * coefficient * (-1) ** sum(right)
    return tuple((pressure, alpha, power, c) for (pressure, alpha, power), c in terms.items() if c)


def _block(z, source, target, velocity, pressure, vp, pp, mu):
    torch = _torch()
    shape = z.shape[:-1]
    source, target = torch.broadcast_to(source, shape), torch.broadcast_to(target, shape)
    vp = torch.broadcast_to(vp, shape + (vp.shape[-1],))
    pp = torch.broadcast_to(pp, shape + (pp.shape[-1],))
    mu = torch.broadcast_to(mu, shape)
    out = torch.zeros(shape, dtype=z.dtype, device=z.device)
    for s in source.unique().tolist():
        for t in target.unique().tolist():
            mask = (source == s) & (target == t)
            if not mask.any():
                continue
            points, v, p, viscosity = z[mask], vp[mask], pp[mask], mu[mask]
            value = torch.zeros(len(points), dtype=z.dtype, device=z.device)
            for is_pressure, alpha, power, coefficient in _terms(s, t):
                kernel, params = (pressure, p) if is_pressure else (velocity, v)
                value = value + coefficient * viscosity ** power * kernel.derivative(points, alpha, parameters=params)
            out[mask] = value
    return out


@dataclass(frozen=True)
class TorchBackend(BackendSettings):
    """CPU Float64, 2D unaugmented physical-coordinate Stokes local systems."""
    batch_size: int = 64
    threads: int = 1
    device: str = 'cpu'
    compute_condition: bool = False

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError('threads must be a positive integer')
        if self.device != 'cpu':
            raise NotImplementedError('The first Torch LHI backend supports CPU only')
        if type(self.compute_condition) is not bool:
            raise TypeError('compute_condition must be bool')
        if self.shape_rule not in ('fixed', 'hardy'):
            raise ValueError('Unknown shape rule')

    def prepare(self, velocity, pressure, precision, dimension=2):
        if dimension != 2:
            raise NotImplementedError('Torch Stokes LHI supports 2D')
        if precision.local_digits is not None or precision.global_dtype != 'float64' or precision.global_digits is not None:
            raise ValueError('Torch LHI requires Float64; use C++ MPFR for extended precision')
        return (TorchKernel(velocity).prepare(2, 6), TorchKernel(pressure).prepare(2, 2))

    def assemble(self, method, problem, cloud):
        from .lhi_backends import _prepare, _finish
        begin = time.perf_counter()
        if method.stencil_policy.scaling != 'physical' or method.polynomial_degree is not None:
            raise NotImplementedError('Torch LHI currently requires unaugmented physical-coordinate stencils')
        vk, pk = self.prepare(method.kernel, method.pressure_kernel or method.kernel, method.precision, cloud.dimension)
        prepared = time.perf_counter()
        a, tasks, stencils = _prepare(method, problem, cloud, self.shape_rule)
        torch = _torch()
        groups = defaultdict(list)
        for i, task in enumerate(tasks):
            groups[len(task['points'])].append(i)
        results = [None] * len(tasks)
        matrix_seconds = solve_seconds = 0.0
        batches = 0
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.threads)
            with torch.no_grad():
                for ids in groups.values():
                    for offset in range(0, len(ids), self.batch_size):
                        selected = ids[offset:offset + self.batch_size]
                        batch = [tasks[i] for i in selected]
                        tensor = lambda data: torch.as_tensor(np.asarray(data, dtype=float), dtype=torch.float64, device=self.device)
                        points = tensor([t['points'] for t in batch])
                        codes = torch.tensor([t['codes'] for t in batch], device=self.device)
                        vp = tensor([as_bound(t['vk']).values for t in batch])[:, None, None, :]
                        pp = tensor([as_bound(t['pk']).values for t in batch])[:, None, None, :]
                        mu = tensor([float(t['mu']) for t in batch])[:, None, None]
                        start = time.perf_counter()
                        G = _block(points[:, :, None, :] - points[:, None, :, :], codes[:, None, :],
                                   codes[:, :, None], vk, pk, vp, pp, mu)
                        G = G.tril() + G.tril(-1).transpose(-1, -2)
                        origin = tensor([t['origin'] for t in batch])
                        z = (origin[:, None, None, :] - points[:, None, :, :]).expand(-1, 4, -1, -1)
                        Q = _block(z, codes[:, None, :], torch.arange(5, 9, device=self.device)[None, :, None], vk, pk, vp, pp, mu)
                        matrix_seconds += time.perf_counter() - start
                        start = time.perf_counter()
                        W, residual, condition = solve_weights(G, Q.transpose(-1, -2), compute_condition=self.compute_condition)
                        solve_seconds += time.perf_counter() - start
                        for j, i in enumerate(selected):
                            results[i] = dict(weights=W[j].T.tolist(), residual=float(residual[j]),
                                              condition=None if condition is None else float(condition[j]))
                        batches += 1
        finally:
            torch.set_num_threads(old_threads)
        system = _finish(method, problem, cloud, a, tasks, stencils, results,
                         dict(backend='torch_float64', device=self.device, torch_version=torch.__version__,
                              threads=self.threads, batch_size=self.batch_size, batches=batches,
                              prepare_seconds=prepared-begin, matrix_seconds=matrix_seconds, solve_seconds=solve_seconds,
                              compute_condition=self.compute_condition, condition_norm='2', local_digits=None,
                              shape_rule=self.shape_rule, off_node_backend='lazy Python'))
        system.backend_diagnostics['assembly_seconds'] = time.perf_counter() - begin
        return system
