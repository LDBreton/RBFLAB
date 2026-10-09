"""A small Re=100 lid-driven cavity using staggered RBF-FD.

Run ``python -m examples.navier_stokes_cavity`` after installing
``rbflab[examples]``. The default Python backend needs no C++ build or mesher.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import splu
import rbflab as r
from rbflab import viz


# --8<-- [start:clouds]
def clouds(cells):
    """Vertices carry pressure; triangle edge midpoints carry velocity."""
    pressure = r.geometry.unit_box_grid(cells)
    node = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    a, b = node[:-1, :-1].ravel(), node[1:, :-1].ravel()
    c, d = node[1:, 1:].ravel(), node[:-1, 1:].ravel()
    triangles = np.vstack((np.column_stack((a, b, c)),
                           np.column_stack((a, c, d))))
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                       triangles[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    points = pressure.points[edges].mean(axis=1)
    sides = (("left", 0, 0, -1), ("right", 0, 1, 1),
             ("bottom", 1, 0, -1), ("top", 1, 1, 1))
    boundary, normals = {}, {}
    for name, axis, value, sign in sides:
        ids = np.flatnonzero(np.isclose(points[:, axis], value))
        normal = np.zeros(2); normal[axis] = sign
        boundary[name] = ids
        normals[name] = np.tile(normal, (len(ids), 1))
    return r.PointCloud(points, boundary, normals), pressure


# --8<-- [end:clouds]
def solve(*, cells=16, dt=.00125, end=20., backend="python", frames=60):
    """Advance CN diffusion / AB2 convection with a coupled pressure constraint.

    All arrays and sparse operators use Float64. A compatibility multiplier
    allows a spatially constant divergence defect; this is reported explicitly.
    Returns velocity samples, not a pressure time history. See the cavity
    tutorial for the block equations and the meaning of each diagnostic.
    """
    if cells < 6 or dt <= 0 or end <= 0 or frames < 2:
        raise ValueError("Require cells >= 6 and positive dt, end, and frames >= 2")
    steps = round(end / dt)
    if not np.isclose(steps * dt, end, atol=1e-12, rtol=0):
        raise ValueError("end must be an integer multiple of dt")
    local = (r.PythonBackend(compute_condition=False) if backend == "python" else
             r.CppBackend(threads=8, compute_condition=False) if backend == "cpp" else None)
    if local is None:
        raise ValueError("backend must be 'python' or 'cpp'")
    # --8<-- [start:operators]
    velocity, pressure = clouds(cells)
    rbffd = r.RBFFD(spaces={"u": r.ScalarSpace(r.PHS(7), 3),
                            "p": r.PressureSpace(r.PHS(7), 3)},
                    stencil_size=28, local_backend=local)
    differential = {"dx": r.Derivative(0), "dy": r.Derivative(1),
                    "lap": r.Laplacian()}
    def operators(source, target, space):
        return rbffd.operators(source=source, targets=target, space=space,
                               operators=differential)
    vv = operators(velocity, velocity, "u")
    vp = operators(velocity, pressure, "u")
    pv = operators(pressure, velocity, "p")
    dx, dy, lap = (vv[name].matrix for name in ("dx", "dy", "lap"))
    Dx, Dy = vp.dx.matrix, vp.dy.matrix
    Gx, Gy = pv.dx.matrix, pv.dy.matrix

    # --8<-- [end:operators]
    # --8<-- [start:assembly]
    inside, wall = velocity.interior_indices, velocity.boundary_indices
    n, m = len(velocity.points), len(pressure.points)
    viscosity = .01  # Re = lid speed * box width / viscosity = 100
    # CN diffusion: A multiplies u^{n+1}, B multiplies u^n.
    A = eye(n, format="csr") / dt - .5 * viscosity * lap
    B = eye(n, format="csr") / dt + .5 * viscosity * lap
    a = A[inside][:, inside]
    zero = csr_matrix(a.shape)
    ones = csr_matrix(np.ones((m, 1)))
    # Unknowns: [u_I, v_I, pressure, compatibility multiplier].
    # Last row fixes the arithmetic pressure mean; the column of ones allows
    # D_x u + D_y v = -lambda at every pressure node, not exactly zero.
    coupled = bmat([[a, zero, Gx[inside], None],
                    [zero, a, Gy[inside], None],
                    [Dx[:, inside], Dy[:, inside], None, ones],
                    [None, None, ones.T, None]], format="csc")
    factor = splu(coupled)
    fixed = np.zeros((n, 2))
    fixed[velocity.boundary["top"], 0] = 1.
    # Eliminate the known velocity boundary values from momentum/continuity.
    correction = A[inside][:, wall] @ fixed[wall]
    continuity = -Dx[:, wall] @ fixed[wall, 0] - Dy[:, wall] @ fixed[wall, 1]
    # --8<-- [end:assembly]
    # --8<-- [start:time_loop]
    u = fixed.copy()
    previous = None
    times, samples = [0.], [u.copy()]
    stride = max(1, steps // (frames - 1))
    for step in range(1, steps + 1):
        # Advective (not conservative/skew-symmetric) convection, evaluated at velocity nodes.
        convection = u[:, 0, None] * (dx @ u) + u[:, 1, None] * (dy @ u)
        explicit = convection if previous is None else 1.5 * convection - .5 * previous
        rhs = B[inside] @ u - explicit[inside] - correction
        # Reuse the same sparse LU because dt, viscosity and nodes are fixed.
        answer = factor.solve(np.r_[rhs[:, 0], rhs[:, 1], continuity, 0.])
        new = fixed.copy()
        new[inside, 0] = answer[:len(inside)]
        new[inside, 1] = answer[len(inside):2 * len(inside)]
        if not np.isfinite(new).all() or np.max(np.abs(new)) > 5:
            raise RuntimeError(f"Cavity solve became unstable at t={step * dt:g}")
        u, previous = new, convection
        if step % stride == 0 or step == steps:
            times.append(step * dt)
            samples.append(u.copy())
    # --8<-- [end:time_loop]
    divergence = Dx @ u[:, 0] + Dy @ u[:, 1]
    last_rhs = np.r_[rhs[:, 0], rhs[:, 1], continuity, 0.]
    diagnostics = {"velocity_nodes": n, "pressure_nodes": m,
                   "linear_residual_max": float(np.max(np.abs(coupled @ answer - last_rhs))),
                   "continuity_equation_max": float(np.max(np.abs(divergence + answer[-1]))),
                   "max_divergence": float(np.max(np.abs(divergence))),
                   "compatibility": float(answer[-1]),
                   "pressure_mean": float(np.mean(answer[2 * len(inside):2 * len(inside) + m])),
                   "max_speed": float(np.max(np.linalg.norm(u, axis=1)))}
    return velocity.points, np.asarray(times), np.asarray(samples), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--end", type=float, default=20.)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--output", type=Path,
                        default=Path("docs/assets/navier_stokes_cavity.gif"))
    args = parser.parse_args()
    points, times, velocities, diagnostics = solve(cells=args.cells,
                                                    end=args.end, backend=args.backend)
    viz.animate_velocity_samples(points, times, velocities, args.output,
                                 title="Lid-driven cavity  |  Re = 100",
                                 poster=args.output.with_suffix(".png"))
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")
    print(diagnostics)


if __name__ == "__main__":
    main()
