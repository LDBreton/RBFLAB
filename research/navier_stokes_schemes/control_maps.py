"""Build and audit the original Stokes maps used by the control experiment.

The historical smooth-solution recipe is reproduced without a pressure
correction. Its stability on a different viscosity/grid is measured, not assumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sympy as sp
from scipy.linalg import eig
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

import rbflab as r
from .cavity import Cavity
from .reused import assemble


NAMES = ("M", "A", "B", "SL", "PY", "PB", "PL",
         "DxU", "DxB", "DxL", "DyU", "DyB", "DyL")


def load_maps(folder):
    """Load only this experiment's explicit original-recipe manifest."""
    folder = Path(folder)
    recipe = json.loads((folder / "configuration.json").read_text())
    if (recipe["pressure_amplitude"] not in ("q^4", "nu^2*q^4")
            or recipe["augmentation"] not in ([6, 1], [6, 2])):
        raise ValueError("Expected uncorrected degree-6/1 or degree-6/2 Stokes maps")
    problem = Cavity(cells=recipe["cells"], nu=recipe["nu"], build_maps=False)
    pressure_degree = recipe["augmentation"][1]
    problem.map_configuration = recipe
    if recipe["pressure_amplitude"] == "nu^2*q^4":
        set_normalized_space(problem, pressure_degree=pressure_degree)
    elif pressure_degree != 1:
        raise ValueError("Quadratic pressure maps require explicit viscosity-normalized kernels")
    nodes = np.load(folder / "nodes.npz")
    np.testing.assert_array_equal(nodes["points"], problem.points)
    np.testing.assert_array_equal(nodes["boundary_points"], problem.boundary_points)
    for name in NAMES:
        setattr(problem, name, load_npz(folder / f"{name}.npz"))
    return problem


def set_normalized_space(problem, pressure_degree=1):
    """Represent pressure as nu times its viscosity-one counterpart."""
    q = sp.Rational(problem.cells, 44)
    viscosity = sp.Rational(str(problem.nu))
    s = sp.Symbol("s", nonnegative=True)
    problem.space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), 6),
        "p": r.PressureSpace(r.Kernel(viscosity**2*q**4*sp.exp(-q*q*s), s)(), pressure_degree)})
    problem.u, problem.p = problem.space.fields
    momentum = -viscosity*r.laplacian(problem.u)+r.gradient(problem.p)
    problem.source = {
        "velocity": r.Samples(problem.points, operator=problem.u, within=problem.stencil),
        "boundary": r.Samples(problem.boundary_points, operator=problem.u, within=problem.stencil),
        "momentum": r.Samples(problem.points, operator=momentum,
                              within=problem.stencil, target="exclude")}


def rescale(source, cells, folder, nu=.01):
    """Exact viscosity/pressure congruence, then Float64 matrix export.

    With p_nu=nu*p_1, momentum observations also scale by nu. The pressure
    kernel amplitude scales by nu**2. This changes only units in the same
    local saddle system, including an invertible rescaling of pressure
    polynomial columns. MPI parity is verified separately, on selected rows.
    """
    source, folder = Path(source), Path(folder)
    manifest = json.loads((source / "configuration.json").read_text())
    expected = dict(nu=1.0, stencil=37, augmentation="6/1", scaling="q=cells/44",
                    pressure="pure Gaussian cbar=1", local_digits=80)
    if any(manifest.get(key) != value for key, value in expected.items()) or cells not in manifest["cells"]:
        raise ValueError("Source is not the validated viscosity-one recipe")
    raw = source / f"maps_cells{cells}"
    weight_recipe = json.loads((source / f"weights_cells{cells}" / "recipe.json").read_text())
    cloud = r.geometry.unit_box_grid(cells)
    if weight_recipe["cloud_sha256"] != hashlib.sha256(cloud.points.tobytes()).hexdigest():
        raise ValueError("Source cloud differs")
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(cells=cells, total_nodes=(cells+1)**2, interior_nodes=(cells-1)**2,
                  nu=nu, stencil=37, augmentation=[6, 1], q=f"{cells}/44",
                  velocity="exp(-10*q^2*r^2)+q^7*r^7",
                  pressure="nu^2*q^4*exp(-q^2*r^2)", pressure_amplitude="nu^2*q^4",
                  pressure_cbar=1, precision_digits=80, source=str(source.resolve()),
                  build="exact viscosity rescaling of saved MPI8 MPFR80 weights; selected rows freshly verified",
                  corner_subtraction=False, pressure_correction=False,
                  boundary="unit lid; corner nodes and other walls stationary")
    factors = {name: (nu if name in ("A", "B", "PY", "PB") else
                     1/nu if name in ("DxL", "DyL") else 1) for name in NAMES}
    source_hashes, destination_hashes = {}, {}
    for name, factor in factors.items():
        path = raw / f"{name}.npz"
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        matrix = load_npz(path) * factor
        if not np.isfinite(matrix.data).all():
            raise ValueError(f"Nonfinite map {name}")
        save_npz(folder / path.name, matrix)
        destination_hashes[name] = hashlib.sha256((folder / path.name).read_bytes()).hexdigest()
    nodes = np.load(raw / "nodes.npz")
    np.testing.assert_array_equal(nodes["points"], cloud.interior)
    np.testing.assert_array_equal(nodes["boundary_points"], cloud.points[cloud.boundary_indices])
    np.savez(folder / "nodes.npz", points=nodes["points"], boundary_points=nodes["boundary_points"])
    (folder / "configuration.json").write_text(json.dumps(config, indent=2))
    evidence = dict(source_hashes=source_hashes, destination_hashes=destination_hashes,
                    scale_factors=factors,
                    original_mpi_assembly=json.loads((source / f"weights_cells{cells}" / "assembly.json").read_text()))
    (folder / "rescaling.json").write_text(json.dumps(evidence, indent=2))
    print(f"Rescaled maps ready: {folder}", flush=True)
    return load_maps(folder)


def verify_mpi(folder, ranks=8):
    """Recompute wall, corner-adjacent, and interior owners independently."""
    folder = Path(folder)
    saved = load_maps(folder)
    problem = Cavity(cells=saved.cells, nu=saved.nu, ranks=ranks, digits=80,
                     native_timeout=1800, build_maps=False)
    set_normalized_space(problem)
    # Exact decimal viscosity matches the normalized kernel congruence.
    problem.nu = sp.Rational(str(saved.nu))
    side = problem.cells - 1
    axis = sorted(set((0, 1, side//2, side-2, side-1)))
    owners = sorted(set(a*side+b for a in axis for b in axis))
    fresh, execution = assemble(problem, r.geometry.unit_box_grid(problem.cells), owners=owners)
    rows = np.array([[2*i, 2*i+1] for i in owners]).ravel()
    comparisons = {}
    for name in NAMES:
        actual, expected = fresh[name][rows], getattr(saved, name)[rows]
        difference = actual-expected
        absolute = float(np.max(abs(difference.data))) if difference.nnz else 0.
        scale = float(np.max(abs(expected.data))) if expected.nnz else 0.
        relative = absolute/max(scale, 1e-300)
        comparisons[name] = dict(max_absolute=absolute, max_relative=relative)
        if relative > 2e-12 and absolute > 2e-12:
            raise AssertionError((name, absolute, relative))
    result = dict(owners=owners, owner_count=len(owners), ranks=ranks, digits=80,
                  execution=execution, comparisons=comparisons, status="passed")
    (folder / "mpi_parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(folder=str(folder), parity=result["status"],
                          owners=len(owners), comparisons=comparisons)), flush=True)
    return result


def audit(problem, folder):
    """Full small-grid Stokes spectrum and a matched eigenmode evolution."""
    folder = Path(folder)
    if len(problem.U0) > 2000:
        raise ValueError("Dense spectra are restricted to small implementation checks")
    A, M = problem.A.toarray(), problem.M.toarray()
    values, vectors = eig(-A, M)
    finite = np.isfinite(values)
    index = np.flatnonzero(finite)[np.argmax(values[finite].real)]
    value, vector = values[index], vectors[:, index]
    action = -A @ vector - value * (M @ vector)
    residual = np.linalg.norm(action) / (
        np.linalg.norm(A @ vector) + abs(value) * np.linalg.norm(M @ vector))
    velocity = vector.real if np.linalg.norm(vector.real) >= np.linalg.norm(vector.imag) else vector.imag
    velocity /= np.max(abs(velocity))
    spectrum = dict(leading_real=float(value.real), leading_imag=float(value.imag),
                    positive_count=int(np.sum(values[finite].real > 1e-7)),
                    finite_count=int(finite.sum()), size=len(values),
                    action_relative_residual=float(residual),
                    mass_action_ratio=float(np.linalg.norm(M @ vector) / np.linalg.norm(vector)))
    probes = []
    for dt in (.00025, .000125):
        solve = splu((problem.M / dt + problem.A).tocsc())
        U = velocity.copy()
        history = [float(np.max(abs(U)))]
        for _ in range(round(.01 / dt)):
            U = solve.solve(problem.M @ U / dt)
            history.append(float(np.max(abs(U))))
        probes.append(dict(dt=dt, T=.01, final_gain=history[-1],
                           peak_gain=max(history), history=history))
    grad = np.column_stack((problem.points[:, 1], problem.points[:, 0])).ravel()
    pressure = dict(xy_constraint_inf=float(np.max(abs(problem.M @ grad))),
                    xy_gradient_error_inf=float(np.max(abs(problem.PL @ grad - grad))),
                    xy_dx_velocity_inf=float(np.max(abs(problem.DxL @ grad))),
                    xy_dy_velocity_inf=float(np.max(abs(problem.DyL @ grad))))
    result = dict(stokes_spectrum=spectrum, homogeneous_mode_probes=probes,
                  pressure=pressure, scope="Float64 full generalized eigensolve; no stability theorem")
    np.savez(folder / "stokes_spectrum.npz", values=values, leading_mode=velocity)
    (folder / "stokes_audit.json").write_text(json.dumps(result, indent=2))
    return result


def homogeneous_probes(folder, end=.1):
    """Bounded large-grid Stokes checks; no absence-of-growing-modes claim."""
    folder = Path(folder)
    p = load_maps(folder)
    x, y = p.points.T
    smooth = np.column_stack((2*np.pi*np.sin(np.pi*x)**2*np.sin(np.pi*y)*np.cos(np.pi*y),
                             -2*np.pi*np.sin(np.pi*y)**2*np.sin(np.pi*x)*np.cos(np.pi*x))).ravel()
    random = np.random.default_rng(20261030).standard_normal(len(smooth))
    initial = np.column_stack((smooth/np.linalg.norm(smooth), random/np.linalg.norm(random)))
    results = []
    for dt in (.00025, .000125):
        start = time.perf_counter()
        solve = splu((p.M/dt+p.A).tocsc(), permc_spec="MMD_AT_PLUS_A")
        factor_seconds = time.perf_counter()-start
        U = initial.copy()
        history = [np.linalg.norm(U, axis=0).tolist()]
        residual_max = 0.
        for _ in range(round(end/dt)):
            rhs = p.M @ U / dt
            U = solve.solve(rhs)
            residual = (p.M/dt+p.A) @ U-rhs
            residual_max = max(residual_max, float(np.max(abs(residual))))
            history.append(np.linalg.norm(U, axis=0).tolist())
            if not np.isfinite(U).all() or max(history[-1]) > 1e6:
                break
        result = dict(dt=dt, intended_T=end, actual_T=dt*(len(history)-1),
                      initial_fields=["analytic divergence-free smooth curl", "seeded nodal random; not globally solenoidal"],
                      final_l2_gain=history[-1], peak_l2_gain=np.max(history,axis=0).tolist(),
                      history=history, solve_residual_inf=residual_max,
                      factor_seconds=factor_seconds, total_seconds=time.perf_counter()-start,
                      factor_nnz=int(solve.L.nnz+solve.U.nnz))
        results.append(result)
        np.savez(folder/f"homogeneous_dt{dt}.npz", U0=initial, U=U, points=p.points)
        print(json.dumps({k:v for k,v in result.items() if k!="history"}), flush=True)
        (folder/"homogeneous_probes.json").write_text(json.dumps(dict(
            scope="Two chosen nodal directions, finite interval; not a spectral or worst-case transient certificate",
            results=results),indent=2))
    return results


def pressure_patches(folder):
    """Check affine and quadratic pressures at the actual solution nodes."""
    folder = Path(folder)
    p = load_maps(folder)
    x, y = p.points.T
    fields = {"x-y": np.column_stack((np.ones_like(x), -np.ones_like(y))).ravel(),
              "xy": np.column_stack((y, x)).ravel()}
    result = {}
    for name, gradient in fields.items():
        result[name] = dict(constraint_inf=float(np.max(abs(p.M@gradient))),
                            gradient_error_inf=float(np.max(abs(p.PL@gradient-gradient))),
                            dx_velocity_inf=float(np.max(abs(p.DxL@gradient))),
                            dy_velocity_inf=float(np.max(abs(p.DyL@gradient))))
    (folder/"pressure_patch.json").write_text(json.dumps(result, indent=2))
    return result


def build(cells, folder, ranks=8):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    recipe = dict(cells=cells, total_nodes=(cells+1)**2, interior_nodes=(cells-1)**2,
                  nu=.01, stencil=37, augmentation=[6, 1], q=f"{cells}/44",
                  velocity="exp(-10*q^2*r^2)+q^7*r^7",
                  pressure="q^4*exp(-q^2*r^2)", pressure_amplitude="q^4",
                  pressure_cbar=1, precision_digits=80, ranks=ranks,
                  geometry="Float64 deterministic unit_box_grid",
                  boundary="unit lid; corner nodes and other walls stationary",
                  corner_subtraction=False, pressure_correction=False)
    (folder / "configuration.json").write_text(json.dumps(recipe, indent=2))
    problem = Cavity(cells=cells, nu=.01, ranks=ranks, digits=80,
                     native_timeout=1800, build_maps=False,
                     weight_folder=folder / "weights")
    start = time.perf_counter()
    blocks, execution = assemble(problem, r.geometry.unit_box_grid(cells))
    for name, matrix in blocks.items():
        setattr(problem, name, matrix)
        save_npz(folder / f"{name}.npz", matrix)
    np.savez(folder / "nodes.npz", points=problem.points, boundary_points=problem.boundary_points)
    hashes = {name: hashlib.sha256((folder / f"{name}.npz").read_bytes()).hexdigest() for name in NAMES}
    report = dict(seconds=time.perf_counter()-start, execution=execution, hashes=hashes)
    (folder / "assembly.json").write_text(json.dumps(report, indent=2))
    if len(problem.U0) <= 2000:
        result = audit(problem, folder)
        print(json.dumps(dict(folder=str(folder), seconds=report["seconds"],
                              spectrum=result["stokes_spectrum"])), flush=True)
    else:
        homogeneous_probes(folder)
    pressure_patches(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--rescale-source")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.probe_only:
            homogeneous_probes(args.output)
            pressure_patches(args.output)
        elif args.verify_only:
            verify_mpi(args.output, args.ranks)
        elif args.rescale_source:
            rescale(args.rescale_source, args.cells, args.output)
            verify_mpi(args.output, args.ranks)
        else:
            build(args.cells, args.output, args.ranks)
