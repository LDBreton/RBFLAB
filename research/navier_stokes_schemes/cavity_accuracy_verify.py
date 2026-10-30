"""Nodal accuracy and nonlinear checks of a saved cavity candidate.

Prepare once, then advance each case in bounded chunks with exact BDF history.
Ghia is used only for reporting, never for choosing reconstruction coefficients.
"""
import argparse
import contextlib
import copy
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from .pressure_lifting import load_problem, all_pressure_controls
from .cavity_nullspace_lifting import coupled_spectrum
from .cavity_candidates import reference_at_nodal_lines
from .equilibrium_segment_engine import evolve
from .metric_lifting import spectrum


def problem(source, cells):
    p = load_problem(cells, source / "maps")
    if p.B.shape[1] == len(p.g0) + 1:
        p.g0 = np.r_[p.g0, 1.]
    if p.B.shape[1] != len(p.g0):
        raise ValueError("Saved boundary map has an unsupported affine layout")
    return p


def prepare(source, cells, output):
    output.mkdir(parents=True, exist_ok=False)
    p = problem(source, cells)
    saved = np.load(source / "steady.npz")
    U, L = saved["U"], saved["L"]
    info, values, left, right, mass, resolved = coupled_spectrum(p, U, L, True)
    ids = np.flatnonzero(resolved)
    k = ids[np.argmax(values[ids].real)]
    mode = right[:, k].astype(complex).copy()
    mode *= np.exp(-1j * np.angle(mode[np.argmax(abs(mode[:len(U)]))]))
    mode /= np.max(abs(mode[:len(U)].real))
    np.savez(output / "initial.npz", U=U, L=L, mode=mode, eigenvalue=values[k])
    report = dict(source=str(source), cells=cells, total_nodes=(cells+1)**2,
                  spectrum=info, rest=spectrum(p), pressure=all_pressure_controls(p),
                  ghia=reference_at_nodal_lines(p.points, U),
                  scope="Ghia reference interpolated to existing centerline nodes")
    (output / "configuration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(cells=cells, spectrum=info)), flush=True)


def advance(source, cells, output, scheme, dt, control, steps, end, amplitude):
    p = problem(source, cells)
    saved = np.load(output / "initial.npz")
    Uref, Lref = saved["U"], saved["L"]
    case = output / f"{control}_{scheme}_{dt}"
    case.mkdir(exist_ok=True)
    history = None
    amplitude = amplitude if control == "perturb" else 0.
    if control == "pressure":
        gradient = np.column_stack((p.points[:, 1], p.points[:, 0])).ravel()
        p.g0 = np.zeros_like(p.g0)
        p.force = lambda t: gradient.copy()
        Uref, Lref = np.zeros_like(Uref), gradient
    if (case / "checkpoint.npz").exists():
        checkpoint = np.load(case / "checkpoint.npz")
        U, L = checkpoint["U"], checkpoint["L"]
        next_step = int(checkpoint["next_step"])
        history = (checkpoint["Uold"], checkpoint["Lold"], next_step)
    else:
        mode = saved["mode"].real
        U = Uref + amplitude * mode[:len(Uref)]
        L = Lref + amplitude * mode[len(Uref):]
        next_step = 1
    total_steps = round(end / dt)
    count = min(steps, total_steps - next_step + 1)
    if count <= 0:
        return
    p.initial = lambda: (U.copy(), L.copy())
    p._stability_reference = Uref
    with (case / "run.log").open("a") as log, contextlib.redirect_stdout(log):
        result = evolve(p, scheme, dt, count, rtol=0, atol=2e-11, _history=history)
    oldU, oldL, next_step = result["_history"]
    np.savez(case / "checkpoint.npz", U=result["U"], L=result["L"],
             Uold=oldU, Lold=oldL, next_step=next_step)
    diagnostics_path = case / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else []
    diagnostics.extend(result["diagnostics"])
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))
    t = (next_step - 1) * dt
    dx = p.DxU @ result["U"] + p.DxB @ p.g0 + p.DxL @ result["L"]
    dy = p.DyU @ result["U"] + p.DyB @ p.g0 + p.DyL @ result["L"]
    row = dict(control=control, scheme=scheme, dt=dt, time=t, amplitude=amplitude,
               status="completed" if next_step > total_steps else "checkpoint",
               velocity_change_inf=float(np.max(abs(result["U"] - Uref))),
               nodal_divergence_inf=float(np.max(abs(dx[::2] + dy[1::2]))),
               max_constraint=max(d["constraint_residual_inf"] for d in diagnostics),
               max_nonlinear_residual=max(d["nonlinear_residual_inf"] for d in diagnostics),
               newton_corrections=sum(d["newton_iterations"] for d in diagnostics))
    if control == "pressure":
        row["pressure_gradient_error_inf"] = float(np.max(abs(result["gradient"] - Lref)))
    else:
        row["ghia"] = reference_at_nodal_lines(p.points, result["U"])
    if amplitude:
        row["perturbation_gain"] = row["velocity_change_inf"] / amplitude
    (case / "result.json").write_text(json.dumps(row, indent=2))
    np.savez(case / "fields.npz", U=result["U"], L=result["L"], gradient=result["gradient"])
    print(json.dumps({k: v for k, v in row.items() if k != "ghia"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--scheme", choices=["oseen", "oseen_newton"], default="oseen")
    parser.add_argument("--dt", type=float, default=.00025)
    parser.add_argument("--control", choices=["hold", "pressure", "perturb"], default="perturb")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--end", type=float, default=.1)
    parser.add_argument("--amplitude", type=float, default=1e-3)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.prepare:
            prepare(args.source, args.cells, args.output)
        else:
            advance(args.source, args.cells, args.output, args.scheme, args.dt,
                    args.control, args.steps, args.end, args.amplitude)
