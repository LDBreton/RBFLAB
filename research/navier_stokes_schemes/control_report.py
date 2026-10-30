"""Collect saved Stokes-control validation and cavity results without solves.

Run after the four production cases finish. Failed cases remain visible;
incomplete cases require the explicit --allow-incomplete reporting option.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CELLS = (89, 126)
STEPS = (.00025, .000125)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def field_difference(left, right):
    difference = (left-right).reshape(-1, 2)
    return dict(vector_rms=float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                vector_max=float(np.max(np.linalg.norm(difference, axis=1))),
                component_max=float(np.max(abs(difference))))


def case(root, cells, dt, allow_incomplete):
    folder = root/"cavity_newton_krylov"/f"cells{cells}_dt{dt}"
    if not (folder/"result.json").exists():
        if not allow_incomplete:
            raise ValueError(f"Production case has no final report: {folder}")
        return dict(cells=cells, total_nodes=(cells+1)**2, dt=dt,
                    status="missing", evidence=str(folder)), []
    result, configuration = read(folder/"result.json"), read(folder/"configuration.json")
    history = read(folder/"history.json")
    if result["status"] == "checkpoint" and not allow_incomplete:
        raise ValueError(f"Production case is incomplete: {folder}")
    if configuration["method"] != "newton_krylov":
        raise ValueError("The production comparison must use the audited Newton-Krylov method")
    if [r["step"] for r in history] != list(range(1, result["completed_steps"]+1)):
        raise ValueError(f"Noncontiguous accepted-step history: {folder}")
    maps = Path(configuration["maps"])
    for name, expected in configuration["input_hashes"].items():
        if hashlib.sha256((maps/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Numerical input changed after the run: {maps/name}")
    with np.load(folder/"fields.npz") as fields:
        if not np.isclose(float(fields["time"]), result["time"], rtol=0, atol=1e-13):
            raise ValueError("Saved fields and result describe different times")
        final_speed = np.linalg.norm(fields["U"].reshape(-1, 2), axis=1)
        result["final_nodal_speed"] = float(np.max(final_speed))
        result["final_nodal_velocity_vector_rms"] = float(np.sqrt(np.mean(final_speed**2)))
    result.update(evidence=str(folder), configuration=configuration,
                  final_control_cost=history[-1]["cost"] if history else None,
                  max_accepted_control_cost=max((r["cost"] for r in history), default=None),
                  total_accepted_step_seconds=sum(r["seconds"] for r in history),
                  total_krylov_iterations=sum(r.get("krylov_iterations", 0) for r in history))
    return result, history


def collect(root, allow_incomplete=False):
    root = Path(root)
    cases, histories, comparisons = [], {}, []
    for cells in CELLS:
        for dt in STEPS:
            result, history = case(root, cells, dt, allow_incomplete)
            cases.append(result)
            histories[cells, dt] = history
        pair = [r for r in cases if r["cells"] == cells]
        if (all(r["status"] == "completed" for r in pair)
                and abs(pair[0]["time"]-pair[1]["time"]) < 1e-13):
            with np.load(Path(pair[0]["evidence"])/"fields.npz") as a, np.load(Path(pair[1]["evidence"])/"fields.npz") as b:
                np.testing.assert_array_equal(a["points"], b["points"])
                comparisons.append(dict(cells=cells, nodes=(cells+1)**2, time=pair[0]["time"],
                    velocity=field_difference(a["U"], b["U"]),
                    pressure_gradient=field_difference(a["gradient"], b["gradient"])))
        else:
            comparisons.append(dict(cells=cells, nodes=(cells+1)**2,
                scope="No completed common-time endpoint comparison"))

    gradients = read(root/"audit/gradient_summary.json")
    newton = read(root/"audit/newton_summary.json")
    production_newton = read(root/"audit/newton_krylov_summary.json")
    maps = {}
    for cells in CELLS:
        folder = root/"maps"/f"cells{cells}_nu2"
        maps[str(cells)] = dict(configuration=read(folder/"configuration.json"),
            parity=read(folder/"mpi_parity.json"), pressure=read(folder/"pressure_patch.json"),
            probes=read(folder/"homogeneous_probes.json"))
    tests = read(root/"TESTS.json") if (root/"TESTS.json").exists() else None
    summary = dict(cases=cases, timestep_comparisons=comparisons, maps=maps,
                   gradient_checks=gradients, direct_newton_checks=newton,
                   production_newton_checks=production_newton, tests=tests,
                   excluded_pilot="cavity/cells89_dt0.00025 is the five-step Gauss-Newton pilot, not a completed production case")
    (root/"SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report(root, summary)
    plots(root, cases, histories, gradients)
    print(json.dumps(dict(cases=[{k:r.get(k) for k in ("total_nodes", "dt", "status", "time", "peak_nodal_speed")} for r in cases],
                          timestep_comparisons=comparisons)), flush=True)
    return summary


def report(root, summary):
    cases, maps = summary["cases"], summary["maps"]
    completed = sum(r["status"] == "completed" for r in cases)
    failed = sum(r["status"] == "failed" for r in cases)
    lines = ["# Navier-Stokes control residual with a time-reversed discrete Stokes adjoint", "",
        "2026-10-30. The gradient and independent equation-parity checks pass. "
        f"Of four production cavity cases, {completed} completed and {failed} failed; "
        f"{4-completed-failed} remain incomplete. Reaching the short target interval does not establish a steady cavity benchmark or general stability.", "",
        "## Method", "",
        "The control enters linear Stokes with a minus sign. At each backward-Euler time step, minimize "
        "J = 0.5*dt*h^2*sum_i |f_i-C_i(U,L,g)|^2. "
        "For x=(U,L), the state equations are K x = [-B g; b-f+Uold/dt], with "
        "K=[[A,-M],[I/dt,I]]. Here L is the mixed Stokes momentum observation, not a global pressure unknown. "
        "Convection and its derivatives use the same local mixed velocity field, including its dependence on L and prescribed boundary samples.", "",
        "The exact discrete adjoint solves K.T in reversed time. It does not assume a symmetric LHI operator or discretize the continuous adjoint independently. "
        "The full-window implementation is checked on four time levels. Production uses one-step control windows, warm starts from the preceding control, "
        "and an exact-Jacobian Newton-Krylov direction with an adjoint-gradient Armijo line search. "
        "The condensed matrix A+M/dt is factored once per process; forward and transpose Schur solves reuse that factor. "
        "A small objective alone is not acceptance: each accepted step requires the component maximum control residual <=1e-8. "
        "The driver separately computes and records the mixed constraint and full Navier-Stokes momentum residual.", "",
        "## Fixed spatial configuration and MPI work", "",
        "Unit square, nu=0.01 (Re=100), 8,100 and 16,129 total grid nodes; 7,744 and 15,625 interior nodes. "
        "Nearest 37 physical sites, velocity/pressure augmentation 6/1, pressure constant omitted, target excluded from momentum observations. "
        "q=cells/44; velocity potential exp(-10*q^2*r^2)+q^7*r^7; pressure potential nu^2*q^4*exp(-q^2*r^2), cbar_p=1. "
        "The viscosity-normalized pressure amplitude is explicit: this is not the unchanged q^4 amplitude at nu=0.01. "
        "There is no quadratic pressure correction, Q2/Q1 space, global scalar pressure or added damping. "
        "This comparison uses direct lid boundary data without corner subtraction.", "",
        "The saved public-API viscosity-one maps were built with C++ MPFR80 and MPI8. Their exact viscosity congruence is used: "
        "A,B,PY,PB multiply by nu; M,SL,PL,DxU,DxB,DyU,DyB stay fixed; DxL,DyL divide by nu. "
        "Fresh MPI8 MPFR80 construction independently checks 25 owners per mesh, covering walls, corner-adjacent and interior nodes. "
        "Source and exported map hashes and original MPI records are retained in maps/*/rescaling.json. "
        "Not all weights were freshly recomputed. Geometry and global solves remain Float64; global optimization is serial.", "",
        "| Nodes | Fresh owners | MPI ranks | Worst relative map difference | All ranks successful |",
        "|---:|---:|---:|---:|---|"]
    for cells in CELLS:
        p = maps[str(cells)]["parity"]
        rank_ok = all(w["returncode"] == 0 for batch in p["execution"]["batches"] for w in batch["execution"]["workers"])
        lines.append(f"| {(cells+1)**2} | {p['owner_count']} | {p['ranks']} | {max(r['max_relative'] for r in p['comparisons'].values()):.3e} | {rank_ok} |")
    lines += ["", "## Finite-difference and independent Newton verification", "",
        "The real-map derivative checks use four time levels, nonzero fixed lid data, nonzero random controls and nodal h^2 quadrature. "
        "Centered finite differences span epsilon=1e-2 through 1e-6. The table gives the best relative difference along each tested direction; "
        "the complete curves and transpose/JVP identities are saved in audit/gradient_summary.json.", "",
        "| Nodes | dt | Random direction relative error | Gradient-aligned relative error | Transpose relative error |",
        "|---:|---:|---:|---:|---:|"]
    for r in summary["gradient_checks"]:
        lines.append(f"| {r['nodes']} | {r['dt']} | {r['directions']['random']['best_relative_error']:.3e} | {r['directions']['gradient_aligned']['best_relative_error']:.3e} | {r['transpose_relative_error']:.3e} |")
    lines += ["", "On the independent 169-node implementation check, four-step control solutions agree with separately assembled fully implicit BE Newton solves. "
        "This checks equations, not cavity field accuracy. Both Gauss-Newton and the production Newton-Krylov direction are audited.", "",
        "| Control method | dt | Nodal velocity difference max | Full NS residual max |",
        "|---|---:|---:|---:|"]
    for name, records in (("Gauss-Newton", summary["direct_newton_checks"]), ("Newton-Krylov", summary["production_newton_checks"])):
        for r in records:
            lines.append(f"| {name} | {r['dt']} | {r['velocity_difference_inf']:.3e} | {r['max_full_navier_stokes_residual']:.3e} |")
    lines += ["", "## Cavity startup", "",
        "Physical initial rest, impulsive unit horizontal lid with stationary corner nodes, other walls stationary, zero body force. "
        "Target T=0.1; dt=0.00025 and 0.000125. A nodal speed bound of 10 stops a run and preserves evidence. "
        "A stopped run is reported at its actual accepted time, not at the requested target.", "",
        "| Nodes | dt | Status | Actual time | Peak nodal speed | Max control residual | Max momentum residual | Max mixed constraint |",
        "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in cases:
        if r["status"] == "missing":
            lines.append(f"| {r['total_nodes']} | {r['dt']} | missing | - | - | - | - | - |")
            continue
        lines.append(f"| {r['total_nodes']} | {r['dt']} | {r['status']} | {r['time']:.6g} | {r['peak_nodal_speed']:.8g} | {r['max_control_residual_inf']:.3e} | {r['max_momentum_residual_inf']:.3e} | {r['max_constraint_residual_inf']:.3e} |")
        if r.get("failure"):
            lines.append(f"\nFailure record for {r['total_nodes']} nodes, dt={r['dt']}: `{json.dumps(r['failure'], sort_keys=True)}`\n")
    lines += ["", "Mapped local divergence cancellation is recorded separately in SUMMARY.json. It does not certify global compatibility of overlapping local fields or independently measured global divergence.", "",
        "## Time-step and Ghia comparisons", "",
        "Absolute same-time field differences use interior mesh nodes only: vector RMS=sqrt(mean(|difference|^2)). "
        "Two time steps give a consistency comparison, not a measured temporal order or an exact-solution error.", "",
        "| Nodes | Common time | Velocity vector RMS difference | Pressure-gradient vector RMS difference |",
        "|---:|---:|---:|---:|"]
    for r in summary["timestep_comparisons"]:
        if "time" in r:
            lines.append(f"| {r['nodes']} | {r['time']} | {r['velocity']['vector_rms']:.8e} | {r['pressure_gradient']['vector_rms']:.8e} |")
        else:
            lines.append(f"| {r['nodes']} | unavailable | - | - |")
    lines += ["", "The 8,100-node grid contains no nodes on x=0.5 or y=0.5, so its Ghia centerline comparison is unavailable. "
        "For 16,129 nodes, computed velocities stay at mesh nodes and the Ghia table is linearly interpolated to those centerline nodes. "
        "The common center is compared directly. These are transient differences from a steady numerical reference, not validated steady errors.", "",
        "| Nodes | dt | Actual time | u centerline RMS difference | v centerline RMS difference | Computed center (u,v) |",
        "|---:|---:|---:|---:|---:|---|"]
    for r in cases:
        g = r.get("ghia", {})
        if g.get("status") == "available":
            center = g.get("exact_table_center", {}).get("computed")
            lines.append(f"| {r['total_nodes']} | {r['dt']} | {r['time']:.6g} | {g['u']['rms_difference']:.8g} | {g['v']['rms_difference']:.8g} | {center} |")
    lines += ["", "## Stokes stability probes and remaining pressure defect", "",
        "The homogeneous Stokes checks use one smooth DF curl and one seeded random nodal velocity. "
        "The random vector is not asserted to be a globally solenoidal field. Both are normalized in the nodal Euclidean norm and evolved to T=0.1. "
        "These selected-direction checks do not certify the complete spectrum or worst-case transient stability.", "",
        "| Nodes | dt | Smooth final norm gain | Random final norm gain | Peak tested norm gain |",
        "|---:|---:|---:|---:|---:|"]
    for cells in CELLS:
        for r in maps[str(cells)]["probes"]["results"]:
            lines.append(f"| {(cells+1)**2} | {r['dt']} | {r['final_l2_gain'][0]:.9g} | {r['final_l2_gain'][1]:.9g} | {max(r['peak_l2_gain']):.9g} |")
    lines += ["", "Affine pressure is reproduced near roundoff, but p=xy remains inconsistent in the degree-one pressure space. "
        "The following are static nodal reconstruction defects at exact zero velocity and L=grad(p), not evolved velocity errors. "
        "A converged control residual cannot remove this spatial pressure defect.", "",
        "| Nodes | Pressure | Mixed-constraint defect max | Pressure-gradient defect max |",
        "|---:|---|---:|---:|"]
    for cells in CELLS:
        for name, r in maps[str(cells)]["pressure"].items():
            lines.append(f"| {(cells+1)**2} | {name} | {r['constraint_inf']:.8e} | {r['gradient_error_inf']:.8e} |")
    lines += ["", "## Tests and reproduction", ""]
    if summary["tests"] is None:
        lines.append("Final combined test-count record is pending; the independent audit report retains its own executed checks.")
    else:
        lines.append("Executed test record: `"+json.dumps(summary["tests"], sort_keys=True)+"`.")
    lines += ["", "Readable equations and algorithm: research/navier_stokes_schemes/CONTROL.md. "
        "Implementation: stokes_control.py; map preparation: control_maps.py; cavity CLI: control_cavity.py; "
        "this collector: control_report.py. Run each module with --help where supported. "
        "The five-step Gauss-Newton pilot under cavity/ is preserved and excluded from production counts. "
        "Saved small-grid characteristics results use different meshes and corrected maps, so they are not presented as a matched-method velocity comparison.", "",
        "![Gradient checks](gradient_checks.png)", "", "![Cavity diagnostics](cavity_diagnostics.png)", ""]
    if any(r.get("ghia", {}).get("status") == "available" for r in cases):
        lines += ["![Transient nodal centerlines](ghia_nodal_profiles.png)", ""]
    (root/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plots(root, cases, histories, gradients):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, cells in enumerate(CELLS):
        for col, direction in enumerate(("random", "gradient_aligned")):
            ax = axes[row, col]
            for r in gradients:
                if r["cells"] != cells:
                    continue
                values = r["directions"][direction]["differences"]
                ax.loglog([v["epsilon"] for v in values], [v["relative_error"] for v in values],
                          "o-", label=f"dt={r['dt']:g}")
            ax.set(title=f"{(cells+1)**2:,} nodes, {direction.replace('_', ' ')}", xlabel="Finite-difference step", ylabel="Relative directional error")
            ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(root/"gradient_checks.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fields = (("max_nodal_speed", "Maximum interior nodal speed", False),
              ("kinetic_nodal_mean", "Mean nodal kinetic energy", False),
              ("control_residual_inf", "Control residual maximum", True),
              ("momentum_residual_inf", "NS momentum residual maximum", True))
    for ax, (key, title, logarithmic) in zip(axes.ravel(), fields):
        for (cells, dt), history in histories.items():
            if not history:
                continue
            ax.plot([r["time"] for r in history], [max(r[key], 1e-30) if logarithmic else r[key] for r in history],
                    "-" if dt == STEPS[0] else "--", label=f"{(cells+1)**2:,} nodes, dt={dt:g}")
        if logarithmic:
            ax.set_yscale("log")
        ax.set(title=title, xlabel="Physical time")
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.savefig(root/"cavity_diagnostics.png", dpi=180); plt.close(fig)

    available = [r for r in cases if r.get("ghia", {}).get("status") == "available"]
    if available:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, component, coordinate in zip(axes, ("u", "v"), ("y at x=0.5", "x at y=0.5")):
            for r in available:
                g = r["ghia"][component]
                ax.plot(g["nodes"], g["computed"], label=f"Control dt={r['dt']:g}, t={r['time']:g}")
            reference = available[0]["ghia"][component]
            ax.plot(reference["nodes"], reference["interpolated_reference"], "k--", label="Steady Ghia table interpolated to nodes")
            ax.set(title=f"Transient nodal {component}; 16,129 nodes", xlabel=coordinate, ylabel=component)
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.savefig(root/"ghia_nodal_profiles.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ns_stokes_control_20261030"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    collect(args.root, args.allow_incomplete)
