"""Collect completed characteristic/Oseen experiments without rerunning solves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .characteristic_cavity import paired_difference


def read(path):
    return json.loads(Path(path).read_text())


def collect(root):
    cases, pairs, comparisons, timestep = [], [], [], []
    for mesh in ("coarse169", "fine289"):
        char = root / "characteristic" / mesh
        oseen = root / "oseen_be" / mesh
        be_paired = root / "oseen_be_paired" / mesh
        for dt in (.00025, .000125):
            hold = char / f"hold_characteristic_{dt}"
            perturb = char / f"perturb_characteristic_{dt}"
            paired = paired_difference(hold, perturb, root / "paired" / f"{mesh}_{dt}")
            be_pair = paired_difference(be_paired/f"hold_be_oseen_{dt}",
                be_paired/f"perturb_be_oseen_{dt}", root/"paired"/f"{mesh}_{dt}_be_oseen")
            pairs.append(dict(mesh=mesh, dt=dt, be_oseen_pair=be_pair, **paired))
            for method, directory, controls in (("characteristic", char, ("startup", "hold", "perturb", "pressure")),
                                                ("be_oseen", oseen, ("startup", "hold", "perturb", "pressure"))):
                for control in controls:
                    current_directory = be_paired if method == "be_oseen" and control in ("hold", "perturb") else directory
                    folder = current_directory / f"{control}_{method}_{dt}"
                    result = read(folder / "result.json")
                    if result["status"] == "checkpoint":
                        raise ValueError(f"Incomplete case: {folder}")
                    diagnostics = read(folder / "diagnostics.json")
                    assert [r["step"] for r in diagnostics] == list(range(1, len(diagnostics)+1))
                    cases.append(dict(mesh=mesh, evidence=str(folder), **result))
            for control in ("startup", "perturb", "pressure"):
                be_directory = be_paired if control == "perturb" else oseen
                folders = [char/f"{control}_characteristic_{dt}", be_directory/f"{control}_be_oseen_{dt}"]
                results = [read(f/"result.json") for f in folders]
                row = dict(mesh=mesh, dt=dt, control=control,
                           characteristic_time=results[0]["time"], oseen_time=results[1]["time"])
                if abs(results[0]["time"]-results[1]["time"]) < 1e-13:
                    values = [np.load(f/"fields.npz") for f in folders]
                    row["velocity_difference_inf"] = float(np.max(abs(values[0]["U"]-values[1]["U"])))
                else:
                    row["comparison"] = "Different stopping times; endpoint fields not subtracted"
                comparisons.append(row)
        for control in ("startup", "hold", "perturb", "pressure"):
            folders = [char/f"{control}_characteristic_{dt}" for dt in (.00025, .000125)]
            results = [read(f/"result.json") for f in folders]
            row = dict(mesh=mesh, control=control, times=[r["time"] for r in results])
            if abs(results[0]["time"]-results[1]["time"]) < 1e-13:
                values = [np.load(f/"fields.npz") for f in folders]
                row["velocity_difference_inf"] = float(np.max(abs(values[0]["U"]-values[1]["U"])))
            timestep.append(row)
    assert len(cases) == 32
    mpi = {}
    for cells in (12, 16):
        cache = Path(f"outputs/ns_characteristic_reconstruction_20261030/cells{cells}")
        reports = list(cache.glob("*/build.json"))
        if len(reports) != 1:
            raise ValueError("Select the uniquely validated factor cache")
        mpi[str(cells)] = read(reports[0])
    summary = dict(cases=cases, paired_perturbations=pairs, method_comparisons=comparisons,
                   characteristic_timestep_comparisons=timestep, mpi=mpi,
                   previous_bdf2_comparisons=read(root/"oseen_be/summary.json")["BDF2_comparison"])
    (root/"SUMMARY.json").write_text(json.dumps(summary, indent=2))

    lines = ["# First-order characteristics with corrected 37-point mixed LHI", "",
        "Date: 2026-10-30. Implemented and tested with three Sol agents. The method retains local pressure elimination and the same corrected mixed reconstruction; only convection time treatment changes. The finer spatial configuration was already rejected for a positive mode and is intentionally retained as a negative control. No spatial coefficients were retuned in this study.", "",
        "The characteristics implementation passes reconstruction and pressure controls, but does not repair the rejected fine-grid operator. Coarse paired perturbations remain bounded through T=.1 (final gains about .927); fine gains reach 119.17/103.70 through T=.01, closely following BE-Oseen. Both fine startup runs hit the speed bound. Coarse startup velocities differ from BE-Oseen by 1.19e-5/5.95e-6 at the two steps. These transient runs do not establish improved steady Ghia accuracy.", "",
        "## Method and fixed configuration", "",
        "At each interior node, X=x-dt*Uold. Evaluate transported=F_owner(Uold,gold,Lold)(X) with the original arrival-owner stencil and target-momentum exclusion. Solve [A,-M;I/dt,I][Unew,Lnew]=[-B*gnew;fnew+transported/dt]. The characteristic global matrix is factored once per foreground process/chunk. Feet outside the physical square are rejected without clipping or an exterior continuation. Boundary values enter the new algebraic equation directly.", "",
        "Unit square, Re=100, nu=.01, unit horizontal lid except stationary corner nodes, stationary other walls, zero force for cavity cases. Grids: 169/289 total nodes, 121/225 interior. Nearest 37 physical sites. q=cells/44; potentials exp(-10q²r²)+q⁷r⁷ and nu²q⁴exp(-q²r²). Base DF6/P1 plus the saved common-E DF6/P2 correction. Coarse maps are the early4 checkpoint from the previous study; fine maps use its rejected x=0 full-lid root. No corner subtraction is added in this comparison.", "",
        "Source factors are freshly built with MPI 8/C++ MPFR 80. Persistent native evaluation retains MPFR80 coefficient solves and analytic kernel/polynomial contraction, with eight OpenMP threads. Geometry, incoming global states, E/C coefficients and global sparse solves remain Float64. Coarse/fine factor construction uses 121/86 unique jobs, respectively. All owners retain the exact source observations. This is a research fitted-field evaluator using RBFLAB internals, independently checked against public LocalApproximation; it is not a new public API promise.", "",
        "## Validation", "",
        "For the tested random mixed observations, all arrival velocities matched nodal data exactly after Float64 output conversion on both grids. Saved-map derivative differences are at most 1.51e-12. Independent public MPFR 80 off-node velocity differences are at most 1.03e-16, and derivative differences at most 4.57e-15 (three selected owners per grid). Mixed polynomial and p=xy off-node controls pass. Characteristic-versus-frozen-Oseen one-step differences scale toward dt², as expected for two first-order schemes with the same semidiscrete limit. See ../ns_characteristic_audit_20261030/RESULTS.md for coverage and limitations. The focused Python suite passed 70 tests with 2 opt-in native tests skipped; broader actual native/MPI audits on both meshes were executed separately and are detailed in that report.", "",
        "## Cases and stopping rules", "",
        "Both dt=.00025 and .000125 are tested. Startup uses physical rest and targets T=.1. Coarse root perturbations have amplitude 1e-3 and target T=.1; fine perturbations have amplitude 1e-7 and target T=.01. Pressure controls use homogeneous walls, f=grad(xy), U=0, L=grad(xy), through T=.02. The numerical speed bound 10 stops a failed run and preserves its finite state; it is not a blowup theorem. Historical BDF2-Oseen comparisons are reused only at matching times/initial states and are distinct from the new backward-Euler Oseen controls.", "",
        "| Mesh | Method | Control | dt | Actual time | Status | Peak nodal speed |",
        "|---|---|---|---:|---:|---|---:|"]
    for r in cases:
        lines.append(f"| {r['mesh']} | {r['scheme']} | {r['control']} | {r['dt']} | {r['time']:.6g} | {r['status']} | {r.get('peak_nodal_speed',float('nan')):.7g} |")
    lines += ["", "## Stability: subtract the matching unperturbed trajectory", "",
        "A semidiscrete steady root is not generally an exact fixed point of a finite characteristic step. The reported characteristic gain below is therefore ||U_pert-U_hold||inf divided by its actual initial value, with identical dt, initial root, boundary data and input hashes. Distance from the original root is reported separately. A raw distance/root perturbation-amplitude ratio is not interpreted as amplification.", "",
        "| Mesh | dt | Characteristic final paired gain | Peak paired gain | Unperturbed root drift | BE-Oseen gain |",
        "|---|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['mesh']} | {p['dt']} | {p['final_gain_inf']:.9g} | {p['peak_gain_inf']:.9g} | {p['hold_root_distance_inf']:.4e} | {p['be_oseen_pair']['final_gain_inf']:.9g} |")
    lines += ["", "Only the tested perturbation direction and finite interval are covered. These results are not a general nonnormal-growth, basin-of-attraction or long-time stability certificate.", "",
        "## Nodal velocity comparisons with Ghia", "",
        "The reference table is linearly interpolated onto existing interior centerline mesh nodes; computed fields are not interpolated to off-node Ghia locations. The following are transient endpoint differences, not errors of demonstrated converged steady cavity solutions. Fine cases stopped at different times must not be compared as same-time fields.", "",
        "| Mesh | Method | Control | dt | Time | u RMS difference | v RMS difference |",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in cases:
        if r['control'] not in ('startup','perturb'):
            continue
        g=r['ghia']
        lines.append(f"| {r['mesh']} | {r['scheme']} | {r['control']} | {r['dt']} | {r['time']:.6g} | {g['u']['rms_difference']:.9g} | {g['v']['rms_difference']:.9g} |")
    lines += ["", "Exact same-time method/step differences, all controls, input fingerprints and failure records are in SUMMARY.json and the individual folders. Endpoint maps retain local reconstructed divergence cancellation; that does not establish global compatibility of overlapping fields.", "",
        "## Reproduction and artifacts", "",
        "Implementation and runnable CLI: research/navier_stokes_schemes/characteristic_cavity.py. Exact mixed evaluator: characteristic_reconstruction.py. Method description: research/navier_stokes_schemes/CHARACTERISTICS.md. Call the driver with --help for source/maps/parameters/cache/initial and chunk options; each completed case contains its full configuration and hashes. MPI source manifests, inputs and cached factors are in outputs/ns_characteristic_reconstruction_20261030. New folders preserve all older experiments. The coarse pilot without full per-step trajectories is excluded. There are 32 distinct characteristic/BE-Oseen cases in the aggregate; four original BE perturbation runs were repeated in the separate paired tree to retain complete traces, with their old fields preserved for parity checks."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    fig, axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax, mesh in zip(axes,("coarse169","fine289")):
        for dt in (.00025,.000125):
            p=np.load(root/"paired"/f"{mesh}_{dt}"/"paired_trajectory.npz")
            ax.plot(p['times'],p['gain_inf'],label=f"Characteristics dt={dt:g}")
            be=np.load(root/"paired"/f"{mesh}_{dt}_be_oseen"/"paired_trajectory.npz")
            ax.plot(be['times'],be['gain_inf'],"--",label=f"BE-Oseen dt={dt:g}")
        ax.set(title=mesh,xlabel="Time",ylabel="Nodal perturbation infinity-norm gain")
        if mesh=="fine289":ax.set_yscale('log')
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(root/"paired_stability.png",dpi=180);plt.close(fig)
    print(json.dumps(dict(cases=len(cases),pairs=pairs,comparisons=comparisons)),flush=True)


if __name__ == "__main__":
    collect(Path("outputs/ns_characteristic_comparison_20261030"))
