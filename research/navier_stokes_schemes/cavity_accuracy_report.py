"""Collect the bounded cavity-accuracy study without rerunning any PDE solve."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    names = ("early4", "strong", "fine_rejected")
    configurations, cases, comparisons = {}, [], []
    for name in names:
        root = Path(f"outputs/ns_accuracy_verify_{name}_20261029")
        configurations[name] = json.loads((root / "configuration.json").read_text())
        for path in sorted(root.glob("*/result.json")):
            row = json.loads(path.read_text())
            diagnostics = json.loads((path.parent / "diagnostics.json").read_text())
            assert row["status"] == "completed"
            assert [d["step"] for d in diagnostics] == list(range(1, len(diagnostics)+1))
            row.update(candidate=name, evidence=str(path))
            cases.append(row)
        for scheme in ("oseen", "oseen_newton"):
            paths = [root / f"perturb_{scheme}_{dt}" for dt in (.00025, .000125)]
            fields = [np.load(path / "fields.npz") for path in paths]
            comparisons.append(dict(candidate=name, scheme=scheme,
                nodal_velocity_dt_difference_inf=float(np.max(abs(fields[0]["U"]-fields[1]["U"]))),
                nodal_gradient_dt_difference_inf=float(np.max(abs(fields[0]["gradient"]-fields[1]["gradient"])))))
    assert len(cases) == 32
    baseline = json.loads(Path("outputs/ns_equilibrium_nullspace_lifting_20261029/results.json").read_text())
    result = dict(configurations=configurations, cases=cases, timestep_comparisons=comparisons,
                  previous_baseline_reference=baseline["reference"])
    (output / "results.json").write_text(json.dumps(result, indent=2))

    lines = ["# Cavity accuracy study: modest coarse-grid improvement, refinement failure", "",
        "Date: 2026-10-29. Three Sol agents assisted the mathematical, refinement and independent audits. All field comparisons are at mesh nodes. No reference values entered the coefficient objective; the early checkpoint was selected after inspecting Ghia diagnostics, so this is benchmark-informed selection, not independent validation of a general method.", "",
        "## Configuration", "",
        "Unit square, Re=100, nu=0.01, unit lid excluding stationary corner nodes, other walls stationary, zero body force. Coarse/fine grids have 169/289 total nodes (121/225 interior); nearest37 sites. Full scaling q=cells/44; velocity potential exp(-10 q² r²)+q⁷r⁷ and pressure potential nu² q⁴ exp(-q²r²). Base DF6/P1 LHI plus common-E DF6/P2 reproduction. Pressure remains locally eliminated; no FE representation or separate global pressure unknown. The selected comparison candidates impose the lid directly; corner subtraction was tested separately and rejected. Local sources are C++ MPFR80/MPI8, with Float64 geometry, corrections and global solves.", "",
        "## Actual steady roots", "",
        "Ghia RMS is the absolute component difference on the existing interior centerline nodes, with the reference table linearly interpolated to those nodes. Different meshes have different comparison-node sets; no convergence order is claimed.", "",
        "| Candidate | Nodes | u RMS | v RMS | Leading modal real part | Assessment |",
        "|---|---:|---:|---:|---:|---|"]
    ref = baseline["reference"]
    lines.append(f"| Previous baseline | 169 | {ref['u']['rms_difference']:.9f} | {ref['v']['rms_difference']:.9f} | {baseline['moving']['real']:.8f} | Previous bounded stability evidence |")
    for name in names:
        config = configurations[name]; ref = config["ghia"]
        assessment = "Rejected: positive modes" if name == "fine_rejected" else "Bounded local perturbations; weak margin" if name == "early4" else "Stable tests but poorer accuracy"
        lines.append(f"| {name} | {config['total_nodes']} | {ref['u']['rms_difference']:.9f} | {ref['v']['rms_difference']:.9f} | {config['spectrum']['real']:.8f} | {assessment} |")
    lines += ["", "The early correction improves the two RMS differences by approximately 0.86% and 0.33%, which is small relative to the remaining error. Its exact center-point component errors worsen: approximately (0.03787,0.01271) becomes (0.04028,0.01484). Thus even the modest improvement is not uniform across nodes. The stronger correction reduces the central independent FD4 divergence objective by 72.3% but worsens both Ghia differences (11.34% and 6.81%) and the cell-flux holdout. Reducing this consistency diagnostic alone does not solve the accuracy problem.", "",
        "## Both time steps and nonlinear solvers", "",
        "BE startup/BDF2, Oseen and Oseen predictor plus Newton, dt=0.00025 and 0.000125. These are checks about saved actual equilibria, not startup-to-steady cavity benchmarks. For the coarse candidates, finite leading-mode perturbations of amplitude 1e-3 run to T=0.1; equilibrium holds and p=xy controls run to T=0.02. Newton uses rtol=0 and atol=2e-11 and performs one correction at every perturbed step (400/800). For the rejected fine candidate, a 1e-7 perturbation runs only to T=0.01; all four runs show growth, so longer evolution is unnecessary to reject it.", "",
        "| Candidate | Scheme | dt | T | Perturbation gain | u RMS at endpoint | v RMS at endpoint |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in cases:
        if row["control"] != "perturb":
            continue
        ref = row["ghia"]
        lines.append(f"| {row['candidate']} | {row['scheme']} | {row['dt']} | {row['time']} | {row['perturbation_gain']:.8f} | {ref['u']['rms_difference']:.9f} | {ref['v']['rms_difference']:.9f} |")
    lines += ["", "The fine growth is consistent with its positive semidiscrete eigenvalue (+451.2694); exp(lambda*T) is approximately 91.17 at T=0.01. The independent BE-start/BDF2 scalar-mode predictions are 93.64706 and 91.79953, matching the two Newton field evolutions within about 1e-11. Halving dt does not cure it. Newton solves its nonlinear algebra accurately while the physical discrete perturbation grows. For early/strong candidates, all per-step perturbation norms stay below their initial values through T=0.1. This is one leading-mode direction, not a general nonnormal-growth bound.", "",
        "| Candidate | Scheme | Coarse/fine nodal velocity difference, infinity norm |",
        "|---|---|---:|"]
    for row in comparisons:
        lines.append(f"| {row['candidate']} | {row['scheme']} | {row['nodal_velocity_dt_difference_inf']:.3e} |")
    lines += ["", "Equilibrium-hold Ghia RMS values agree with the saved roots at both steps; the maximum hold drift is below 5.6e-13. These very small temporal differences do not establish spatial accuracy.", "",
        "## Pressure and divergence", ""]
    for name in names:
        controls = [r for r in cases if r["candidate"] == name and r["control"] == "pressure"]
        lines.append(f"- {name}: p=xy maximum nodal velocity {max(r['velocity_change_inf'] for r in controls):.3e}; pressure-gradient component maximum error {max(r['pressure_gradient_error_inf'] for r in controls):.3e} through T=0.02.")
    lines += ["", "All 32 saved endpoint fields retain exact Float64 cancellation of the local reconstructed nodal divergence. This concerns the owner reconstruction at each node. Independent FD and cell-flux diagnostics examine compatibility of nodal data under other reconstructions; their nonzero values include truncation and corner effects. Shared sample values do not force neighboring local fields to have identical derivatives. The local kernel property has not been lost; global compatibility does not follow from it alone.", "",
        "## Fresh MPI and rejected searches", "",
        "Every one of the 225 fine-grid owners was freshly recomputed with eight MPI ranks, C++ and 80 digits: 29 checkpointed batches, 225 unique jobs, zero cache hits, all worker return codes zero. Across all 12 source blocks the largest difference from saved weights is 2.3543e-71. See ../ns_cavity_accuracy_mpi_20261029/SUMMARY.json. Corrected local maps and coefficient optimization remain Float64. The independent focused regression suite passes 32 tests, including three new driver-level tests for nonlinear checkpoint history, affine homogeneous pressure controls, and complex descriptor-mode normalization.", "",
        "The finer-grid failures include transferred coarse coefficients, full corner subtraction, and 19 accepted true-root spectral correction steps at a partial lid amplitude 0.325. That partial-lid branch remains unstable and is not compared with the unit-lid Ghia benchmark. The full-lid negative control reported above is a genuinely converged root. Smooth higher-moment improvement barely changed Ghia errors; a corner correction with fixed weights improved errors substantially but reintroduced growing modes. All failures are preserved.", "",
        "## Evidence and implementation", "",
        "- Coarse optimization and gradient audit: ../ns_accuracy_divergence_final_audit_20261029/REPORT.md; early checkpoint ../ns_accuracy_divergence_early4_20261029/.",
        "- Independent nodal compatibility audit: ../ns_corner_accuracy_global_audit_20261029/REPORT.md.",
        "- Fine transfer and continuation: ../ns_refined_transfer_20261029/REPORT.md and ../ns_refined_lid_equilibrium_20261029/REPORT.md.",
        "- All 32 completed evolution cases, exact settings, fields and per-step diagnostics: ../ns_accuracy_verify_early4_20261029/, ../ns_accuracy_verify_strong_20261029/, ../ns_accuracy_verify_fine_rejected_20261029/.",
        "- Aggregate results and timestep differences: results.json in this directory. Nodal coarse-grid comparison: centerlines.png.",
        "- Drivers: research/navier_stokes_schemes/cavity_accuracy_verify.py, cavity_accuracy_mpi.py and cavity_accuracy_report.py; optimization and audit helpers are isolated research modules.", "",
        "This study did not obtain a robust accuracy improvement under refinement. The early coarse improvement is an exploratory result with post-hoc benchmark selection; no accurate cavity default, convergence order, long-time stability or global smooth reconstruction is established."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, component in zip(axes, ("u", "v")):
        ref = baseline["reference"][component]
        ax.plot(ref["nodes"], ref["interpolated_reference"], "ko--", label="Ghia at mesh nodes", markersize=3)
        ax.plot(ref["nodes"], ref["computed"], "s-", label="Previous stable baseline", markersize=3)
        for name, label in (("early4", "Early correction"), ("strong", "Stronger consistency correction")):
            ref = configurations[name]["ghia"][component]
            ax.plot(ref["nodes"], ref["computed"], ".-", label=label)
        ax.set(xlabel="y at x=0.5" if component == "u" else "x at y=0.5", ylabel=component,
               title="169-node actual equilibria")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "centerlines.png", dpi=170)
    plt.close(fig)
    print(json.dumps(dict(cases=len(cases), output=str(output), comparisons=comparisons)), flush=True)


if __name__ == "__main__":
    collect(Path("outputs/ns_cavity_accuracy_comparison_20261029"))
