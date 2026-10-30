"""Summarize saved runs without executing solvers or rebuilding weights.

python -m research.navier_stokes_schemes.report --inputs run1 run2 --output reportdir
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np


def _number(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) else "—"


def _configuration_key(configuration):
    # Execution/output choices may differ; physical/spatial choices may not.
    excluded = {"output", "schemes", "dt", "end_time", "cells", "ranks",
                "linear_solver", "global_ordering", "global_threads", "native_timeout",
                "reuse_stencils", "maps_from"}
    return json.dumps({k: v for k, v in configuration.items() if k not in excluded},
                      sort_keys=True)


def _field_difference(first, second):
    """Compare saved fields only when node locations agree exactly."""
    paths = [Path(row["field_file"]) for row in (first, second)]
    if not all(path.is_file() for path in paths):
        return {"status": "missing_npz"}
    try:
        with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
            if a["points"].shape != b["points"].shape or not np.array_equal(a["points"], b["points"]):
                return {"status": "different_nodes"}
            differences = {}
            for label, key in (("velocity", "U"), ("gradient", "gradient")):
                if a[key].shape != b[key].shape:
                    return {"status": "different_field_shapes"}
                delta = (a[key]-b[key]).reshape(-1, 2)
                if not np.isfinite(delta).all():
                    return {"status": "nonfinite_field_difference"}
                lengths = np.linalg.norm(delta, axis=1)
                differences[label+"_difference_rms"] = float(np.sqrt(np.mean(lengths**2)))
                differences[label+"_difference_max"] = float(lengths.max())
            return {"status": "compared", **differences}
    except (OSError, KeyError, ValueError) as error:
        return {"status": "unreadable_npz", "reason": str(error)}


def build_report(inputs, output):
    rows, configurations = [], []
    for source in inputs:
        source = Path(source).resolve()
        result_file = source/"results.json" if source.is_dir() else source
        directory = result_file.parent
        configuration_file = directory/"configuration.json"
        config = json.loads(configuration_file.read_text(encoding="utf-8")) if configuration_file.exists() else {}
        config_id = len(configurations)
        configurations.append({"source": str(directory), "configuration": config})
        loaded = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected results list in {result_file}")
        for result in loaded:
            row = dict(result, source=str(directory), configuration_id=config_id)
            row["end_time"] = result.get("end_time", config.get("end_time"))
            if row["end_time"] is None:
                raise ValueError(f"Missing end_time in {result_file}; cannot group safely")
            row["field_file"] = str(directory/f"{row['scheme']}_cells{row['cells']}_dt{row['dt']}.npz")
            rows.append(row)

    temporal_groups = defaultdict(list)
    for row in rows:
        config = configurations[row["configuration_id"]]["configuration"]
        # Missing configuration does not authorize comparing unrelated runs.
        config_key = _configuration_key(config) if config else row["source"]
        temporal_groups[(row["end_time"], row["cells"], row["scheme"], config_key)].append(row)
    differences = []
    for (end_time, cells, scheme, _), group in temporal_groups.items():
        complete = sorted((r for r in group if r.get("status") == "completed"),
                          key=lambda r: r["dt"], reverse=True)
        for coarse, fine in zip(complete, complete[1:]):
            if coarse["dt"] == fine["dt"]:
                continue
            comparison = dict(end_time=end_time, cells=cells, scheme=scheme,
                              coarse_dt=coarse["dt"], fine_dt=fine["dt"],
                              coarse_source=coarse["source"], fine_source=fine["source"],
                              **_field_difference(coarse, fine))
            comparison["gradient_error_increased"] = (
                fine.get("gradient_rms", float("inf")) > coarse.get("gradient_rms", float("inf")))
            comparison["coarse_gradient_rms"] = coarse.get("gradient_rms")
            comparison["fine_gradient_rms"] = fine.get("gradient_rms")
            differences.append(comparison)

    readme = Path(__file__).with_name("README.md").resolve()
    try:
        readme_link = Path(os.path.relpath(readme, Path(output).resolve())).as_posix()
    except ValueError:  # Windows output on another drive.
        readme_link = readme.as_posix()
    node_counts = sorted({row.get("nodes", (row["cells"]+1)**2) for row in rows})
    mesh_scope = "Compared node counts: " + ", ".join(map(str, node_counts)) + ". "
    if max(node_counts) < 1000:
        mesh_scope += "These are coarse implementation checks, not the earlier fine-grid benchmark."
    else:
        mesh_scope += "Assess mesh and time refinement from the saved nodal fields and errors below."
    schemes = {row['scheme'] for row in rows}
    startup_note = ("The benchmark uses velocity offset 0 and homogeneous prescribed boundary "
                    "velocity. Exact analytic initial U,L need not satisfy the assembled discrete "
                    "mixed constraint exactly.")
    if schemes & {'imex', 'imex_midpoint'}:
        startup_note += (" Crank–Nicolson can retain algebraic startup oscillations. "
                         "This is a possible explanation for midpoint pressure behavior, "
                         "not a verified cause in these runs.")
    lines = ["# Navier–Stokes scheme comparison", "",
             "This report reads saved results and fields; it does not run experiments. "
             "See the [method definitions and limits]("+readme_link+").",
             "", "## Scope and interpretation", "",
             mesh_scope, "",
             "Finite completion and small algebraic residuals do not establish evolution "
             "stability. Velocity and physical pressure-gradient RMS errors measure "
             "interior nodal accuracy. There is no between-node or integral error claim.", "",
             startup_note, "",
             "Short runs can be sensitive to startup. Velocity and pressure-gradient "
             "temporal accuracy must be assessed separately.", "",
             "The runner requests MPI-managed local weight assembly with 80-digit MPFR "
             "arithmetic. Geometry and global sparse solves remain Float64. Saved "
             "configuration is listed below; local precision does not upgrade geometry "
             "or establish stability.", "",
             "End times are grouped separately. Temporal field differences compare "
             "matching saved nodes, mesh, scheme, physical configuration, and end time. "
             "They are differences of computed fields, not convergence orders or "
             "ratios of errors against exact data. A spatial-error floor can obscure "
             "temporal refinement.", "",
             "Seconds measure evolution only: the initial shared-map assembly is excluded "
             "and logged separately. Moving-query rebuilds in characteristics and per-step "
             "Hermite rebuilds in scheme 7 are included in their evolution times.", ""]
    time_groups = defaultdict(list)
    for row in rows:
        time_groups[row["end_time"]].append(row)
    for end_time, group in sorted(time_groups.items()):
        lines += [f"## End time T = {_number(end_time)}", "",
                  "| Cells | Nodes | Scheme | dt | Velocity RMS | Gradient RMS | Status | Seconds | Source |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in sorted(group, key=lambda r: (r["cells"], r["scheme"], -r["dt"], r["configuration_id"])):
            lines.append("| "+" | ".join(map(str, [row["cells"], row.get("nodes", "—"), row["scheme"],
                         _number(row["dt"]), _number(row.get("velocity_rms")),
                         _number(row.get("gradient_rms")), row.get("status", "unknown"),
                         _number(row.get("seconds")), row["configuration_id"]]))+" |")
        lines += [""]
        for row in group:
            if row.get("status") != "completed":
                error = str(row.get("error", "No saved error message")).replace("\n", " ")
                lines += [f"Failure: `{row['scheme']}`, cells={row['cells']}, dt={row['dt']}: {error}", ""]

    lines += ["## Temporal field differences", "",
              "| T | Cells | Scheme | Coarse dt | Fine dt | Velocity difference RMS | Gradient difference RMS | Field status |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for pair in differences:
        lines.append("| "+" | ".join(map(str, [_number(pair["end_time"]), pair["cells"], pair["scheme"],
                     _number(pair["coarse_dt"]), _number(pair["fine_dt"]),
                     _number(pair.get("velocity_difference_rms")), _number(pair.get("gradient_difference_rms")),
                     pair["status"]]))+" |")
    if not differences:
        lines += ["", "No matching completed time-step pairs were available."]
    lines += ["", "## Pressure-gradient observations", ""]
    midpoint = [pair for pair in differences if pair["scheme"] in ("imex", "imex_midpoint")
                and pair["gradient_error_increased"]]
    if midpoint:
        for pair in midpoint:
            lines += [f"For midpoint IMEX at T={_number(pair['end_time'])}, cells={pair['cells']}, "
                      f"reducing dt from {_number(pair['coarse_dt'])} to {_number(pair['fine_dt'])} "
                      f"increased physical pressure-gradient RMS error from {_number(pair['coarse_gradient_rms'])} "
                      f"to {_number(pair['fine_gradient_rms'])}. This is saved-data deterioration, "
                      "not evidence of temporal convergence or a general failure theorem.", ""]
    elif schemes & {'imex', 'imex_midpoint'}:
        lines += ["No midpoint pressure-gradient deterioration was identified in comparable saved "
                  "time-step pairs. Missing pairs do not validate midpoint pressure accuracy.", ""]
    if 'rotational' in schemes:
        lines += ["Rotational results report physical pressure after subtracting the reconstructed "
              "kinetic-energy gradient. The discrete mixed derivative maps need not obey "
              "an exact product rule, so equivalence of the continuous formulations does "
              "not imply matching discrete accuracy. Inspect its velocity and gradient "
                  "errors separately; completion does not establish an accuracy advantage.", ""]
    if 'imex_ars222' in schemes:
        lines += ['ARS222 endpoint pressure gradients are reconstructed from the last-stage '
                  'mixed observations. Assess their errors independently of velocity accuracy; '
                  'a second-order velocity method does not by itself validate endpoint pressure.', '']
    krylov = [row for row in rows if row.get('linear_solver', {}).get('method') in
              {'gmres', 'gmres_scaled'}]
    if krylov:
        lines += ["## Oseen linear-solver checks", "",
                  "GMRES solves the same coupled equations using a reused LU preconditioner. "
                  "An unresolved iteration triggers a fresh factorization. The last time step "
                  "is also solved directly; differences below compare that same linear system, "
                  "not independent full trajectories.", "",
                  "Scaled GMRES uses a diagonally equilibrated system and refreshes at the "
                  "BE/BDF2 transition. The table reports original unscaled residuals; the "
                  "convergence criterion is applied to the transformed equations. "
                  "Preconditioner factorization counts exclude the additional final direct check.", "",
                  "| Nodes | dt | Method | Preconditioner LU builds | Iterations | Maximum original linear residual | Final direct velocity difference | Final direct gradient difference |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in krylov:
            stats = row['linear_solver']
            last = row['diagnostics'][-1]
            values = [row['nodes'], row['dt'], stats['method'], stats['factorizations'], stats['iterations'],
                      stats['residual_max'], last.get('direct_check_velocity_max'),
                      last.get('direct_check_gradient_max')]
            lines.append('| ' + ' | '.join(value if isinstance(value, str) else _number(value)
                                         for value in values) + ' |')
        lines += ['']
    lines += ["## Saved configurations", ""]
    for index, config in enumerate(configurations):
        lines += [f"Source {index}: `{config['source']}`", "", "```json",
                  json.dumps(config["configuration"], indent=2), "```", ""]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    comparison = {"configurations": configurations, "results": rows,
                  "temporal_field_differences": differences,
                  "end_times": sorted(time_groups), "orders_estimated": False}
    (output/"comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Run directories or results.json paths")
    parser.add_argument("--output", required=True, help="New report directory")
    args = parser.parse_args()
    build_report(args.inputs, args.output)
