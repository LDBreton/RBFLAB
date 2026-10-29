"""Numerical execution of generic mixed functional local requests.

Python is the reference path. C++ evaluates each field kernel and contracts
mixed differential rows natively. Optional MPI schedules the same requests;
it does not move global equation assembly or solves onto MPI.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from .lhi_backends import PythonBackend, CppBackend


@dataclass(frozen=True)
class LocalExecution:
    """Schedule mixed local jobs, independently of their numerical backend.

    scheduler: ``serial`` or real ``mpi`` execution. MPI requires CppBackend.
    ranks: Number of MPI ranks (idle ranks are allowed for small requests).
    launcher: Optional MPI command prefix, e.g. ("mpiexec", "--oversubscribe").
    wsl_distribution: On Windows, explicitly select a WSL MPI launcher. If no
        native MPI exists, the default Windows fallback uses Ubuntu.
    reuse: ``none`` or ``exact`` identical-request reuse. This does not identify
        approximately equal or translated geometries.
    """
    scheduler: str = "serial"
    ranks: int = 1
    launcher: tuple | None = None
    wsl_distribution: str | None = None
    reuse: str = "none"

    def __post_init__(self):
        if self.scheduler not in ("serial", "mpi"):
            raise ValueError("LocalExecution.scheduler must be serial or mpi")
        if type(self.ranks) is not int or self.ranks < 1:
            raise ValueError("LocalExecution.ranks must be a positive integer")
        if self.scheduler == "serial" and self.ranks != 1:
            raise ValueError("Multiple ranks require scheduler='mpi'")
        if self.scheduler == "serial" and (self.launcher is not None or self.wsl_distribution is not None):
            raise ValueError("MPI launcher/WSL options require scheduler='mpi'")
        if self.launcher is not None and (not isinstance(self.launcher, (tuple, list))
                or not self.launcher or any(not isinstance(x, str) or not x for x in self.launcher)):
            raise ValueError("launcher must be a nonempty command sequence")
        if self.wsl_distribution is not None and (not isinstance(self.wsl_distribution, str) or not self.wsl_distribution):
            raise ValueError("wsl_distribution must be a nonempty name")
        if self.reuse not in ("none", "exact"):
            raise ValueError("reuse must be none or exact")
        if self.launcher is not None:
            object.__setattr__(self, "launcher", tuple(self.launcher))


def _orders(owner):
    orders = []
    def order(rows, f):
        return max((sum(alpha) for row in rows for (field, component), op in row.items()
                    if field == f for alpha, coefficient in op.terms), default=0)
    for f, field in enumerate(owner.fields):
        value = max((max(order(j["source_rows"], f), order(j["target_rows"], f))
                     + order(j["trial_rows"], f) for j in owner.jobs), default=0)
        value += 2 if field.vector else 0
        if value > 6:
            raise NotImplementedError("Mixed native kernels support derivatives through order six")
        orders.append(value)
    return orders


def _compile(owner):
    from .kernel_compiler import as_bound, request_cache, record_request, header, build, ROOT, PRELUDE
    from .native_toolchain import executable_path
    bounds = [as_bound(f.kernel) for f in owner.fields]
    orders = _orders(owner)
    runtime = (ROOT / "cpp/mixed_functional_local.hpp").read_text(encoding="utf-8-sig")
    arithmetic = "mpfr" if owner.precision.local_digits else "float64"
    index, cached = request_cache("mixed_functional_local", tuple(b.family for b in bounds),
        arithmetic, owner.backend.cache_dir, dimension=owner.dimension, orders=orders,
        vectors=[f.vector for f in owner.fields],
        runtime=hashlib.sha256(runtime.encode()).hexdigest(),
        bridge=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if cached:
        return executable_path(cached / "kernel"), bounds
    pieces = [PRELUDE, "\nconstexpr int DIM=%d,FIELDS=%d;\n" % (owner.dimension, len(bounds))]
    alphas = []
    for f, (bound, order) in enumerate(zip(bounds, orders)):
        code, alpha = header(bound.family, owner.dimension, order, "field%d" % f,
                             arithmetic=arithmetic)
        pieces.append(code)
        alphas.append(alpha)
    dispatch = ["\nReal mixed_radial(int field,const std::vector<Real>&z,const std::vector<int>&a,const std::vector<Real>&params){"]
    for f, alpha in enumerate(alphas):
        dispatch.append("if(field==%d){" % f)
        for i, multi in enumerate(alpha):
            condition = " && ".join("a[%d]==%d" % (j, v) for j, v in enumerate(multi))
            coords = ",".join("z[%d]" % j for j in range(owner.dimension))
            dispatch.append("if(%s) return field%d::eval(%d,%s,params);" % (condition, f, i, coords))
        dispatch.append("}")
    dispatch.append('throw std::runtime_error("Unprepared mixed derivative/field");}\n')
    pieces += ["\n".join(dispatch), runtime]
    directory = build("".join(pieces), dict(kind="mixed_functional_local",
        dimension=owner.dimension, orders=orders), arithmetic, owner.backend.cache_dir)
    record_request(index, directory)
    return executable_path(directory / "kernel"), bounds


def _serialize(owner, jobs, bounds, inspect=False):
    a = owner.arithmetic
    digits = owner.precision.local_digits or 17
    num = lambda value: a.ctx.nstr(a.number(value), digits+5) if a.ctx else format(float(value), ".17g")
    field_lines = []
    for field, bound in zip(owner.fields, bounds):
        field_lines.append(" ".join([str(int(field.vector)), str(field.components),
                                    str(len(bound.values)), *[num(v) for v in bound.values]]))
    def row_line(row):
        terms = [(f, c, alpha, coefficient) for (f, c), op in sorted(row.items())
                 for alpha, coefficient in op.terms]
        return " ".join([str(len(terms))] + [value for f, c, alpha, coefficient in terms
            for value in [str(f), str(c), num(coefficient), *map(str, alpha)]])
    chunks = []
    for job in jobs:
        n, m, k = job["size"], job["polynomials"], len(job["target_rows"])
        if n != len(job["source_rows"]) or n != len(job["trial_rows"]):
            raise ValueError("Mixed source/trial scalar row counts differ")
        lines = ["%d %d %d %s" % (n, m, k, num(job["scale"]))]
        for key in ("source_points", "trial_points"):
            lines.extend(" ".join(num(v) for v in p) for p in job[key])
        for key in ("source_rows", "trial_rows"):
            lines.extend(row_line(row) for row in job[key])
        lines.extend(" ".join(num(v) for v in p) for p in job["target_points"])
        lines.extend(row_line(row) for row in job["target_rows"])
        for key, count in (("top", n), ("bottom", n), ("target_poly", k)):
            block = job[key]
            for i in range(count):
                lines.append(" ".join(num(block[i, j]) for j in range(m)))
        chunks.append("\n".join(lines)+"\n")
    return field_lines, chunks


def _input(owner, fields, chunks, inspect):
    digits = owner.precision.local_digits or 17
    return ("%d %d %d %d %d\n" % (digits, owner.backend.threads,
        int(owner.backend.compute_condition), int(inspect), len(chunks))
        + "\n".join(fields)+"\n"+"".join(chunks))


def _wsl(path):
    s = Path(path).resolve().as_posix()
    if len(s) < 3 or s[1] != ":":
        raise ValueError("WSL bridge requires a drive-qualified Windows path")
    return "/mnt/"+s[0].lower()+s[2:]


def _mpi(owner, executable, fields, chunks, folder, inspect):
    execution = owner.execution
    helper = Path(__file__).with_name("_mpi_local_rank.py")
    ranks = execution.ranks
    paths = [(folder / ("rank%d.in" % r), folder / ("rank%d.out" % r)) for r in range(ranks)]
    assignments = [list(range(r, len(chunks), ranks)) for r in range(ranks)]
    for (source, target), ids in zip(paths, assignments):
        if ids:
            source.write_text(_input(owner, fields, [chunks[i] for i in ids], inspect), encoding="ascii")
    use_wsl = os.name == "nt" and (execution.wsl_distribution is not None
        or (execution.launcher is None and shutil.which("mpiexec") is None))
    config = dict(ranks=ranks, executable=_wsl(executable) if use_wsl else str(executable),
        inputs=[str(p.resolve()) for p, q in paths], outputs=[str(q.resolve()) for p, q in paths],
        counts=list(map(len, assignments)), timeout=owner.backend.timeout)
    (folder / "config.json").write_text(json.dumps(config))
    if use_wsl:
        if shutil.which("wsl.exe") is None:
            raise FileNotFoundError("MPI requires native mpiexec or a configured WSL distribution")
        prefix = ["wsl.exe", "-d", execution.wsl_distribution or "Ubuntu", "--cd", "/tmp", "-e"]
        launcher = list(execution.launcher or ("mpiexec", "--oversubscribe"))
        command = [*prefix, *launcher, "-n", str(ranks), "python3", _wsl(helper), _wsl(folder)]
    else:
        launcher = list(execution.launcher or ("mpiexec",))
        if shutil.which(launcher[0]) is None:
            raise FileNotFoundError("MPI launcher not found: "+launcher[0])
        command = [*launcher, "-n", str(ranks), sys.executable, str(helper), str(folder)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=owner.backend.timeout)
    if run.returncode:
        raise RuntimeError("Mixed MPI launch failed: "+run.stderr+run.stdout)
    restored = [None]*len(chunks)
    records = []
    for rank, ((source, target), ids) in enumerate(zip(paths, assignments)):
        record = json.loads((folder / ("rank%d.json" % rank)).read_text())
        if (record["rank"], record["size"], record["returncode"], record["jobs"]) != (rank, ranks, 0, len(ids)):
            raise RuntimeError("Invalid MPI rank completion record")
        records.append({key: record[key] for key in ("rank", "size", "hostname", "seconds", "returncode", "jobs")})
        lines = target.read_text().splitlines() if ids else []
        if len(lines) != len(ids):
            raise RuntimeError("Incomplete MPI rank output")
        for local_id, (global_id, line) in enumerate(zip(ids, lines)):
            first, rest = line.split(" ", 1)
            if int(first) != local_id:
                raise RuntimeError("Wrong MPI local job order")
            restored[global_id] = str(global_id)+" "+rest
    return restored, dict(scheduler="mpi", ranks=ranks, workers=records,
                          command=command, bridge="wsl" if use_wsl else "native")


def _cpp(owner, jobs, inspect=False):
    a, backend = owner.arithmetic, owner.backend
    if backend.executable is not None:
        raise ValueError("Mixed local backend uses its compiled cached executable")
    executable, bounds = _compile(owner)
    fields, chunks = _serialize(owner, jobs, bounds, inspect)
    mapping = list(range(len(chunks)))
    # Inspection reconstructs one mathematical request; it does not replace the
    # measured execution report of the original build or relaunch its MPI ranks.
    configured = getattr(owner, "execution", None)
    execution = LocalExecution() if inspect or configured is None else configured
    if execution.reuse == "exact" and not inspect:
        seen, unique, mapping = {}, [], []
        for chunk in chunks:
            if chunk not in seen:
                seen[chunk] = len(unique)
                unique.append(chunk)
            mapping.append(seen[chunk])
        chunks = unique
    before = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rbflab_mixed_") as tmp:
        folder = Path(tmp)
        if execution.scheduler == "mpi":
            owner.execution = execution
            lines, report = _mpi(owner, executable, fields, chunks, folder, inspect)
        else:
            source, target = folder / "input", folder / "output"
            source.write_text(_input(owner, fields, chunks, inspect), encoding="ascii")
            run = subprocess.run([str(executable), str(source), str(target)], capture_output=True,
                                 text=True, timeout=backend.timeout)
            if run.returncode:
                raise RuntimeError("Mixed C++ backend: "+run.stderr)
            lines = target.read_text().splitlines()
            report = dict(scheduler="serial", ranks=1)
    if len(lines) != len(chunks):
        raise RuntimeError("Incomplete mixed native output")
    if not inspect:
        owner.execution_info = dict(report, requested_jobs=len(jobs), unique_jobs=len(chunks),
            cache_hits=len(jobs)-len(chunks), numerical_backend="CppBackend",
            local_digits=owner.precision.local_digits, seconds=time.perf_counter()-before)
    unique_results = []
    first_jobs = [None]*len(chunks)
    for requested, unique in enumerate(mapping):
        if first_jobs[unique] is None:
            first_jobs[unique] = requested
    for index, line in enumerate(lines):
        data = line.split()
        if len(data) < 3:
            raise RuntimeError("Truncated mixed output")
        job_id, n, k = map(int, data[:3])
        job = jobs[first_jobs[index]]
        if job_id != index or n != job["size"]+job["polynomials"] or k != len(job["target_rows"]):
            raise RuntimeError("Wrong mixed native output shape/order")
        offset = 3 if inspect else 5
        values = [a.number(v) for v in data[offset:]]
        if len(values) != (n*n+n*k if inspect else n*k):
            raise RuntimeError("Wrong mixed native output length")
        if not all(a.ctx.isfinite(v) if a.ctx else np.isfinite(v) for v in values):
            raise FloatingPointError("Nonfinite mixed native output")
        def matrix(values, rows, cols):
            return a.ctx.matrix([values[i*cols:(i+1)*cols] for i in range(rows)]) if a.ctx else np.asarray(values).reshape(rows, cols)
        if inspect:
            unique_results.append((matrix(values[:n*n], n, n), matrix(values[n*n:], n, k)))
        else:
            unique_results.append(dict(weights=matrix(values, n, k),
                condition=None if data[3] == "none" else float(data[3]), residual=float(data[4])))
    # Keep separate result matrices for reused requests; caller mutation cannot
    # alter a different local row's diagnostic reconstruction.
    if inspect:
        return [unique_results[i] for i in mapping]
    return [dict(unique_results[i], weights=unique_results[i]["weights"].copy()) for i in mapping]


def execute(owner):
    """Execute mixed jobs with one factorization and all target RHS per job."""
    execution = validate_execution(owner.backend, getattr(owner, "execution", None))
    if owner.backend.shape_rule != "fixed" or owner.backend.local_solver != "lu":
        raise NotImplementedError("Mixed functional execution supports fixed parameters and LU")
    if isinstance(owner.backend, CppBackend):
        return _cpp(owner, owner.jobs)
    if execution.scheduler != "serial" or execution.reuse != "none":
        raise NotImplementedError("MPI/reuse currently requires CppBackend")
    if not isinstance(owner.backend, PythonBackend):
        raise NotImplementedError("Mixed approximation supports PythonBackend and CppBackend")
    if owner.backend.workers != 1:
        raise NotImplementedError("Mixed Python reference currently uses one worker; use C++/MPI for parallel weights")
    from .mixed_local import arrays
    a = owner.arithmetic
    results = []
    before = time.perf_counter()
    for job in owner.jobs:
        G, Q = arrays(owner, job)
        factor = a.factor(G, compute_condition=owner.backend.compute_condition)
        W = factor.solve(Q, transpose=True)
        error = G.T*W-Q if a.ctx else G.T@W-Q
        maximum = lambda x: max((abs(v) for v in (x if a.ctx else x.flat)), default=0)
        residual = float(maximum(error)/(maximum(Q) or 1))
        results.append(dict(weights=W, condition=factor.condition, residual=residual))
    owner.execution_info = dict(scheduler="serial", ranks=1, numerical_backend="PythonBackend",
        requested_jobs=len(owner.jobs), unique_jobs=len(owner.jobs), cache_hits=0,
        local_digits=owner.precision.local_digits, seconds=time.perf_counter()-before)
    return results


def reconstruct(owner, index):
    """Rebuild physical mixed Gram/RHS arrays using the selected backend."""
    if isinstance(owner.backend, CppBackend):
        return _cpp(owner, [owner.jobs[index]], inspect=True)[0]
    from .mixed_local import arrays
    return arrays(owner, owner.jobs[index])


def validate_execution(backend, execution=None):
    """Validate supported scheduling before constructing any local matrices."""
    execution = LocalExecution() if execution is None else execution
    if not isinstance(execution, LocalExecution):
        raise TypeError("execution must be LocalExecution")
    if isinstance(backend, CppBackend) and backend.executable is not None:
        raise ValueError("Mixed local backend uses its compiled cached executable")
    if isinstance(backend, PythonBackend) and backend.workers != 1:
        raise NotImplementedError("Mixed Python reference currently uses one worker; use C++/MPI for parallel weights")
    if not isinstance(backend, CppBackend) and (execution.scheduler != "serial" or execution.reuse != "none"):
        raise NotImplementedError("MPI/reuse currently requires CppBackend")
    return execution
