"""Matched native Windows/WSL protocol benchmark; run from the checkout root.

Windows Python generates each problem once. The same generated source, Eigen
headers, serialized geometry, operators and RHS are used by both toolchains.
Only process launch and platform arithmetic libraries differ. Each EVAL receives
the coefficients produced by that platform's SOLVE, not the captured coefficients.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import mpmath
import numpy as np
import sympy as sp
import rbflab as r
import rbflab.global_backend as gb
from rbflab.eigen_build import eigen_dependency
from rbflab.legacy_cpp import _linux_path
from rbflab.native_toolchain import compiler, compiler_environment, compiler_identity, arithmetic_flags
from examples.benchmark_global_cpp import processor_model

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def capture(kind, digits, cells, threads, output):
    workers = []
    original = gb._Worker

    class Recorder(original):
        def __init__(self, executable, backend):
            self.commands = []
            self.executable = Path(executable)
            workers.append(self)
            super().__init__(executable, backend)

        def request(self, command):
            reply = super().request(command)
            self.commands.append(command)
            return reply

    gb._Worker = Recorder
    try:
        cloud = r.geometry.unit_box_grid(cells, 2)
        queries = np.random.default_rng(20261022).uniform(.05, .95, (256, 2))
        backend = r.CppGlobalBackend(threads=threads, compute_condition=False,
                                    reuse_workers=False, cache_dir=output/'cache', timeout=300)
        if kind == 'poisson':
            model = r.SymbolicScalar(2)
            u = model.field
            x, y = model.coordinates
            exact = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
            problem = model.stationary(sp.Eq(-model.laplacian(u), 2*sp.pi**2*exact),
                boundary=[model.bc('boundary', sp.Eq(u, 0))])
            method = r.GlobalCollocation(r.PHS(7), polynomial_degree=3, backend=backend,
                                        precision=r.Precision(global_digits=digits))
        else:
            from examples.benchmark_global_cpp_fields import problem_data
            model, problem, *_ = problem_data(2, 1)
            u, p = model.fields
            method = r.GlobalCollocation(spaces={
                u: r.DivergenceFreeSpace(r.PHS(7), polynomial_degree=2),
                p: r.PressureSpace(r.PHS(3), polynomial_degree=1)},
                backend=backend, precision=r.Precision(global_digits=digits))
        start = time.perf_counter()
        with method.assemble(problem, cloud) as system:
            prepared = time.perf_counter()-start
            size = system.size
            solution = system.solve()
            if kind == 'poisson':
                (solution.evaluate_mp if digits else solution.evaluate)(queries)
            else:
                solution.velocity(queries, extended=bool(digits))
                solution.pressure_gradient(queries, extended=bool(digits))
                solution.pressure(queries, extended=bool(digits))
                solution.divergence(queries, extended=bool(digits))
        assert len(workers) == 1
        worker = workers[0]
        return worker.commands, worker.executable.parent, dict(
            nodes=len(cloud.points), unknowns=size, queries=len(queries),
            capture_assembly_including_preparation_s=prepared)
    finally:
        gb._Worker = original


def compile_pair(directory, arithmetic, output):
    target = output/'binaries'/arithmetic/directory.name
    target.mkdir(parents=True, exist_ok=True)
    source = (directory/'kernel.cpp').read_text()
    # Benchmark-only timing surrounds each native protocol command, including
    # number parsing and output formatting, but excluding pipe IO/process launch.
    source = source.replace('try{std::ostringstream out;',
        'try{auto bench_start=std::chrono::steady_clock::now();std::ostringstream out;')
    marker = 'std::cout<<out.str()<<std::endl;'
    assert source.count(marker) == 1
    source = source.replace(marker,
        'std::cerr<<"BENCH "<<line.substr(0,line.find(\' \'))<<" "<<std::setprecision(17)'
        '<<std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_start).count()<<std::endl;\n'+marker)
    (target/'kernel.cpp').write_text(source)
    for header in directory.glob('*.hpp'):
        (target/header.name).write_bytes(header.read_bytes())
    evidence = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    eigen_sha256=eigen_dependency()[1], compile_s={})
    executables = {}
    for host in ('windows', 'wsl'):
        executable = target/('worker.exe' if host == 'windows' else 'worker-linux')
        start = time.perf_counter()
        if host == 'windows':
            cmd = [compiler(), '-O3', '-std=c++17', '-fopenmp',
                   '-I'+str(eigen_dependency()[0]), str(target/'kernel.cpp'),
                   *arithmetic_flags(arithmetic), '-o', str(executable)]
            env = compiler_environment()
        else:
            include = ROOT/'cpp/deps/usr/include'
            flags = ['-DRBFLAB_DOUBLE'] if arithmetic == 'float64' else [
                '-I'+_linux_path(include), '-I'+_linux_path(include/'x86_64-linux-gnu'),
                '-Wl,-l:libmpfr.so.6', '-Wl,-l:libgmp.so.10']
            cmd = ['wsl.exe', '-d', 'Ubuntu', '--', 'g++', '-O3', '-std=c++17', '-fopenmp',
                   '-I'+_linux_path(eigen_dependency()[0]), _linux_path(target/'kernel.cpp'),
                   *flags, '-o', _linux_path(executable)]
            env = os.environ.copy()
        run = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        (target/(host+'-compile.log')).write_text(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(run.stderr)
        evidence['compile_s'][host] = time.perf_counter()-start
        executables[host] = executable
    save(target/'build.json', evidence)
    return executables, evidence


def replay(host, executable, commands):
    launch = ([str(executable)] if host == 'windows' else
              ['wsl.exe', '-d', 'Ubuntu', '--', _linux_path(executable)])
    start = time.perf_counter()
    process = subprocess.Popen(launch, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1)
    # WSL getline sees CR literally; send identical LF-only protocol bytes.
    process.stdin.reconfigure(newline='\n')
    responses = []
    coefficients = None
    startup = None
    for command in commands:
        kind = command.split()[0]
        if kind == 'EVAL':
            assert coefficients is not None
            tokens = command.split()
            tokens[-len(coefficients):] = coefficients
            command = ' '.join(tokens)
        process.stdin.write(command+'\n')
        process.stdin.flush()
        response = process.stdout.readline().strip()
        if startup is None:
            startup = time.perf_counter()-start
        if not response.startswith('OK'):
            process.kill()
            raise RuntimeError(host+' '+kind+': '+response+' '+process.stderr.read())
        responses.append(dict(command=kind, response=response))
        if kind == 'SOLVE':
            tokens = response.split()
            assert tokens[1] == '1'
            coefficients = tokens[2:-1]
    process.stdin.close()
    errors = process.stderr.read()
    process.wait(timeout=300)
    total = time.perf_counter()-start
    if process.returncode:
        raise RuntimeError(errors)
    phases = defaultdict(float)
    for line in errors.splitlines():
        if line.startswith('BENCH '):
            _, kind, seconds = line.split()
            phases[kind] += float(seconds)
    return dict(total_s=total, startup_s=startup, build_s=phases['BUILD'],
                solve_s=phases['SOLVE'], evaluation_s=phases['EVAL'],
                native_total_s=sum(phases.values()), phases=dict(phases)), responses


def compare(left, right, digits):
    ctx = mpmath.mp.clone(); ctx.dps = 100
    grouped = defaultdict(list)
    assert len(left) == len(right)
    evaluation = 0
    for a, b in zip(left, right):
        assert a['command'] == b['command']
        at, bt = a['response'].split(), b['response'].split()
        assert len(at) == len(bt)
        name = a['command']
        if name == 'SOLVE':
            # Count is protocol metadata; residual is checked separately from
            # the possibly much larger coefficient scale.
            grouped['SOLVE_residual'].append((ctx.mpf(at[-1]),ctx.mpf(bt[-1])))
            at, bt = at[2:-1], bt[2:-1]
            name = 'SOLVE_coefficients'
        elif name == 'EVAL':
            assert at[1] == bt[1]
            at, bt = at[2:], bt[2:]
            name = 'EVAL_' + str(evaluation)
            evaluation += 1
        else:
            at, bt = at[1:], bt[1:]
        for av, bv in zip(at, bt):
            try:
                av, bv = ctx.mpf(av), ctx.mpf(bv)
            except ValueError:
                assert av == bv
                continue
            assert ctx.isfinite(av) and ctx.isfinite(bv)
            grouped[name].append((av, bv))
    result = {}
    for name, values in grouped.items():
        gap = max(abs(a-b) for a,b in values)
        scale = max(ctx.mpf(1), max(abs(a) for a,b in values), max(abs(b) for a,b in values))
        result[name] = dict(max_abs=ctx.nstr(gap,30), scaled_max=ctx.nstr(gap/scale,30))
    # A scale-aware tolerance allows rounding amplified by the dense solve.
    tolerance = ctx.mpf('1e-48') if digits else ctx.mpf('1e-8')
    passed = all(ctx.mpf(v['scaled_max']) < tolerance for v in result.values())
    return dict(passed=passed,
        fields_passed=all(ctx.mpf(v['scaled_max']) < tolerance for k,v in result.items() if k.startswith('EVAL')),
        coefficients_passed=ctx.mpf(result['SOLVE_coefficients']['scaled_max']) < tolerance,
        residuals_passed=ctx.mpf(result['SOLVE_residual']['scaled_max']) < tolerance,
        scaled_tolerance=str(tolerance), commands=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/windows-wsl-native')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cells', type=int, nargs='+', default=[6,10])
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    data = dict(started=datetime.now(timezone.utc).isoformat(),
        environment=dict(cpu=processor_model(), windows=platform.platform(), python=sys.version,
            windows_compiler=compiler_identity(), wsl_compiler=subprocess.check_output(
                ['wsl.exe','-d','Ubuntu','--','g++','--version'],text=True).splitlines()[0]),
        methodology='Identical generated C++ and Eigen headers; PHS scalar Poisson and solenoidal Stokes; '
        '256 deterministic query points; fresh workers; one untimed warmup per host; five alternating '
        'host samples by default; 60 decimal digit MPFR or Float64; condition estimation off. '
        'Native timings include protocol parsing/formatting but exclude pipe I/O and startup. '
        'Total includes process/WSL launch and Python protocol I/O, excludes compilation and symbolic setup. '
        'All EVAL commands use the coefficients computed by the host being tested. '
        'Geometry remains Float64. Windows GCC/static runtime versus WSL GCC/dynamic runtime; '
        'these are installed-toolchain comparisons, not isolated OS effects.',
        builds=[], results=[])
    binaries = {}
    for kind in ('poisson','stokes'):
        for digits in (None,60):
            arithmetic = 'mpfr' if digits else 'float64'
            for cells in args.cells:
                for threads in (1,4):
                    label = f'{kind}-{arithmetic}-n{cells}-t{threads}'
                    print('Preparing', label, flush=True)
                    commands, directory, meta = capture(kind,digits,cells,threads,output)
                    save(output/(label+'-input.json'), commands)
                    key = (kind,arithmetic)
                    if key not in binaries:
                        binaries[key], evidence = compile_pair(directory,arithmetic,output)
                        data['builds'].append(dict(kind=kind,arithmetic=arithmetic,**evidence))
                    samples = {'windows':[], 'wsl':[]}
                    outputs = {}
                    for host in samples:
                        replay(host,binaries[key][host],commands)  # warmup excluded
                    for repeat in range(args.repeats):
                        order = ('windows','wsl') if repeat%2 == 0 else ('wsl','windows')
                        for host in order:
                            metrics, responses = replay(host,binaries[key][host],commands)
                            samples[host].append(metrics)
                            outputs[host] = responses
                    save(output/(label+'-outputs.json'),outputs)
                    medians = {host:{name:statistics.median(s[name] for s in values)
                        for name in ('total_s','startup_s','build_s','solve_s','evaluation_s','native_total_s')}
                        for host,values in samples.items()}
                    comparison = compare(outputs['windows'],outputs['wsl'],digits)
                    record = dict(label=label,kind=kind,arithmetic=arithmetic,digits=digits,
                        cells=cells,threads=threads,**meta,samples=samples,medians=medians,
                        windows_speedup={k:medians['wsl'][k]/medians['windows'][k] for k in medians['windows']},
                        agreement=comparison)
                    data['results'].append(record)
                    save(output/'results.json',data)
                    print(label, 'PASS' if comparison['passed'] else 'FAIL',
                          'native Windows/WSL:',medians['windows']['native_total_s'],
                          medians['wsl']['native_total_s'],flush=True)
    data['finished'] = datetime.now(timezone.utc).isoformat()
    data['all_agree'] = all(row['agreement']['passed'] for row in data['results'])
    save(output/'results.json',data)


if __name__ == '__main__':
    main()
