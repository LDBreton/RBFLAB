"""Reproducible API-level Python/C++ global-collocation scaling benchmark.

Run from a native-enabled source checkout:
    python -m examples.benchmark_global_cpp --output outputs/global_benchmark

Assembly includes eager LU on both implementations. Condition diagnostics are
disabled by a benchmark-only Arithmetic.factor override on the Python side;
CppGlobalBackend has a public switch. No production implementation is modified.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import platform
import statistics
import time
import warnings

import mpmath
import numpy as np
import scipy
import sympy as sp
from threadpoolctl import threadpool_info, threadpool_limits

import rbflab as r
from rbflab.nodal import Arithmetic


def processor_model():
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor()


@contextmanager
def without_condition_diagnostics():
    original = Arithmetic.factor

    def factor(self, matrix, *, compute_condition=True):
        return original(self, matrix, compute_condition=False)

    Arithmetic.factor = factor
    try:
        yield
    finally:
        Arithmetic.factor = original


def problem_data(query_count):
    model = r.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    problem = model.stationary(
        sp.Eq(-model.laplacian(u), 2*sp.pi**2*exact),
        boundary=[model.bc('boundary', sp.Eq(u, 0))],
    )
    queries = np.random.default_rng(20261012).uniform(.01, .99, (query_count, 2))
    ctx = mpmath.mp.clone()
    ctx.dps = 100
    truth = model.data(exact).mp_values(ctx, queries)
    return problem, queries, ctx, truth


def accuracy(values, ctx, truth):
    converted = [ctx.mpf(float(v)) if isinstance(v, (float, np.floating)) else ctx.mpf(v)
                 for v in values]
    error = [a-b for a, b in zip(converted, truth)]
    max_error = max(map(abs, error))
    relative_l2 = ctx.sqrt(ctx.fsum(v*v for v in error)/ctx.fsum(v*v for v in truth))
    return dict(max_abs_error=float(max_error), relative_l2_error=float(relative_l2),
                max_abs_error_decimal=ctx.nstr(max_error, 30)), converted


def measure(method, problem, cloud, queries, digits, repeats, reference_ctx, truth):
    samples = []
    last_values = None
    all_warnings = []
    for repeat in range(repeats):
        system = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                start = time.perf_counter()
                system = method.assemble(problem, cloud)
                assembled = time.perf_counter()
                solution = system.solve()
                solved = time.perf_counter()
                values = solution.evaluate_mp(queries) if digits else solution.evaluate(queries)
                evaluated = time.perf_counter()
            errors, last_values = accuracy(values, reference_ctx, truth)
            samples.append(dict(
                assembly_lu_s=assembled-start, solve_s=solved-assembled,
                evaluation_s=evaluated-solved, total_s=evaluated-start,
                relative_residual=float(solution.diagnostics['relative_residual']),
                **errors,
            ))
            all_warnings.extend(str(w.message) for w in caught)
        finally:
            if system is not None and hasattr(system, 'close'):
                system.close()
    result = {key: statistics.median(s[key] for s in samples)
              for key in samples[0] if key != 'max_abs_error_decimal'}
    result.update(samples=samples, warnings=sorted(set(all_warnings)))
    return result, last_values


def save(data, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def profile_native_setup(cache_dir, intervals=(4, 32), repeats=3):
    """Separate warm lookup, worker acquisition and native BUILD in double mode."""
    from unittest.mock import patch
    import rbflab.global_backend as native

    problem, _, _, _ = problem_data(1)
    original_compile = native._compiled
    original_acquire = native._acquire_worker
    original_request = native._Worker.request
    records = []
    with threadpool_limits(limits=1):
        for size in intervals:
            samples = []
            method = r.GlobalCollocation(r.PHS(5), polynomial_degree=2, backend=r.CppGlobalBackend(
                threads=1, compute_condition=False, cache_dir=cache_dir))
            with method.assemble(problem, r.geometry.unit_box_grid(2)):
                pass
            for _ in range(repeats):
                timings = {}

                def compile_timed(*args, **kwargs):
                    start = time.perf_counter()
                    result = original_compile(*args, **kwargs)
                    timings['cached_lookup_s'] = time.perf_counter()-start
                    return result

                def acquire_timed(*args, **kwargs):
                    start = time.perf_counter()
                    result = original_acquire(*args, **kwargs)
                    timings['worker_acquire_s'] = time.perf_counter()-start
                    return result

                def request_timed(self, command):
                    start = time.perf_counter()
                    result = original_request(self, command)
                    if command == 'BUILD':
                        timings['native_assembly_lu_s'] = time.perf_counter()-start
                    return result

                cloud = r.geometry.unit_box_grid(size)
                with patch.object(native, '_compiled', compile_timed), \
                     patch.object(native, '_acquire_worker', acquire_timed), \
                     patch.object(native._Worker, 'request', request_timed):
                    start = time.perf_counter()
                    with method.assemble(problem, cloud):
                        timings['assembly_lu_total_s'] = time.perf_counter()-start
                timings['remaining_frontend_s'] = timings['assembly_lu_total_s'] - sum(
                    timings[k] for k in ('cached_lookup_s', 'worker_acquire_s', 'native_assembly_lu_s'))
                samples.append(timings)
            records.append(dict(nodes=len(cloud.points), samples=samples,
                                **{k: statistics.median(v[k] for v in samples) for k in samples[0]}))
    return records


def plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    measures = [('assembly_lu_s', 'Assembly + LU'), ('solve_s', 'Solve + residual'),
                ('evaluation_s', f"Evaluation ({data['settings']['queries']} points)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    styles = {('python', 'symmetric'): ('#2266aa', '-'),
              ('cpp', 'symmetric'): ('#d66027', '-'),
              ('python', 'asymmetric'): ('#2266aa', '--'),
              ('cpp', 'asymmetric'): ('#d66027', '--')}
    for row, precision in enumerate(('double', 'mp')):
        for col, (metric, title) in enumerate(measures):
            ax = axes[row, col]
            for (backend, scheme), (color, linestyle) in styles.items():
                rows = sorted((v for v in data['results'] if v['precision'] == precision
                               and v['backend'] == backend and v['scheme'] == scheme),
                              key=lambda v: v['nodes'])
                if rows:
                    ax.plot([v['nodes'] for v in rows], [v[metric] for v in rows],
                            color=color, linestyle=linestyle, marker='o', markersize=4,
                            label=f'{backend.upper()} {scheme}')
            ax.set_yscale('log')
            ax.set_xlabel('Nodes (6 polynomial coefficients added)')
            ax.set_ylabel('Seconds, median')
            ax.grid(True, which='both', alpha=.2)
            arithmetic = 'Double' if precision == 'double' else f"{data['settings']['mp_digits']} digits"
            ax.set_title(f'{arithmetic} · {title}')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Global collocation: Python vs C++\n'
                 f"PHS r⁵ + degree-2 polynomials, Python BLAS 1 thread / C++ {data['settings']['threads']}, cached compilation",
                 fontsize=13)
    fig.savefig(output/'timings.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for ax, scheme in zip(axes, ('symmetric', 'asymmetric')):
        for precision, color in [('double', '#2266aa'), ('mp', '#d66027')]:
            for backend, linestyle in [('python', '-'), ('cpp', '--')]:
                rows = sorted((v for v in data['results'] if v['precision'] == precision
                               and v['backend'] == backend and v['scheme'] == scheme),
                              key=lambda v: v['nodes'])
                if rows:
                    label = 'Double' if precision == 'double' else f"{data['settings']['mp_digits']} digits"
                    ax.plot([v['nodes'] for v in rows], [v['max_abs_error'] for v in rows],
                            color=color, linestyle=linestyle, marker='o', markersize=4,
                            label=f'{backend.upper()} {label}')
        ax.set_yscale('log')
        ax.set_xlabel('Nodes')
        ax.set_ylabel('Maximum sampled absolute solution error')
        ax.set_title(scheme.capitalize())
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Accuracy against sin(πx) sin(πy) at fixed off-node points')
    fig.savefig(output/'accuracy.png', dpi=180)
    plt.close(fig)


def write_report(data, output, results_file='results.json'):
    records = {(v['nodes'], v['precision'], v['scheme'], v['backend']): v for v in data['results']}
    rows = []
    for item in sorted(data['comparisons'], key=lambda v: (v['precision'], v['scheme'], v['nodes'])):
        n, precision, scheme = item['nodes'], item['precision'], item['scheme']
        py = records[(n, precision, scheme, 'python')]
        cpp = records[(n, precision, scheme, 'cpp')]
        label = 'double' if precision == 'double' else f"{data['settings']['mp_digits']} digits"
        cells = [str(n), label, scheme]
        for key in ('assembly_lu_s', 'solve_s', 'evaluation_s'):
            cells.extend([f'{py[key]:.6f}', f'{cpp[key]:.6f}'])
        cells += [f"{item['speedups']['total_s']:.2f}×", f"{py['max_abs_error']:.3e}",
                  f"{cpp['max_abs_error']:.3e}", f"{item['backend_max_gap']:.3e}"]
        rows.append('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in cells) + '</tr>')
    env = data['environment']
    largest_double = max((v['nodes'] for v in data['results'] if v['precision'] == 'double'), default=0)
    largest_mp = max((v['nodes'] for v in data['results'] if v['precision'] == 'mp'), default=0)
    prep = ', '.join(f"{v['precision']}/{v['scheme']}: {v['seconds']:.2f} s" for v in data['preparation'])
    profile = data.get('setup_profile', [])
    profile_html = ''
    if profile:
        profile_html = '<h2>Native setup profile (double, symmetric)</h2><table><tr><th>Nodes</th><th>Cached lookup (s)</th><th>Worker acquisition (s)</th><th>Native assembly + LU (s)</th><th>Full assembly call (s)</th></tr>'
        for v in profile:
            acquire = v.get('worker_acquire_s', v.get('worker_startup_s', 0))
            profile_html += f"<tr><td>{v['nodes']}</td><td>{v['cached_lookup_s']:.6f}</td><td>{acquire:.6f}</td><td>{v['native_assembly_lu_s']:.6f}</td><td>{v['assembly_lu_total_s']:.6f}</td></tr>"
        profile_html += '</table>'
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RBFLAB global collocation benchmark</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#172536;background:#f6f8fb;max-width:1380px;margin:auto;padding:32px}}
h1,h2{{line-height:1.2}}h1{{font-size:32px}}h2{{margin-top:36px}}p{{max-width:1000px}}
.panel{{background:white;padding:20px;border:1px solid #dce3eb;border-radius:8px;margin:18px 0}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:12px;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:8px;border-bottom:1px solid #dde4ed;text-align:right;white-space:nowrap}}th{{background:#e9eef6;position:sticky;top:0}}
tr:nth-child(even){{background:#f3f6fa}}.scroll{{overflow-x:auto}}a{{color:#135c9e}}code{{background:#e9eef6;padding:2px 5px}}</style>
<h1>RBFLAB global collocation benchmark</h1>
<p>Python and C++ on identical 2D Poisson problems: <code>u = sin(πx)sin(πy)</code> on the unit square,
zero Dirichlet boundaries, <code>PHS(5)</code>, degree-2 polynomial augmentation. Each matrix has N + 6 unknowns.</p>
<div class="panel"><strong>Measurement setup</strong><ul>
<li>{html.escape(env['processor'])}; {html.escape(env['platform'])}; Python {env['python']}.</li>
<li>NumPy {env['numpy']}, SciPy {env['scipy']}, mpmath {env['mpmath']} ({env['mpmath_backend']} integer backend).
C++ MPFR and Eigen through WSL; {html.escape(env['compiler'])}.</li>
<li>Python BLAS limited to one thread; native OpenMP threads: {data['settings']['threads']}.
Median of {data['settings']['repeats']} independent system constructions; all raw samples retained.</li>
<li>Assembly includes eager LU. Solve includes residual diagnostics and Python matrix fingerprinting.
Evaluation uses {data['settings']['queries']} fixed random off-node points; MP values are evaluated without rounding to double.</li>
<li>Condition-number computation disabled on both sides. Python uses a benchmark-only <code>Arithmetic.factor</code>
override because its global frontend has no public switch.</li>
<li>Compilation and small-system warmup excluded. Cache validation, worker acquisition, serialization and IPC remain included.
The optimized backend reuses idle workers after close; these measurements represent warm repeated use, not cold process startup.</li>
<li>Largest tested clouds: double {largest_double} nodes; {data['settings']['mp_digits']}-digit arithmetic {largest_mp} nodes. Geometry is Float64 in both modes.</li>
</ul><p>Initial preparation: {html.escape(prep)}.</p></div>
<div class="panel"><img src="timings.png" alt="Assembly, solve and evaluation scaling by precision and scheme"></div>
<div class="panel"><img src="accuracy.png" alt="Maximum sampled solution error versus node count"></div>
<h2>All measurements</h2><p>Times in seconds. Total speedup is Python / C++; greater than one favors C++.
Maximum errors are sampled, not certified domain-wide bounds. The backend gap is the largest difference between their evaluated values.</p>
<div class="panel scroll"><table><thead><tr><th>Nodes</th><th>Precision</th><th>Scheme</th>
<th>Assembly + LU: Py</th><th>C++</th><th>Solve: Py</th><th>C++</th><th>Evaluate: Py</th><th>C++</th>
<th>Total speedup</th><th>Max error: Py</th><th>C++</th><th>Backend gap</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{profile_html}
<h2>Interpretation limits</h2><p>This is a single-host benchmark of one PHS discretization, not a claim about all kernels,
geometries or thread counts. Python extended arithmetic uses the installed pure-Python mpmath backend; adding GMP acceleration
would change that baseline. Nearly overlapping double/MP error curves indicate discretization error dominates this problem;
the benchmark does not test the accuracy benefit of multiprecision for very flat smooth kernels.</p>
<p><a href="{html.escape(results_file)}">Raw results and methodology (JSON)</a></p></html>'''
    (output/'report.html').write_text(body, encoding='utf-8')


def compare_baseline(data, baseline_path, output):
    """Compare identical cases to a saved run without overwriting its evidence."""
    import matplotlib.pyplot as plt

    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    for group, fields in [('settings', ('queries', 'threads', 'mp_digits')),
                          ('methodology', ('pde', 'kernel', 'polynomial_degree', 'dimension'))]:
        for field in fields:
            if baseline[group][field] != data[group][field]:
                raise ValueError(f'Baseline differs in {group}.{field}; timings are not comparable')
    key = lambda v: (v['precision'], v['scheme'], v['nodes'], v['backend'])
    previous = {key(v): v for v in baseline['results']}
    comparison = []
    for current in data['results']:
        old = previous.get(key(current))
        if old is None or current['backend'] != 'cpp':
            continue
        comparison.append(dict(precision=current['precision'], scheme=current['scheme'],
            nodes=current['nodes'], before=old, after=current,
            speedups={k: old[k]/current[k] for k in
                      ('assembly_lu_s', 'solve_s', 'evaluation_s', 'total_s')}))
    save(dict(baseline=str(Path(baseline_path).resolve()), comparisons=comparison),
         output/'before_after.json')
    measures = [('assembly_lu_s', 'Assembly + LU'), ('evaluation_s', 'Evaluation')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, (metric, label) in zip(axes, measures):
        for scheme, style in [('symmetric', '-'), ('asymmetric', '--')]:
            rows = sorted((v for v in comparison if v['precision'] == 'double'
                           and v['scheme'] == scheme), key=lambda v: v['nodes'])
            for revision, color in [('before', '#777777'), ('after', '#d66027')]:
                ax.plot([v['nodes'] for v in rows], [v[revision][metric] for v in rows],
                        color=color, linestyle=style, marker='o', markersize=4,
                        label=f'{revision.capitalize()} {scheme}')
        ax.set(xlabel='Nodes', ylabel='Seconds, median', title=label, yscale='log')
        ax.grid(True, which='both', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('C++ double: original versus optimized backend (one thread, warm use)')
    fig.savefig(output/'before_after.png', dpi=180)
    plt.close(fig)
    rows = []
    for v in comparison:
        cells = [v['precision'], v['scheme'], str(v['nodes'])]
        for metric in ('assembly_lu_s', 'solve_s', 'evaluation_s'):
            cells += [f"{v['before'][metric]:.6f}", f"{v['after'][metric]:.6f}",
                      f"{v['speedups'][metric]:.2f}×"]
        cells += [f"{v['before']['max_abs_error']:.6e}", f"{v['after']['max_abs_error']:.6e}"]
        rows.append('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in cells)+'</tr>')
    path = output/'report.html'
    report = path.read_text(encoding='utf-8')
    section = '''<h2>Original versus optimized C++</h2>
<p>Historical baseline versus this run, same problem and node counts. Times in seconds;
speedup is original / optimized. Warm repeated use includes worker acquisition;
the original backend launched a worker for every system.</p>
<div class="panel"><img src="before_after.png" alt="Before and after double timings"></div>
<div class="panel scroll"><table><tr><th>Precision</th><th>Scheme</th><th>Nodes</th>
<th>Assembly before</th><th>After</th><th>Speedup</th><th>Solve before</th><th>After</th><th>Speedup</th>
<th>Evaluation before</th><th>After</th><th>Speedup</th><th>Error before</th><th>Error after</th></tr>'''
    section += ''.join(rows)+'</table></div><p><a href="before_after.json">Raw before/after comparisons</a></p>'
    path.write_text(report.replace('</html>', section+'</html>'), encoding='utf-8')


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    problem, queries, ctx, truth = problem_data(args.queries)
    data = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        settings=vars(args),
        environment=dict(platform=platform.platform(), processor=processor_model(),
                         logical_cpus=os.cpu_count(), python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__, mpmath=mpmath.__version__,
                         mpmath_backend=mpmath.libmp.BACKEND),
        methodology=dict(
            pde='-Laplacian(u)=2*pi^2*sin(pi*x)*sin(pi*y), unit square, zero Dirichlet',
            kernel='PHS(5)', polynomial_degree=2, dimension=2,
            assembly='Public assemble() includes matrix construction AND eager LU on both backends',
            solve='Public solve() includes reused-factor solve and residual; Python also fingerprints the matrix',
            evaluation='Public evaluate() for double, evaluate_mp() for extended precision',
            diagnostics='Condition estimation disabled on both; benchmark-only Arithmetic.factor override for Python',
            native_overhead='Warm cache lookup, worker acquisition and IPC INCLUDED; default bounded worker reuse enabled',
            precision='Python mpmath versus C++ MPFR; geometry Float64 on both',
            repetitions='Median of independent system constructions; one untimed small-system warmup per configuration',
        ),
        preparation=[], results=[], comparisons=[], failures=[],
    )
    path = output/'results.json'
    with threadpool_limits(limits=1), without_condition_diagnostics():
        data['environment']['threadpools'] = threadpool_info()
        from rbflab.kernel_compiler import toolchain
        data['environment']['compiler'] = toolchain()[1]
        configurations = {}
        for precision in args.precisions:
            digits = args.mp_digits if precision == 'mp' else None
            for scheme in args.schemes:
                common = dict(kernel=r.PHS(5), scheme=scheme, polynomial_degree=2,
                              precision=r.Precision(global_digits=digits))
                python = r.GlobalCollocation(**common)
                native = r.GlobalCollocation(**common, backend=r.CppGlobalBackend(
                    threads=args.threads, compute_condition=False,
                    cache_dir=args.cache_dir or output/'native_cache', timeout=600))
                start = time.perf_counter()
                native = native.prepare(problem, dimension=2)
                preparation = time.perf_counter()-start
                data['preparation'].append(dict(precision=precision, scheme=scheme,
                                                seconds=preparation, artifact=str(native.compiled_artifact)))
                for backend, method in [('python', python), ('cpp', native)]:
                    measure(method, problem, r.geometry.unit_box_grid(2), queries[:4], digits, 1,
                            ctx, truth[:4])
                    configurations[(precision, scheme, backend)] = method
                print(f'Prepared {precision} {scheme}: {preparation:.3f}s', flush=True)
                save(data, path)

        for intervals in args.intervals:
            cloud = r.geometry.unit_box_grid(intervals)
            for precision in args.precisions:
                digits = args.mp_digits if precision == 'mp' else None
                for scheme in args.schemes:
                    pair = {}
                    # Alternate order across sizes to reduce systematic ordering bias.
                    order = ['python', 'cpp'] if args.intervals.index(intervals) % 2 == 0 else ['cpp', 'python']
                    for backend in order:
                        metadata = dict(nodes=len(cloud.points), unknowns=len(cloud.points)+6,
                                        intervals=intervals, precision=precision, scheme=scheme,
                                        backend=backend)
                        print(f"Starting N={len(cloud.points)} {precision} {scheme} {backend}", flush=True)
                        try:
                            result, values = measure(configurations[(precision, scheme, backend)],
                                problem, cloud, queries, digits, args.repeats, ctx, truth)
                            record = dict(**metadata, **result)
                            data['results'].append(record)
                            pair[backend] = (record, values)
                            print(f"  assembly+LU={result['assembly_lu_s']:.6f}s solve={result['solve_s']:.6f}s "
                                  f"eval={result['evaluation_s']:.6f}s error={result['max_abs_error']:.3e}", flush=True)
                        except Exception as exc:
                            data['failures'].append(dict(**metadata, error=repr(exc)))
                            print(f'  FAILED: {exc}', flush=True)
                        save(data, path)
                    if len(pair) == 2:
                        a, av = pair['python']
                        b, bv = pair['cpp']
                        gap = max(abs(x-y) for x, y in zip(av, bv))
                        data['comparisons'].append(dict(nodes=len(cloud.points), precision=precision, scheme=scheme,
                            speedups={name: a[name]/b[name] for name in ('assembly_lu_s', 'solve_s', 'evaluation_s', 'total_s')},
                            backend_max_gap=float(gap), backend_max_gap_decimal=ctx.nstr(gap, 30)))
                        save(data, path)
    data['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(data, path)
    plot(data, output)
    write_report(data, output)
    if args.baseline:
        compare_baseline(data, args.baseline, output)
    print(f'Results: {path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/global_benchmark')
    parser.add_argument('--cache-dir', default=None, help='Reuse a prepared cache across benchmark runs')
    parser.add_argument('--baseline', default=None, help='Saved results JSON for before/after comparisons')
    parser.add_argument('--intervals', type=int, nargs='+', default=[4, 6, 8, 10, 12])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--queries', type=int, default=128)
    parser.add_argument('--mp-digits', type=int, default=60)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--schemes', nargs='+', choices=['symmetric', 'asymmetric'], default=['symmetric', 'asymmetric'])
    parser.add_argument('--precisions', nargs='+', choices=['double', 'mp'], default=['double', 'mp'])
    arguments = parser.parse_args()
    if min(arguments.intervals) < 2 or arguments.repeats < 1 or arguments.queries < 1:
        parser.error('Use intervals >=2, repeats >=1 and queries >=1')
    run(arguments)
