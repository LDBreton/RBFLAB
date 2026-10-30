"""Stationary divergence-free global backend scaling benchmark.

Run: python -m examples.benchmark_global_cpp_fields --output outputs/global_fields
Assembly includes LU; evaluation includes velocity and pressure gradient.
Compilation and warmup are excluded. Results are saved after every measurement.
"""
import argparse
from datetime import datetime, timezone
import html
import platform
import statistics
import time
from pathlib import Path

import mpmath
import numpy as np
import scipy
import sympy as sp
from threadpoolctl import threadpool_info, threadpool_limits
import rbflab as r
from examples.benchmark_global_cpp import processor_model, save, without_condition_diagnostics


def problem_data(d, count):
    model = r.SymbolicStokes(d)
    U, p = model.fields
    x = model.coordinates
    psi = sp.prod(sp.sin(sp.pi*t) for t in x)/sp.pi
    if d == 2:
        velocity = sp.ImmutableMatrix([sp.diff(psi, x[1]), -sp.diff(psi, x[0])])
    else:
        velocity = sp.ImmutableMatrix([3*sp.diff(psi,x[1])-2*sp.diff(psi,x[2]),
                                      sp.diff(psi,x[2])-3*sp.diff(psi,x[0]),
                                      2*sp.diff(psi,x[0])-sp.diff(psi,x[1])])
    pressure = sp.prod(sp.cos(sp.pi*t) for t in x)
    lhs = -model.laplacian(U)+model.gradient(p)
    forcing = lhs.subs(dict(zip(U, velocity)) | {p: pressure}).doit()
    problem = model.stationary([sp.Eq(lhs, forcing)],
        boundary=[model.bc('boundary', sp.Eq(U, velocity))])
    queries = np.random.default_rng(20261012+d).uniform(.05,.95,(count,d))
    ctx = mpmath.mp.clone(); ctx.dps = 100
    expressions = dict(velocity=list(velocity), gradient=[sp.diff(pressure,t) for t in x], pressure=[pressure])
    truth = {name: [[v for v in model.data(expr).mp_values(ctx, queries)] for expr in exprs]
             for name,exprs in expressions.items()}
    return model, problem, queries, ctx, truth


def measure(method, problem, cloud, queries, digits, repeats, ctx, truth, backend):
    samples=[]; last=None
    for _ in range(repeats):
        system=None
        try:
            t=time.perf_counter(); system=method.assemble(problem,cloud); built=time.perf_counter()
            if backend == 'python':
                system.factor=system.arithmetic.factor(system.matrix)
            factored=time.perf_counter(); solution=system.solve(); solved=time.perf_counter()
            velocity=solution.velocity(queries,extended=bool(digits)); vt=time.perf_counter()
            gradient=solution.pressure_gradient(queries,extended=bool(digits)); gt=time.perf_counter()
            pressure=solution.pressure(queries, reference=(np.zeros(problem.dimension),1),extended=bool(digits))
            divergence=solution.divergence(queries,extended=bool(digits))
            errors={}; last={}
            for name, values in [('velocity',velocity),('gradient',gradient),('pressure',pressure)]:
                converted=[]; expected=[]
                for i in range(len(queries)):
                    for j in range(len(truth[name])):
                        v=values[i] if name=='pressure' and not digits else values[i,j]
                        converted.append(ctx.mpf(float(v)) if isinstance(v,(float,np.floating)) else ctx.mpf(v))
                        expected.append(truth[name][j][i])
                errors[name+'_max_error']=float(max(abs(a-b) for a,b in zip(converted,expected)))
                last[name]=converted
            errors['divergence_max']=float(max(abs(v) for v in divergence))
            samples.append(dict(assembly_lu_s=factored-t, public_assembly_s=built-t,
                python_factor_s=factored-built if backend=='python' else None,
                solve_s=solved-factored, velocity_s=vt-solved, gradient_s=gt-vt,
                evaluation_s=gt-solved,total_s=gt-t,unknowns=system.size,
                relative_residual=float(solution.diagnostics['relative_residual']),**errors))
            if repeats > 1:
                print(f'  sample {_+1}/{repeats}: assembly+LU {factored-t:.3f}s, '
                      f'solve {solved-factored:.3f}s, evaluation {gt-solved:.3f}s', flush=True)
        finally:
            if system is not None and hasattr(system,'close'): system.close()
    result={k: statistics.median(s[k] for s in samples) for k in samples[0] if samples[0][k] is not None}
    return dict(**result,samples=samples),last


def report(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=sorted(set((v['dimension'],v['precision']) for v in data['results']))
    for kind,metrics in [('timings',[('assembly_lu_s','Assembly + LU (s)'),('solve_s','Reused solve + residual (s)'),('evaluation_s','Velocity + pressure gradient (s)')]),
                         ('accuracy',[('velocity_max_error','Velocity max error'),('gradient_max_error','Pressure gradient max error'),('pressure_max_error','Gauged pressure max error')])]:
        fig,axes=plt.subplots(len(groups),3,figsize=(13,3*len(groups)),squeeze=False,constrained_layout=True)
        for row,(d,precision) in enumerate(groups):
            for col,(key,label) in enumerate(metrics):
                ax=axes[row,col]
                for backend in ['python','cpp']:
                    records=sorted((v for v in data['results'] if (v['dimension'],v['precision'],v['backend'])==(d,precision,backend)),key=lambda v:v['nodes'])
                    ax.plot([v['nodes'] for v in records],[v[key] for v in records], 'o-',label=backend)
                ax.set(xlabel='Nodes',ylabel=label,yscale='log',title=f'{d}D {precision}')
                if kind == 'accuracy' and col > 0:
                    # A zero-based scale makes weak pressure convergence visible.
                    ax.set(yscale='linear',ylim=(0,3.5 if col == 1 else 2.1))
                ax.grid(alpha=.25);ax.legend()
        fig.savefig(output/(kind+'.png'),dpi=160);plt.close(fig)
    keys=['dimension','precision','nodes','backend','unknowns','assembly_lu_s','solve_s','evaluation_s','velocity_max_error','gradient_max_error','pressure_max_error','divergence_max','relative_residual']
    rows=''.join('<tr>'+''.join('<td>'+html.escape(f'{v[k]:.5g}' if isinstance(v[k],float) else str(v[k]))+'</td>' for k in keys)+'</tr>' for v in data['results'])
    comparisons=''
    for v in data['comparisons']:
        fields=[v['dimension'],v['precision'],v['nodes']]
        fields += [f'{x:.3g}' for x in v['speedups'].values()]
        fields += [f'{x:.3e}' for x in v['backend_gaps'].values()]
        comparisons += '<tr>'+''.join('<td>'+str(x)+'</td>' for x in fields)+'</tr>'
    headings=['Dimension','Precision','Nodes','Assembly + LU speedup','Solve speedup','Evaluation speedup','Total speedup','Velocity gap','Gradient gap','Pressure gap']
    comparison_table='<h2>Python / C++ speedup and maximum backend differences</h2><p>A speedup below 1 means Python was faster. Total excludes the untimed pressure and divergence accuracy checks.</p><table><tr>'+''.join('<th>'+k+'</th>' for k in headings)+'</tr>'+comparisons+'</table>'
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Global divergence-free benchmark</title><style>body{font:16px system-ui;margin:32px}img{max-width:100%}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}table{font-size:12px}</style><h1>Global divergence-free C++ versus Python</h1><p>'+html.escape(data['methodology'])+'</p><p>'+html.escape(str(data['environment']))+'</p><p>Backend agreement and PDE accuracy are separate checks. In this run pressure-gradient convergence is weak at the tested resolutions; extra arithmetic digits do not remove the spatial approximation error. Results apply to these kernels and this manufactured problem.</p><img src="timings.png"><img src="accuracy.png">'+comparison_table+'<h2>Median times in seconds and maximum absolute errors</h2><table><tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'+rows+'</table><p><a href="results.json">Raw samples and backend gaps</a></p>',encoding='utf-8')


def run(args):
    if args.repeats < 1 or args.queries < 1 or (args.cells and min(args.cells) < 2):
        raise ValueError('Use positive repeats/queries and at least two cells per axis')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    data=dict(started=datetime.now(timezone.utc).isoformat(),settings=vars(args),
        environment=dict(cpu=processor_model(),platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mpmath.__version__,mpmath_backend=mpmath.libmp.BACKEND),
        methodology='Symmetric stationary Stokes, viscosity 1, unit box. Nonpolynomial analytic curl velocity and cosine-product pressure; full Dirichlet velocity. Velocity PHS(7), solenoidal degree 2; pressure PHS(3), degree 1. One thread. Median independent builds. Assembly includes LU on both; Python additionally builds its transient mass matrix. Solve reuses LU and includes residual and API checks. Evaluation times velocity plus pressure gradient at fixed random points. Pressure gauged to p(0)=1 and divergence checked outside timing. Condition estimates disabled. Compilation and small-system warmup excluded; warm cache, worker acquisition and IPC included. MP is Python mpmath versus C++ MPFR at 60 decimal digits; geometry is Float64. Accuracy truth computed at 100 digits.',
        exact_fields=dict(stream_potential='psi=product(sin(pi*x_j))/pi',
                          velocity_2d='(psi_y,-psi_x)',
                          velocity_3d='curl((psi,2*psi,3*psi))',
                          pressure='product(cos(pi*x_j))'),
        preparation=[],results=[],comparisons=[],failures=[])
    with threadpool_limits(limits=1),without_condition_diagnostics():
        from rbflab.kernel_compiler import toolchain
        data['environment']['compiler']=toolchain()[1]
        data['environment']['threadpools']=threadpool_info()
        for d in args.dimensions:
            model,problem,queries,ctx,truth=problem_data(d,args.queries)
            for precision in args.precisions:
                digits=60 if precision=='mp' else None
                common=dict(spaces=r.MixedSpace({model.fields[0]:r.DivergenceFreeSpace(r.PHS(7),polynomial_degree=2),model.fields[1]:r.PressureSpace(r.PHS(3),polynomial_degree=1)}),precision=r.Precision(global_digits=digits))
                methods=dict(python=r.GlobalCollocation(**common),cpp=r.GlobalCollocation(**common,backend=r.CppGlobalBackend(threads=1,compute_condition=False,timeout=1200,cache_dir=Path(args.cache_dir).resolve())))
                t=time.perf_counter();methods['cpp']=methods['cpp'].prepare(problem,dimension=d)
                data['preparation'].append(dict(dimension=d,precision=precision,seconds=time.perf_counter()-t))
                for backend,method in methods.items():
                    measure(method,problem,r.geometry.unit_box_grid(2,d),queries[:2],digits,1,ctx,{k:[v[:2] for v in vals] for k,vals in truth.items()},backend)
                cells=args.cells or ([2,3,4,5] if precision=='mp' and d==2 else [2,3,4] if precision=='mp' else [2,4,6,8,10] if d==2 else [2,3,4,5,6])
                for index,n in enumerate(cells):
                    cloud=r.geometry.unit_box_grid(n,d);pair={}
                    for backend in (['python','cpp'] if index%2==0 else ['cpp','python']):
                        meta=dict(dimension=d,precision=precision,nodes=len(cloud.points),cells=n,backend=backend)
                        print('Starting '+str(meta),flush=True)
                        try:
                            result,values=measure(methods[backend],problem,cloud,queries,digits,args.repeats,ctx,truth,backend)
                            data['results'].append(dict(**meta,**result));pair[backend]=(result,values)
                            print({k:result[k] for k in ['assembly_lu_s','solve_s','evaluation_s','velocity_max_error','gradient_max_error']},flush=True)
                        except Exception as exc:
                            data['failures'].append(dict(**meta,error=repr(exc)));print(repr(exc),flush=True)
                        save(data,output/'results.json')
                    if len(pair)==2:
                        a,av=pair['python'];b,bv=pair['cpp']
                        data['comparisons'].append(dict(dimension=d,precision=precision,nodes=len(cloud.points),speedups={k:a[k]/b[k] for k in ['assembly_lu_s','solve_s','evaluation_s','total_s']},backend_gaps={k:float(max(abs(x-y) for x,y in zip(av[k],bv[k]))) for k in av}))
                        save(data,output/'results.json')
    data['finished']=datetime.now(timezone.utc).isoformat();save(data,output/'results.json');report(data,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/global_fields')
    parser.add_argument('--cache-dir',default='outputs/global_fields_native_cache')
    parser.add_argument('--dimensions',nargs='+',type=int,choices=[2,3],default=[2,3])
    parser.add_argument('--precisions',nargs='+',choices=['double','mp'],default=['double','mp'])
    parser.add_argument('--cells',nargs='+',type=int)
    parser.add_argument('--queries',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--report-only',action='store_true',help='Regenerate plots and HTML from saved results')
    args=parser.parse_args()
    if args.report_only:
        import json
        output=Path(args.output).resolve()
        report(json.loads((output/'results.json').read_text(encoding='utf-8')),output)
    else:
        run(args)
