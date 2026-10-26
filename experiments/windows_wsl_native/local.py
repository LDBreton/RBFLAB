"""Matched dedicated local Stokes LHI batch benchmark (run after global.py)."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import mpmath
import rbflab as r
from rbflab.legacy_cpp import _linux_path
from rbflab.native_toolchain import compiler, compiler_environment, arithmetic_flags
from rbflab.eigen_build import eigen_dependency
from .benchmark import ROOT, save


def capture(digits, threads, output):
    from examples.tutorials.stokes_spaces import make_problem
    problem, original_spaces = make_problem(False)
    u, p = tuple(original_spaces)
    spaces = {u:r.DivergenceFreeSpace(r.IMQ(2),None),p:r.PressureSpace(r.IMQ(2),None)}
    method = r.LHI(spaces=spaces, stencil_size=12,
        precision=r.Precision(local_digits=digits),
        local_backend=r.CppBackend(threads=threads,compute_condition=False))
    source = output/f"local-{'mpfr' if digits else 'float64'}-t{threads}-input.txt"
    original = subprocess.run

    def record(args, *positional, **kwargs):
        if args and Path(args[0]).name in ('lhi_double.exe','lhi_mpfr.exe'):
            source.write_bytes(Path(args[1]).read_bytes())
        return original(args,*positional,**kwargs)

    subprocess.run = record
    try:
        method.assemble(problem,r.geometry.unit_box_grid(10))
    finally:
        subprocess.run = original
    assert source.is_file()
    return source


def compile_pair(arithmetic, output):
    folder = output/'local-binaries'/arithmetic
    folder.mkdir(parents=True,exist_ok=True)
    source = '#include <chrono>\n'+(ROOT/'cpp/lhi_mpfr.cpp').read_text()
    source = source.replace('int main(int argc,char**argv){try{',
        'int main(int argc,char**argv){try{auto bench_start=std::chrono::steady_clock::now();')
    marker=' return 0;\n}catch'
    assert marker in source
    source=source.replace(marker,
        ' std::cerr<<"BENCH_LOCAL "<<std::setprecision(17)'
        '<<std::chrono::duration<double>(std::chrono::steady_clock::now()-bench_start).count()<<"\\n";\n'+marker)
    (folder/'worker.cpp').write_text(source)
    for name in ('real.hpp','real_double.hpp','eigen_solver.hpp','generated_imq.hpp','generated_hybrid.hpp'):
        (folder/name).write_bytes((ROOT/'cpp'/name).read_bytes())
    binaries={}; compile_times={}
    for host in ('windows','wsl'):
        executable=folder/('worker.exe' if host=='windows' else 'worker-linux')
        if host=='windows':
            cmd=[compiler(),'-O3','-std=c++17','-fopenmp','-I'+str(eigen_dependency()[0]),
                 str(folder/'worker.cpp'),*arithmetic_flags(arithmetic),'-o',str(executable)]
            env=compiler_environment()
        else:
            include=ROOT/'cpp/deps/usr/include'
            flags=['-DRBFLAB_DOUBLE'] if arithmetic=='float64' else [
                '-I'+_linux_path(include),'-I'+_linux_path(include/'x86_64-linux-gnu'),
                '-Wl,-l:libmpfr.so.6','-Wl,-l:libgmp.so.10']
            cmd=['wsl.exe','-d','Ubuntu','--','g++','-O3','-std=c++17','-fopenmp',
                 '-I'+_linux_path(eigen_dependency()[0]),_linux_path(folder/'worker.cpp'),
                 *flags,'-o',_linux_path(executable)]
            env=None
        start=time.perf_counter()
        run=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=300)
        compile_times[host]=time.perf_counter()-start
        (folder/(host+'-compile.log')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(run.stderr)
        binaries[host]=executable
    return binaries,dict(compile_s=compile_times,source_sha256=hashlib.sha256(source.encode()).hexdigest())


def run(host, executable, source, target):
    cmd=([str(executable),str(source),str(target)] if host=='windows' else
         ['wsl.exe','-d','Ubuntu','--',_linux_path(executable),_linux_path(source),_linux_path(target)])
    start=time.perf_counter()
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
    total=time.perf_counter()-start
    if result.returncode:raise RuntimeError(result.stderr)
    native=float(next(line.split()[1] for line in result.stderr.splitlines() if line.startswith('BENCH_LOCAL ')))
    return dict(native_s=native,total_s=total)


def compare(left,right,digits):
    ctx=mpmath.mp.clone();ctx.dps=100
    weights=[]; residuals=[]
    arows,brows=left.read_text().splitlines(),right.read_text().splitlines()
    assert len(arows)==len(brows)
    for a,b in zip(arows,brows):
        at,bt=a.split(),b.split()
        assert at[:3]==bt[:3] and len(at)==len(bt)
        residuals.append(abs(ctx.mpf(at[3])-ctx.mpf(bt[3])))
        weights.extend((ctx.mpf(x),ctx.mpf(y)) for x,y in zip(at[4:],bt[4:]))
    gap=max(abs(a-b) for a,b in weights)
    scale=max(ctx.mpf(1),max(abs(a) for a,b in weights),max(abs(b) for a,b in weights))
    tolerance=ctx.mpf('1e-48') if digits else ctx.mpf('1e-8')
    return dict(stencils=len(arows),max_weight_abs_gap=ctx.nstr(gap,30),
        scaled_weight_gap=ctx.nstr(gap/scale,30),max_residual_gap=ctx.nstr(max(residuals),30),
        scaled_tolerance=str(tolerance),passed=gap/scale<tolerance and max(residuals)<tolerance)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/windows-wsl-native')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    data=dict(methodology='121-node unit-box grid; 81 local interior stencils; 12 geometric neighbors; '
        'stationary 2D Stokes, IMQ(2) velocity/pressure, no polynomial augmentation; Float64/60-digit MPFR; '
        'condition diagnostics off; five alternating samples after one warmup per host. Identical serialized '
        'input and generated native source on both hosts. Native seconds include input/output file I/O, '
        'weight assembly, LU, solves, residual and formatting; total also includes process/WSL launch.',
        builds=[],results=[])
    for digits in (None,60):
        arithmetic='mpfr' if digits else 'float64'
        print('Compile local',arithmetic,flush=True)
        binaries,evidence=compile_pair(arithmetic,output)
        data['builds'].append(dict(arithmetic=arithmetic,**evidence))
        for threads in (1,4):
            source=capture(digits,threads,output)
            targets={h:output/f'local-{arithmetic}-t{threads}-{h}-output.txt' for h in binaries}
            samples={h:[] for h in binaries}
            for h in binaries:run(h,binaries[h],source,targets[h])
            for repeat in range(args.repeats):
                for h in (('windows','wsl') if repeat%2==0 else ('wsl','windows')):
                    samples[h].append(run(h,binaries[h],source,targets[h]))
            medians={h:{k:statistics.median(s[k] for s in values) for k in ('native_s','total_s')}
                     for h,values in samples.items()}
            agreement=compare(targets['windows'],targets['wsl'],digits)
            data['results'].append(dict(arithmetic=arithmetic,threads=threads,samples=samples,
                                       medians=medians,agreement=agreement))
            save(output/'local-results.json',data)
            print(arithmetic,threads,medians,agreement,flush=True)


if __name__=='__main__':main()
