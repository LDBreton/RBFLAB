"""Generate a compact report from preserved Windows/WSL samples and outputs."""
import argparse
import json
from pathlib import Path
import mpmath
from .benchmark import compare, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    data = json.loads((output/'results.json').read_text())
    lines = ['# Native Windows versus WSL C++ benchmark', '',
        data['methodology'], '',
        '## Environment', '',
        f"- CPU: {data['environment']['cpu']}",
        f"- Windows: {data['environment']['windows_compiler']}",
        f"- WSL: {data['environment']['wsl_compiler']}",
        '- Windows MPFR 4.2.2 / GMP 6.3.0; WSL MPFR 4.2.1 / GMP 6.3.0.',
        '- Same generated source, Eigen headers, geometry, operators, RHS and query points.',
        '- Platform compiler/runtime versions differ. Results compare installed toolchains.',
        '', '## Runtime medians', '',
        'All times are seconds. Speedup = WSL / Windows; above 1 means Windows was faster.', '',
        '| Problem | Arithmetic | Nodes | Unknowns | Threads | Windows native | WSL native | Windows speedup | Windows total | WSL total |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['results']:
        w, l = row['medians']['windows'], row['medians']['wsl']
        lines.append(f"| {row['kind']} | {row['arithmetic']} | {row['nodes']} | {row['unknowns']} | "
            f"{row['threads']} | {w['native_total_s']:.6f} | {l['native_total_s']:.6f} | "
            f"{row['windows_speedup']['native_total_s']:.2f}x | {w['total_s']:.6f} | {l['total_s']:.6f} |")
    lines += ['', 'Native time surrounds worker commands, including numeric parsing/formatting. '
        'Total adds process/WSL launch and Windows Python pipe I/O. These fresh-worker totals '
        'do not represent an application that retains workers across multiple assemblies. '
        'The smallest millisecond-scale cases are sensitive to scheduling.', '',
        '## Phase medians (largest problem size)', '',
        '| Problem | Arithmetic | Threads | Windows build/LU | WSL build/LU | Windows solve | WSL solve | Windows eval | WSL eval |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    largest = max(row['nodes'] for row in data['results'])
    for row in data['results']:
        if row['nodes'] != largest:
            continue
        w, l = row['medians']['windows'], row['medians']['wsl']
        values = [w['build_s'],l['build_s'],w['solve_s'],l['solve_s'],w['evaluation_s'],l['evaluation_s']]
        lines.append(f"| {row['kind']} | {row['arithmetic']} | {row['threads']} | "+
                     ' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '## Full precision agreement', '',
        'MPFR text is parsed at 100 digits for comparison. No Float64 conversion is used. '
        'The scale for each coefficient/field vector is max(1, maximum absolute value on either host); '
        'residuals are checked separately. Fixed tolerances are 1e-8 (Float64) and 1e-48 (60-digit MPFR). '
        'Each host evaluates its own solved coefficients. EVAL_0 is scalar value or velocity; '
        'Stokes EVAL_1/2/3 are pressure gradient, pressure and divergence.', '',
        '| Case | Max field absolute gap | Max field scaled gap | Coefficient scaled gap | Pass |',
        '|---|---:|---:|---:|---|']
    ctx = mpmath.mp.clone(); ctx.dps = 100
    for row in data['results']:
        outputs = json.loads((output/(row['label']+'-outputs.json')).read_text())
        row['agreement'] = compare(outputs['windows'],outputs['wsl'],row['digits'])
        checks = row['agreement']['commands']
        fields = [v for k,v in checks.items() if k.startswith('EVAL')]
        absolute = max(ctx.mpf(v['max_abs']) for v in fields)
        scaled = max(ctx.mpf(v['scaled_max']) for v in fields)
        row['max_field_absolute_gap'] = ctx.nstr(absolute,30)
        row['max_field_scaled_gap'] = ctx.nstr(scaled,30)
        coeff = checks['SOLVE_coefficients']['scaled_max']
        lines.append(f"| {row['label']} | {float(absolute):.3e} | {float(scaled):.3e} | "
                     f"{float(coeff):.3e} | {row['agreement']['passed']} |")
    data['all_agree'] = all(row['agreement']['passed'] for row in data['results'])
    data['all_fields_agree'] = all(row['agreement']['fields_passed'] for row in data['results'])
    diagnostic=output/'matrix-diagnostic.json'
    if diagnostic.is_file():
        d=json.loads(diagnostic.read_text())
        lines += ['', f"Larger Float64 Poisson diagnostic: matrix maximum absolute gap {d['matrix_max_abs_gap']:.3e}; "
            f"relative Frobenius gap {d['matrix_relative_frobenius_gap']:.3e}; unscaled augmented matrix "
            f"2-norm condition estimate {d['windows_matrix_condition_2_float64_estimate']:.3e}. "
            'This sensitivity is consistent with larger coefficient differences than field differences. '
            'The original coefficient tolerance failures remain recorded.']
    lines += ['', '## Cold compilation', '',
        'These are single samples, separated from runtime. Production source capture/symbolic preparation '
        'is recorded separately in results.json and excluded from these compiler timings.', '',
        '| Problem | Arithmetic | Windows seconds | WSL seconds |', '|---|---|---:|---:|']
    for row in data['builds']:
        lines.append(f"| {row['kind']} | {row['arithmetic']} | {row['compile_s']['windows']:.3f} | {row['compile_s']['wsl']:.3f} |")
    local=output/'local-results.json'
    if local.is_file():
        local_data=json.loads(local.read_text())
        lines += ['', '## Local Stokes LHI weights', '', local_data['methodology'], '',
            '| Arithmetic | Threads | Windows native | WSL native | Speedup WSL/Windows | Windows total | WSL total | Scaled weight gap | Pass |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for row in local_data['results']:
            w,l=row['medians']['windows'],row['medians']['wsl']
            lines.append(f"| {row['arithmetic']} | {row['threads']} | {w['native_s']:.6f} | {l['native_s']:.6f} | "
                f"{l['native_s']/w['native_s']:.2f}x | {w['total_s']:.6f} | {l['total_s']:.6f} | "
                f"{float(row['agreement']['scaled_weight_gap']):.3e} | {row['agreement']['passed']} |")
        lines += ['', 'Local native times include file I/O through Windows paths or the WSL /mnt mount, '
            'so these are batch-engine timings rather than isolated arithmetic timings.']
    lines += ['', 'Raw samples, serialized inputs, full-precision outputs, source/binary files and build logs '
        'are preserved beside this report. Scope: these 2D stationary scalar and Stokes dense workloads; '
        'plus the local Stokes LHI batch above. CUDA, MPLAPACK and other problem sizes are untested.', '']
    save(output/'results.json',data)
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Fields agree:',data['all_fields_agree'],'all coefficient checks:',data['all_agree'], 'cases:',len(data['results']))
    print(output/'report.md')


if __name__ == '__main__':
    main()
