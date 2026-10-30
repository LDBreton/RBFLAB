"""Collect the degree-6/2 control experiment without rerunning any solve."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .control_report import field_difference


CELLS = (89, 126)
DTS = (.00025, .000125)


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def load_case(root, cells, dt, allow_incomplete=False):
    folder = root/'cavity_newton_krylov'/f'cells{cells}_dt{dt}'
    result = read(folder/'result.json')
    if result is None:
        if not allow_incomplete:
            raise ValueError(f'Missing final case report: {folder}')
        return dict(cells=cells, total_nodes=(cells+1)**2, dt=dt,
                    status='missing', evidence=str(folder)), []
    configuration = read(folder/'configuration.json')
    history = read(folder/'history.json', [])
    if result['status'] == 'checkpoint' and not allow_incomplete:
        raise ValueError(f'Incomplete production case: {folder}')
    if configuration['augmentation'] not in ('6/2', [6, 2]):
        raise ValueError('Production configuration does not declare degree-6/2')
    count = result['completed_steps']
    if [r['step'] for r in history] != list(range(1, count+1)):
        raise ValueError('Accepted history is not contiguous')
    if count and abs(history[-1]['time']-result['time']) > 1e-13:
        raise ValueError('History and final report times differ')
    maps = Path(configuration['maps'])
    for name, digest in configuration['input_hashes'].items():
        if hashlib.sha256((maps/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed matrix input: {maps/name}')
    with np.load(folder/'fields.npz') as fields:
        if abs(float(fields['time'])-result['time']) > 1e-13:
            raise ValueError('Fields and final report times differ')
        if not all(np.isfinite(fields[name]).all() for name in ('U','L','gradient')):
            raise ValueError('Nonfinite saved field')
        result['final_nodal_velocity_vector_rms'] = float(np.sqrt(np.mean(
            np.sum(fields['U'].reshape(-1,2)**2, axis=1))))
    return dict(result, evidence=str(folder), configuration=configuration), history


def compare_fields(left, right):
    if left.get('status') != 'completed' or right.get('status') != 'completed':
        return dict(scope='No comparison: at least one case did not complete')
    if abs(left['time']-right['time']) > 1e-13:
        return dict(scope='No comparison: endpoint times differ')
    with np.load(Path(left['evidence'])/'fields.npz') as a, np.load(Path(right['evidence'])/'fields.npz') as b:
        np.testing.assert_array_equal(a['points'], b['points'])
        return dict(time=left['time'], velocity=field_difference(a['U'],b['U']),
                    pressure_gradient=field_difference(a['gradient'],b['gradient']))


def collect(root, baseline, allow_incomplete=False):
    root, baseline = Path(root), Path(baseline)
    cases, histories, timestep, baseline_comparisons = [], {}, [], []
    for cells in CELLS:
        for dt in DTS:
            result, history = load_case(root,cells,dt,allow_incomplete)
            cases.append(result)
            histories[cells,dt] = history
            old_folder = baseline/'cavity_newton_krylov'/f'cells{cells}_dt{dt}'
            old = read(old_folder/'result.json')
            if old:
                old['evidence'] = str(old_folder)
                baseline_comparisons.append(dict(nodes=(cells+1)**2,dt=dt,
                    **compare_fields(result,old)))
        timestep.append(dict(nodes=(cells+1)**2,
            **compare_fields(cases[-2],cases[-1])))
    maps = {}
    for cells in CELLS:
        folder = root/'maps'/f'cells{cells}_nu2'
        maps[str(cells)] = {name:read(folder/f'{name}.json') for name in
            ('configuration','assembly','pressure_patch','homogeneous_probes','stokes_audit','mpi_parity')}
    summary = dict(cases=cases,maps=maps,timestep_comparisons=timestep,
        baseline_comparisons=baseline_comparisons,
        gradient_checks=read(root/'audit/gradient_summary.json'),
        tests=read(root/'TESTS.json'),
        independent_evolution_audit=read(root/'audit/evolution_audit.json'),
        spectral_checks=read(root/'audit/spectral_summary.json',read(root/'audit/spectrum_summary.json')),
        historical_parity=read(root/'audit/historical_parity.json'),
        small_stokes_audit=read(root/'maps/cells12_nu2/stokes_audit.json'),
        small_pressure_controls=[read(root/f'small_pressure_dt{dt}/result.json') for dt in DTS],
        pressure_controls=[read(root/'pressure_control'/f'cells{cells}_dt{dt}'/'result.json') for cells in CELLS for dt in DTS],
        baseline=str(baseline), historical_scope='Earlier P2 Gaussian maps use the same bitwise cloud. Full Stokes/pressure map parity is saved in audit/historical_parity.json when available; historical convection used separate DF5 maps.')
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    write_report(root,summary)
    plot(root,histories)
    plot_modes(root,summary.get('spectral_checks'))
    print(json.dumps(dict(cases=[{k:r.get(k) for k in ('total_nodes','dt','status','time','peak_nodal_speed')}
        for r in cases],timestep_comparisons=timestep,baseline_comparisons=baseline_comparisons)),flush=True)
    return summary


def number(value):
    return '-' if value is None else f'{value:.8g}'


def write_report(root,s):
    completed=sum(r['status']=='completed' for r in s['cases'])
    failed=sum(r['status']=='failed' for r in s['cases'])
    spectrum=s.get('spectral_checks') or []
    if not isinstance(spectrum,list):
        spectrum=spectrum.get('results',[])
    positive=[r for r in spectrum if r.get('credible_positive_returned',0)]
    xy=[s['maps'][str(cells)]['pressure_patch'].get('xy') for cells in CELLS
        if s['maps'][str(cells)]['pressure_patch']]
    pressure_outcome=(f"True degree-6/2 augmentation reproduces the p=xy patch on both meshes: the largest static pressure-gradient component defect is {max(r['gradient_error_inf'] for r in xy):.3e}. "
        if len(xy)==2 else 'Quadratic pressure reproduction checks are pending. ')
    outcome=('Current-map checks verify a positive finite Stokes mode on '+', '.join(f"{r['total_nodes']:,}" for r in positive)+
        ' nodes. The requested implicit time steps damp that fast growing mode artificially; bounded cavity advancement at those steps does not establish a stable spatial operator. '
        if positive else 'Current large-grid growing-mode checks are pending or inconclusive. ')
    lines=['# Quadratic pressure augmentation with Stokes control and a discrete adjoint','',
        '2026-10-30. '+pressure_outcome+f'Of four requested large-grid cavity runs, {completed} completed, {failed} failed, and {4-completed-failed} remain incomplete. '
        +outcome+'Pressure reproduction, control convergence, linear growth diagnostics and cavity accuracy are reported separately.','',
        '## Fixed method','',
        'Unit-square lid-driven cavity, nu=0.01 (Re=100), unit horizontal lid except stationary corner nodes, stationary other walls, initial rest, zero physical body force. '
        'Meshes: 8,100 and 16,129 total nodes (89 and 126 cells). Nearest-37 physical-site mixed LHI, full scaling q=cells/44, local pressure elimination. '
        'Velocity augmentation remains divergence-free degree 6; pressure augmentation is now complete degree 2 modulo its invisible constant. '
        'The five pressure monomials are x,y,x^2,xy,y^2; these are part of the actual mixed polynomial saddle system, not a later common-moment correction. '
        'Velocity potential exp(-10*q^2*r^2)+q^7*r^7; pressure potential nu^2*q^4*exp(-q^2*r^2), cbar_p=1. '
        'Direct wall data are used, without corner subtraction or a finite-element representation.','',
        'Each production step minimizes J=0.5*dt*h^2*sum |f-C(U,L,g)|^2 under the linear backward-Euler Stokes equations. '
        'The exact discrete adjoint uses the transpose of the assembled Stokes block in reversed time. '
        'Convection and its Jacobian include both velocity and local Stokes momentum observations. '
        'Newton-Krylov gives a residual direction, accepted by an objective line search using the adjoint gradient. '
        'The condensed Stokes LU is reused within a process. A step is accepted only when the control residual component maximum is <=1e-8. '
        'The full momentum and mixed constraint residuals are computed and recorded independently.','',
        'The baseline is the saved degree-6/1 control study. Both use the same kernels, viscosity, cloud, stencils, time steps and wall convention; only the pressure polynomial degree changes. '
        'Geometry and global solves are Float64; new local weights use native MPFR80 with actual MPI execution. '
        'See maps/*/configuration.json and assembly.json for exact ranks, cache/reuse scope and execution records.','',
        '| Nodes | MPI ranks | Requested templates | Fresh templates | Reused templates | Assembly seconds | Rank records successful |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for cells in CELLS:
        record=s['maps'][str(cells)]
        assembly=record.get('assembly') or {}
        execution=assembly.get('execution',{})
        workers=[w for batch in execution.get('batches',[]) for w in batch.get('execution',{}).get('workers',[])]
        okay=all(w.get('returncode')==0 for w in workers) if workers else 'pending'
        lines.append(f"| {(cells+1)**2} | {(record.get('configuration') or {}).get('ranks','-')} | {execution.get('requested_templates','-')} | {execution.get('computed_templates','-')} | {execution.get('reused_templates','-')} | {number(assembly.get('seconds'))} | {okay} |")
    lines += ['',
        '## Pressure-only patch test','',
        'For exact U=0 and homogeneous wall velocity, p=xy has L=grad(p)=(y,x). '
        'The mixed relation requires M L=0; reconstructed pressure gradient requires PL L=L; reconstructed velocity derivatives require DxL L=DyL L=0. '
        'These are absolute nodal component maxima from the static maps, not evolved velocity errors.','',
        '| Nodes | Pressure | Constraint defect max | Gradient defect max | dx velocity defect max | dy velocity defect max |',
        '|---:|---|---:|---:|---:|---:|']
    for cells in CELLS:
        for name,row in (s['maps'][str(cells)]['pressure_patch'] or {}).items():
            lines.append(f"| {(cells+1)**2} | {name} | {number(row.get('constraint_inf'))} | {number(row.get('gradient_error_inf'))} | {number(row.get('dx_velocity_inf'))} | {number(row.get('dy_velocity_inf'))} |")
    lines += ['', 'For comparison, the degree-6/1 baseline p=xy gradient defects were 0.0225359066 and 0.0159182198 on the two meshes. '
        'Polynomial exactness on this test does not establish general pressure convergence or evolution stability.','']
    pressure_controls=[row for row in s.get('pressure_controls',[]) if row]
    if pressure_controls:
        lines += ['The following pressure-only evolution uses the actual nonlinear control solver, homogeneous velocity walls, exact U(0)=0, '
            'L(0)=grad(xy), and physical body force grad(xy). Thus this tests evolved spurious velocity as well as static reproduction.','',
            '| Nodes | dt | Status | Actual time | Peak velocity component error | Peak pressure-gradient component error |',
            '|---:|---:|---|---:|---:|---:|']
        for row in pressure_controls:
            lines.append(f"| {row['total_nodes']} | {row['dt']} | {row['status']} | {row['time']} | {row['peak_velocity_component_max']:.8e} | {row['peak_gradient_error_component_max']:.8e} |")
        lines += ['']
    lines += [
        '## Cavity evolution','',
        'Requested T=0.1, with dt=0.00025 and 0.000125. A nodal speed bound of 10 stops a run and preserves its finite candidate. '
        'Failures are reported at the actual accepted time; a converged optimization objective does not imply an accurate cavity field.','',
        '| Nodes | dt | Status | Actual time | Peak interior nodal speed | Max control residual | Max momentum residual | Max mixed constraint |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in s['cases']:
        lines.append(f"| {row['total_nodes']} | {row['dt']} | {row['status']} | {number(row.get('time'))} | {number(row.get('peak_nodal_speed'))} | {number(row.get('max_control_residual_inf'))} | {number(row.get('max_momentum_residual_inf'))} | {number(row.get('max_constraint_residual_inf'))} |")
        if row.get('failure'):
            lines += ['',f"Failure for {row['total_nodes']} nodes, dt={row['dt']}: `{json.dumps(row['failure'],sort_keys=True)}`",'']
    lines += ['', 'Mapped local divergence cancellation is distinct from global compatibility of overlapping reconstructions. '
        'All solution metrics below use interior mesh nodes only.','',
        '## Same-time nodal comparisons','',
        'Vector RMS is sqrt(mean(|difference|^2)); these are field differences, not exact-solution errors. '
        'Endpoint differences are omitted unless both compared runs completed at the same time.','',
        '| Comparison | Nodes | dt | Time | Velocity RMS difference | Pressure-gradient RMS difference |',
        '|---|---:|---:|---:|---:|---:|']
    for kind,rows in [('P2 time-step halving',s['timestep_comparisons']),('P2 minus P1 baseline',s['baseline_comparisons'])]:
        for row in rows:
            lines.append(f"| {kind} | {row['nodes']} | {row.get('dt','both')} | {number(row.get('time'))} | {number(row.get('velocity',{}).get('vector_rms'))} | {number(row.get('pressure_gradient',{}).get('vector_rms'))} |")
    lines += ['', 'The 8,100-node cloud contains no exact Ghia centerline nodes, so a centerline comparison is unavailable without interpolating the solution. '
        'On 16,129 nodes, the reference table is interpolated onto existing centerline nodes while computed fields remain nodal. '
        'Short-time differences from steady Ghia values do not constitute steady cavity accuracy.','',
        '| Nodes | dt | Time | u centerline RMS difference | v centerline RMS difference |',
        '|---:|---:|---:|---:|---:|']
    for row in s['cases']:
        g=row.get('ghia',{})
        if g.get('status')=='available':
            lines.append(f"| {row['total_nodes']} | {row['dt']} | {row['time']} | {g['u']['rms_difference']:.8g} | {g['v']['rms_difference']:.8g} |")
    lines += ['', '## Gradient, MPI and growth checks','',
        'The full gradient audit is saved in audit/gradient_summary.json, with directional finite-difference and transpose checks. '
        'The independent production audit is saved in audit/evolution_audit.json when present. '
        'Map manifests and MPI launcher/rank records document local assembly.','']
    gradients=s.get('gradient_checks')
    if isinstance(gradients,list):
        lines += ['| Nodes | dt | Random-direction relative FD error | Gradient-aligned relative FD error | Transpose relative error |',
                  '|---:|---:|---:|---:|---:|']
        for row in gradients:
            lines.append(f"| {row['nodes']} | {row['dt']} | {row['directions']['random']['best_relative_error']:.3e} | {row['directions']['gradient_aligned']['best_relative_error']:.3e} | {row['transpose_relative_error']:.3e} |")
    else:
        lines += ['Gradient aggregate is pending; no numerical claim is made from its absence.']
    parity_scope=('Complete entrywise parity of the Stokes and pressure maps is reported below. The old eigenvalue remains an earlier measurement; '
        'current-map eigenvector residuals and growth probes are reported separately. ' if s.get('historical_parity') else
        'Complete entrywise parity with the new matrices is pending, so this remains historical evidence rather than a newly measured eigenvalue of these maps. ')
    lines += ['', 'A historical true-P2 Gaussian cpbar=1, nu^2*q^4 operator at 16,129 nodes had a verified positive finite Stokes mode near 47,595.7 '
        '(reported original-pencil residual 9.68e-14). The historical cloud and node-index arrays are bitwise equal to the current cloud, '
        'and the nearest-neighbor selection code follows the same query/sorting convention. '+parity_scope+'Historical convection used separate DF5 derivatives; '
        'that distinction does not change the homogeneous Stokes pencil. '
        'At sufficiently large implicit steps a fast positive mode can be artificially damped; completing the two requested time steps alone cannot establish absence of growing modes. '
        'Historical source: outputs/lhi_df_cavity_pressure_cp_p2_20261028/REPORT.md.','']
    if s.get('spectral_checks') is not None:
        lines += ['Current-map targeted spectral/perturbation checks are saved in audit/spectrum_summary.json (or spectral_summary.json) and included without modification in SUMMARY.json. '
                  'Bounded or negative searches do not certify the complete spectrum.','',
                  '| Nodes | Selected sampled real rate | Imaginary part | Original-pencil relative residual | Credible positive returned | Search status |',
                  '|---:|---:|---:|---:|---|---|']
        records=s['spectral_checks']
        if not isinstance(records,list):
            records=records.get('results',[])
        for row in records:
            mode=row.get('selected_mode') or {}
            status='Inconclusive: no eigenvector converged' if row.get('search_failures') and not mode else 'Current-pencil residual verified' if mode else 'Bounded search only'
            lines.append(f"| {row.get('total_nodes',(row['cells']+1)**2)} | {number(mode.get('real'))} | {number(mode.get('imag'))} | {number(mode.get('pencil_relative_residual'))} | {row.get('credible_positive_returned','-')} | {status} |")
        lines += ['', '| Nodes | Probe dt | Actual time | Final mode norm gain | Peak mode norm gain |',
                  '|---:|---:|---:|---:|---:|']
        for row in records:
            for probe in row.get('mode_probes',[]):
                lines.append(f"| {row.get('total_nodes',(row['cells']+1)**2)} | {probe['dt']} | {number(probe.get('actual_end'))} | {number(probe.get('final_l2_gain'))} | {number(probe.get('peak_l2_gain'))} |")
        lines += ['', 'For a positive real semidiscrete rate lambda, backward Euler multiplies its mode by 1/(1-dt*lambda). '
            'The two requested steps can lie in the artificial damping regime for a fast positive rate; the much smaller mode-probe step exposes its growth. '
            'A failed ARPACK search is inconclusive and does not establish absence of growing modes.','']
    if s.get('historical_parity'):
        parity=s['historical_parity']
        lines += ['The newly assembled 16,129-node momentum and pressure maps were compared directly with that historical true-P2 Gaussian operator. '
            'Historical component-major rows/columns are permuted to the current node-major order, with a separate boundary-column permutation. '
            'Cloud points and interior/boundary index arrays are bitwise equal. Full comparison records and hashes are in audit/historical_parity.json. '
            'Extremely small differences here arise in near-zero MPFR coefficients exported to Float64; they do not imply that general Float64 algebra has 70-digit accuracy.','',
            '| Map | Max absolute difference | Max relative difference | Relative Frobenius difference |',
            '|---|---:|---:|---:|']
        for row in parity['matrices']:
            lines.append(f"| {row['new_name']} | {row['max_absolute']:.3e} | {row['max_relative']:.3e} | {row['relative_frobenius']:.3e} |")
        lines += ['']
    if s.get('small_stokes_audit'):
        small=s['small_stokes_audit']['stokes_spectrum']
        lines += [f"The separate 169-node implementation check has {small['positive_count']} positive finite Stokes mode(s); "
            f"the leading real part is {small['leading_real']:.10g}, with original-pencil relative residual {small['action_relative_residual']:.3e}. "
            'This is current small-grid evidence, not a measured mode of either large grid.','']
    controls=[r for r in s.get('small_pressure_controls',[]) if r]
    if controls:
        lines += ['On that small grid, direct pressure-only evolution with p=xy remains near zero velocity over its recorded test interval, '
            'even though the homogeneous operator has a growing mode. This illustrates the distinction between polynomial pressure balance and stability.','',
            '| Small-grid pressure dt | Status | Time | Peak velocity component error | Peak pressure-gradient component error |',
            '|---:|---|---:|---:|---:|']
        for row in controls:
            lines.append(f"| {row['dt']} | {row['status']} | {row['time']} | {row['peak_velocity_component_max']:.3e} | {row['peak_gradient_error_component_max']:.3e} |")
        lines += ['']
    for cells in CELLS:
        if s['maps'][str(cells)]['homogeneous_probes']:
            lines += [f"Chosen-direction homogeneous Stokes evolution records for {(cells+1)**2} nodes: maps/cells{cells}_nu2/homogeneous_probes.json. "
                'These probes do not certify worst-case transient stability.','']
    lines += ['## Tests and reproduction','']
    if s.get('tests'):
        lines += ['Executed test record: `'+json.dumps(s['tests'],sort_keys=True)+'`.','']
    lines += ['Research sources: control_quadratic_maps.py builds the actual P2 maps; stokes_control.py implements the common control/adjoint solver; '
        'control_cavity.py runs restartable cavity windows; control_quadratic_report.py only collects saved evidence. '
        'All old experiments and the P1 baseline are preserved. No public API capability is inferred from this research driver.','',
        '![Nodal cavity diagnostics](cavity_diagnostics.png)','']
    if any(row.get('mode_probes') for row in spectrum):
        lines += ['The following plot evolves the same selected current-map homogeneous Stokes eigenmode. '
            'It shows artificial damping at the two requested implicit steps and growth at the smaller probe step. '
            'This is a mode diagnostic, not a cavity velocity error plot.','',
            '![Implicit masking of a positive Stokes mode](mode_masking.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def plot(root,histories):
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    metrics=[('max_nodal_speed','Maximum interior nodal speed',False),
             ('kinetic_nodal_mean','Mean nodal kinetic energy',False),
             ('control_residual_inf','Control residual maximum',True),
             ('momentum_residual_inf','NS momentum residual maximum',True)]
    for ax,(key,title,log) in zip(axes.ravel(),metrics):
        for (cells,dt),history in histories.items():
            if history:
                ax.plot([r['time'] for r in history],[max(r[key],1e-30) if log else r[key] for r in history],
                    '-' if dt==DTS[0] else '--',label=f'{(cells+1)**2:,} nodes, dt={dt:g}')
        if log:ax.set_yscale('log')
        ax.set(title=title,xlabel='Physical time');ax.grid(alpha=.25)
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=7)
    fig.savefig(root/'cavity_diagnostics.png',dpi=180);plt.close(fig)


def plot_modes(root,spectral):
    records=spectral or []
    if not isinstance(records,list):
        records=records.get('results',[])
    available=[row for row in records if row.get('mode_probes')]
    if not available:
        return
    fig,axes=plt.subplots(1,len(available),figsize=(8*len(available),4.5),
                          squeeze=False,constrained_layout=True)
    for ax,row in zip(axes.ravel(),available):
        for probe in row['mode_probes']:
            gains=np.asarray(probe['history'],dtype=float)
            times=probe['dt']*np.arange(len(gains))
            ax.semilogy(times,np.maximum(gains,np.finfo(float).tiny),'o-' if len(gains)<5 else '-',
                markersize=5,label=f"dt={probe['dt']:.8g}, {probe['steps']} step(s)")
        rate=row['selected_mode']['real']
        ax.axhline(1.,color='0.5',linewidth=.8,linestyle=':')
        ax.set(title=f"{row['total_nodes']:,} nodes; positive Stokes rate = {rate:.6g}",
               xlabel='Physical time',ylabel='Nodal mode Euclidean-norm gain')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.savefig(root/'mode_masking.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/ns_stokes_control_p2_20261030'))
    parser.add_argument('--baseline',type=Path,default=Path('outputs/ns_stokes_control_20261030'))
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    collect(args.root,args.baseline,args.allow_incomplete)
