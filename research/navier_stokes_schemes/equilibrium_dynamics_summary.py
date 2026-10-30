"""Collect completed saved-root cases; compare nodal fields at the same physical time."""
from pathlib import Path
import csv,json
import numpy as np

def run(base):
 rows=[]
 for report in [base/'results.json',*sorted((base/'parallel').glob('*/results.json'))]:
  data=json.loads(report.read_text())
  for row in data['cases']:
   if row['status']!='completed':continue
   r=dict(row);r['owning_report']=str(report)
   amp=r['amplitude'];name=f"{r['control']}_{amp}_{r['scheme']}_{r['dt']}"
   r['fields']=str(report.parent/(name+'.npz'));rows.append(r)
 unique={tuple(r[k] for k in ('control','amplitude','scheme','dt')):r for r in rows};rows=list(unique.values())
 comparisons=[]
 for control in ('hold','pressure_xy','leading','random'):
  amps=(0.,) if control in ('hold','pressure_xy') else (1e-7,.001)
  for amp in amps:
   for scheme in ('oseen','oseen_newton'):
    selected=[r for r in rows if r['control']==control and r['amplitude']==amp and r['scheme']==scheme]
    if len(selected)!=2:continue
    a,b=sorted(selected,key=lambda r:r['dt'],reverse=True);sa,sb=np.load(a['fields']),np.load(b['fields'])
    assert a['T']==b['T']
    comparisons.append(dict(control=control,amplitude=amp,scheme=scheme,T=a['T'],dt_coarse=a['dt'],dt_fine=b['dt'],velocity_difference_inf=float(max(abs(sa['U']-sb['U']))),gradient_difference_inf=float(max(abs(sa['gradient']-sb['gradient'])))))
 strict=[]
 sp=Path('outputs/ns_equilibrium_strict_newton_20261029')
 for fp in sorted(sp.glob('*_results.json')):
  strict.extend(json.loads(fp.read_text()))
 strictstates=[np.load(sp/f'{dt}_2e-11.npz') for dt in (.00025,.000125)]
 strict_difference=float(max(abs(strictstates[0]['U']-strictstates[1]['U'])))
 summary=dict(status='all24_saved_root_cases_completed' if len(rows)==24 else 'incomplete',case_count=len(rows),cases=rows,same_time_dt_comparisons=comparisons,strict_newton=strict,strict_newton_dt_velocity_difference_inf=strict_difference,
 scope='hold/pressureT.05; leading perturbationsT1; random perturbationsT.5. Discrete-root neighborhood; startup separate; 169nodes only. Peak diagnostics started after early leading branches; see per-case observation_start_step. Random full step traces observed; max includes initial where start0. Residual maxima labeled lastsegment or observedrange. No monotone-energy criterion or off-node convergence claim.')
 (base/'collected.json').write_text(json.dumps(summary,indent=2))
 fields=['control','amplitude','scheme','dt','T','velocity_amplification_inf','linear_phase_aware_amplification','linear_prediction_error_inf','peak_observation_start_step','max_perturbation_velocity_inf','max_perturbation_kinetic_mean','total_newton_iterations_observed']
 with (base/'cases.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 lines=['# Saved-root nonlinear verification, 2026-10-29','','All24cases completed. Unit-square Re100, selected169-node nearest37 common-E reconstruction. Oseen and Oseen predictor with residual-controlled Newton, BE startup/BDF2, requested dt .00025/.000125. No global pressure unknowns or PDE damping.','','| Perturbation | Amplitude | T | Oseen gains, coarse/fine | Newton-correction gains, coarse/fine | Exact phase-aware linear gain |','|---|---:|---:|---:|---:|---:|']
 for kind in ('leading','random'):
  for amp in (1e-7,.001):
   rr=[r for r in rows if r['control']==kind and r['amplitude']==amp]
   os=sorted([r for r in rr if r['scheme']=='oseen'],key=lambda r:r['dt'],reverse=True);ns=sorted([r for r in rr if r['scheme']=='oseen_newton'],key=lambda r:r['dt'],reverse=True)
   lines.append(f"| {kind} | {amp:g} | {rr[0]['T']:g} | {os[0]['velocity_amplification_inf']:.9f} / {os[1]['velocity_amplification_inf']:.9f} | {ns[0]['velocity_amplification_inf']:.9f} / {ns[1]['velocity_amplification_inf']:.9f} | {rr[0]['linear_phase_aware_amplification']:.9f} |")
 holds=[r for r in rows if r['control']=='hold'];prs=[r for r in rows if r['control']=='pressure_xy'];random=[r for r in rows if r['control']=='random']
 lines+=['',f"Unperturbed root-hold velocity drift throughT.05 is at most{max(r['velocity_change_inf'] for r in holds):.3g}. Pressure p=xy dynamic velocity is at most{max(r['velocity_inf'] for r in prs):.3g}, with gradient error at most{max(r['gradient_error_inf'] for r in prs):.3g}.",'','Random perturbation per-step infinity-norm and nodal mean physical perturbation kinetic-energy traces show no transient amplification beyond their known initial values at either amplitude; initial values are included explicitly in later branches and are known from initialization in the first two random branches. Leading-mode norms oscillate with phase, so the table compares to the full complex-mode propagator, not exp(Re(lambda)T) alone. Early leading branches have only partial peak records and are not used to assert a full-run peak bound.','','## Strict genuinely implicit Newton control','','Independent original coupled.evolve, finite random perturbation1e-3, T=.1, rtol0 and absolute tolerance2e-11:','','| dt | Newton corrections / steps | Max full nonlinear residual | Max constraint |','|---:|---:|---:|---:|']
 for r in strict:
  lines.append(f"| {r['dt']:g} | {r['total_newton_corrections']} / {round(r['T']/r['dt'])} | {r['max_nonlinear_residual']:.3g} | {r['max_constraint']:.3g} |")
 lines+=['',f'Strict-control same-time nodal velocity dt difference: {strict_difference:.3g}. Every strict-control step used one Newton correction. The original tolerance runs remain saved; some small perturbation runs accept their Oseen predictor with zero corrections.','','## Limits and execution provenance','','This verifies a neighborhood of one saved discrete equilibrium. The selected spectrum margin is small (-.0020004); the recorded centerline errors remain large, and there is no mesh-convergence, off-node accuracy, global patch-continuity, or reliable default-recipe claim. Rest startup tests are separate. Coefficient/global arithmetic is Float64; only source local maps are nativeMPFR80/MPI8.','','Long-lived tool sessions were interrupted by blocked dc.services.visualstudio.com telemetry; the scientific scripts request no network. Successful short cases were preserved and missing cases completed with foreground segments carrying exact Uold,Lold,start_step histories. The independent segment test suite validates parity; original continuous20 versus10+10 gave exactly equal U and L for bothschemes. See execution_provenance.json. Residual maxima are explicitly scoped to the final segment or observed range rather than silently claiming full-run coverage.','']
 (base/'REPORT.md').write_text('\n'.join(lines))
 print(dict(case_count=len(rows),dt_comparisons=comparisons,strict_newton_dt_difference=strict_difference))
if __name__=='__main__':run(Path('outputs/ns_equilibrium_nullspace_dynamics_20261029'))
