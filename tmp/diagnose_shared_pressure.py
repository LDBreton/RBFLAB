import json
import numpy as np
from scipy.linalg import svd, orth
from scipy.sparse import save_npz
from examples.stokes_pressure_samples import build
from research.shared_pressure_lhi.solver import SharedPressureStokes
maps,nodes,boundary,_=build(pressure_degree=2,backend='cpp',ranks=2,all_targets=True)
s=SharedPressureStokes(maps,nodes)
rows=[]
for dt in [None, .00025]:
 K=s.matrix(dt); save_npz('outputs/shared_pressure_closure_diagnosis_20261030/matrix_'+str(dt)+'.npz',K)
 norms=np.asarray(abs(K).sum(axis=1)).ravel(); D=K.toarray()/norms[:,None]
 u,sv,vh=svd(D); n=len(nodes); null=vh[sv<sv[0]*len(sv)*np.finfo(float).eps].T
 pn=null[2*n:]; un=null[:2*n]
 basis=np.column_stack([nodes[:,0]**a*nodes[:,1]**b for a in range(5) for b in range(5-a)])
 pressure_only=D[:,2*n:]; sp=np.linalg.svd(pressure_only,compute_uv=False)
 row=dict(dt=dt,singular_values=sv.tolist(),velocity_null_singular=np.linalg.svd(un,compute_uv=False).tolist(),pressure_null_singular=np.linalg.svd(pn,compute_uv=False).tolist(),pressure_only_singular=sp.tolist(),pressure_poly4_projection_residual=float(np.linalg.norm(pn-orth(basis)@(orth(basis).T@pn))),pressure_degree_residuals={})
 for degree in range(1,6):
  V=np.column_stack([nodes[:,0]**a*nodes[:,1]**b for a in range(degree+1) for b in range(degree+1-a)])
  row['pressure_degree_residuals'][degree]=float(np.linalg.norm(pn-orth(V)@(orth(V).T@pn)))
 rows.append(row)
np.savez('outputs/shared_pressure_closure_diagnosis_20261030/geometry.npz',nodes=nodes,boundary=boundary)
open('outputs/shared_pressure_closure_diagnosis_20261030/diagnostic.json','w').write(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
