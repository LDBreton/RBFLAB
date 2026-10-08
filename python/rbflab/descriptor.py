"""Small Float64 index-one descriptor spectrum diagnostic.

For M udot + A u=0, left-null(M) gives algebraic constraints C u=0.
Restrict u to null(C), project onto range(M), then solve the reduced pencil.
Rank thresholds make this a numerical diagnostic, not a formal DAE certificate.
"""
import numpy as np
from scipy.linalg import eig,svd,null_space


def descriptor_spectrum(stiffness,mass,rtol=None):
    a,m=np.asarray(stiffness),np.asarray(mass);n=len(m)
    rtol=rtol or 100*n*np.finfo(float).eps
    u,s,vh=svd(m);rank=int(np.sum(s>rtol*s[0])) if s[0]>0 else 0
    info={'mass_rank':rank,'algebraic_modes':n-rank,'rank_relative_tolerance':rtol}
    if rank==0:
        if np.linalg.matrix_rank(a)<n:raise ValueError('Pure algebraic constraints are rank deficient')
        return np.array([],dtype=complex),dict(info,eigenpair_relative_residual_max=None)
    if rank==n:
        values,vectors=eig(-a,m)
    else:
        constraints=u[:,rank:].T@a
        z=null_space(constraints,rcond=rtol)
        if z.shape[1]!=rank:
            raise ValueError('Descriptor constraints are rank deficient; finite spectrum unresolved')
        reduced_m=u[:,:rank].T@m@z;reduced_a=u[:,:rank].T@a@z
        if np.linalg.matrix_rank(reduced_m,tol=rtol*np.linalg.norm(reduced_m,2))<rank:
            raise ValueError('Descriptor has a singular reduced mass; higher-index spectrum unresolved')
        values,w=eig(-reduced_a,reduced_m);vectors=z@w
    residual=[]
    norm_a,norm_m=np.linalg.norm(a,2),np.linalg.norm(m,2)
    for value,vector in zip(values,vectors.T):
        if not np.isfinite(value):continue
        residual.append(float(np.linalg.norm(-a@vector-value*(m@vector))/(norm_a+abs(value)*norm_m)/np.linalg.norm(vector)))
    info['eigenpair_relative_residual_max']=max(residual,default=None)
    return values,info
