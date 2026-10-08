"""Experimental complex IMQ Stokes weight path for RBF-RA local pilots only."""
import numpy as np
from scipy.linalg import solve
from .kernels import ScalarKernel,_derivative
from .stokes import _blocks
from .operators import Laplacian,Derivative


class _ComplexArithmetic:
    ctx=None
    def __init__(self,c):self.c=c
    def zeros(self,n,m):return np.zeros((n,m),dtype=np.complex128)
    def matrix(self,x,left,y,right,length_scale=None):
        if length_scale is not None:raise NotImplementedError('Physical coordinates only')
        z=np.asarray(x)[:,None,:]-np.asarray(y)[None,:,:];out=self.zeros(len(x),len(y));cache={}
        lg={};rg={}
        for i,op in enumerate(left):lg.setdefault(op,[]).append(i)
        for j,op in enumerate(right):rg.setdefault(op,[]).append(j)
        for l,ii in lg.items():
            for r,jj in rg.items():
                for alpha,ca in l.terms:
                    for beta,cb in r.terms:
                        order=tuple(a+b for a,b in zip(alpha,beta))
                        if order not in cache:
                            val=_derivative('imq',*order)(z[...,0],z[...,1],self.c)
                            cache[order]=np.broadcast_to(val,z.shape[:-1])
                        out[np.ix_(ii,jj)]+=float(ca)*float(cb)*(-1)**sum(beta)*cache[order][np.ix_(ii,jj)]
        return out


class IMQStokesWeightPath:
    """W(epsilon), with c_u=c_u_target*epsilon^2, c_p=c_p_target*epsilon^2.

    Target epsilon=1 recovers the requested kernels; the fixed ratio is preserved.
    Complex128 solves use transpose, never conjugate transpose. No MP fallback.
    """
    def __init__(self,task,*,condition_limit=1e12):
        if task['degree'] is not None or task['scale'] is not None:
            raise NotImplementedError('Unaugmented physical-coordinate stencils only')
        if any(not isinstance(task[k],ScalarKernel) or task[k].family!='imq' for k in ('vk','pk')):
            raise NotImplementedError('IMQ velocity and pressure required')
        if np.asarray(task['points']).shape[1]!=2:raise NotImplementedError('2D Stokes only')
        self.task=task;self.condition_limit=float(condition_limit);self.diagnostics=[]
        if self.condition_limit<=1 or np.isnan(self.condition_limit):raise ValueError('Invalid condition limit')
        self.cu=float(task['vk'].c);self.cp=float(task['pk'].c)
        p=np.vstack([task['points'],task['origin']]);diameter=np.max(np.linalg.norm(p[:,None]-p[None,:],axis=-1))
        self.branch_radius=1/(np.sqrt(max(self.cu,self.cp))*diameter)

    def matrices(self,epsilon):
        if not np.isfinite(epsilon) or epsilon==0 or abs(epsilon)>=self.branch_radius:
            raise ValueError('Nonzero epsilon must remain inside the nearest IMQ branch singularity')
        a=_ComplexArithmetic(self.cu*complex(epsilon)**2);a.pressure_arithmetic=_ComplexArithmetic(self.cp*complex(epsilon)**2)
        task=self.task;mu=float(task['mu']);targets=[{j:-Laplacian()*mu,2:Derivative(j)} for j in (0,1)]+[{2:Derivative(j)} for j in (0,1)]
        G=_blocks(a,task['points'],task['ops'],task['points'],task['ops'])
        B=_blocks(a,np.tile(task['origin'],(4,1)),targets,task['points'],task['ops']).T
        return G,B

    def __call__(self,epsilon):
        G,B=self.matrices(epsilon);D=1/np.sqrt(np.max(abs(G),axis=1));E=D[:,None]*G*D[None,:]
        condition=float(np.linalg.cond(E))
        if not np.isfinite(condition) or condition>self.condition_limit:
            raise np.linalg.LinAlgError(f'Contour condition {condition:.3e} exceeds limit {self.condition_limit:.3e}')
        W=D[:,None]*solve(E.T,D[:,None]*B,assume_a='gen')
        residual=np.linalg.norm(G.T@W-B,np.inf)/max(np.linalg.norm(B,np.inf),np.finfo(float).tiny)
        self.diagnostics.append(dict(epsilon=[float(np.real(epsilon)),float(np.imag(epsilon))],condition=condition,residual=float(residual)))
        return W


def approximate_stokes_weights(task, *, radius, target=1., samples=64,
                               denominator_degree=12, condition_limit=1e12,
                               tolerance=1e-6):
    """Guarded experimental continuation; raises if local checks fail.

    Target=1 retains the task's kernels. Radius is in the path epsilon coordinate.
    Checks contour conditioning, sample-count agreement and independent contour
    values. Passing these checks is not a proof of accuracy at the flat target;
    MP validation is still required before production use.
    """
    from .rbf_ra import fit_even_rational
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive finite tolerance required')
    if not np.isfinite(target) or target<=0 or target>=radius:
        raise ValueError('Positive target must lie strictly inside the contour')
    path=IMQStokesWeightPath(task,condition_limit=condition_limit)
    if not 0<radius<path.branch_radius:raise ValueError('Contour crosses an IMQ branch singularity')
    coarse=fit_even_rational(path,radius=radius,samples=samples,denominator_degree=denominator_degree)
    fine=fit_even_rational(path,radius=radius,samples=2*samples,denominator_degree=denominator_degree)
    first=coarse(target);weights=fine(target)
    error=np.max(abs(first-weights))/max(np.max(abs(weights)),np.finfo(float).tiny)
    checks=[]
    for angle in (.19,.73,1.31):
        epsilon=radius*np.exp(1j*angle);reference=path(epsilon)
        checks.append(float(np.max(abs(fine(epsilon)-reference))/max(np.max(abs(reference)),np.finfo(float).tiny)))
    if error>tolerance or max(checks)>tolerance:
        raise np.linalg.LinAlgError(f'RBF-RA validation failed: sample agreement {error:.3e}, held-out contour error {max(checks):.3e}')
    if np.max(abs(np.imag(weights)))>tolerance*max(np.max(abs(weights)),np.finfo(float).tiny):
        raise np.linalg.LinAlgError('RBF-RA returned non-real physical weights')
    return np.asarray(weights.real,dtype=np.float64),dict(sample_agreement=float(error),
        held_out_contour_error=max(checks),max_contour_condition=max(d['condition'] for d in path.diagnostics),fit=fine.diagnostics)
