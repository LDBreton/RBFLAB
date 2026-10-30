"""Isolated Gaussian pressure-point variational LHI benchmark (Float64).

Stationary Hermite representers, exact pressure constant, BE/BDF2,
local QR elimination of PDE data, sparse global normal equations.
No repository solver or existing result is modified.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, qr
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.special import eval_hermite
from scipy.spatial import cKDTree

MU = .1
# Functional terms: (field component, x derivative, y derivative, coefficient).
OPS = [[(0,0,0,1)], [(1,0,0,1)], [(2,0,0,1)],
       [(0,2,0,-MU),(0,0,2,-MU),(2,1,0,1)],
       [(1,2,0,-MU),(1,0,2,-MU),(2,0,1,1)]]
BASE = {(0,0):(0,2,-1), (0,1):(1,1,1),
        (1,0):(1,1,1), (1,1):(2,0,-1), (2,2):(0,0,1)}

def gram(x, tx, y, ty, eps):
    delta = x[:,None,:]-y[None,:,:]
    g = np.exp(-eps**2*np.sum(delta**2,axis=2))
    derivatives = {}
    def deriv(a,b):
        if (a,b) not in derivatives:
            derivatives[a,b] = (-eps)**(a+b)*eval_hermite(a,eps*delta[:,:,0])*eval_hermite(b,eps*delta[:,:,1])*g
        return derivatives[a,b]
    out = np.zeros_like(g)
    for i in np.unique(tx):
        for j in np.unique(ty):
            mask = (tx[:,None]==i)&(ty[None,:]==j)
            val = np.zeros_like(g)
            for f,a,b,c in OPS[i]:
                for h,d,e,v in OPS[j]:
                    if (f,h) in BASE:
                        aa,bb,cc = BASE[f,h]
                        val += c*v*cc*(-1)**(d+e)*deriv(a+d+aa,b+e+bb)
            out[mask] = val[mask]
    return out

def exact(x,t):
    a,b = np.pi*x.T
    u = np.column_stack((np.sin(a)*np.cos(b),-np.cos(a)*np.sin(b)))*np.exp(-t)
    p = np.sin(a)*np.sin(b)*np.exp(-t)
    gp = np.pi*np.column_stack((np.cos(a)*np.sin(b),np.sin(a)*np.cos(b)))*np.exp(-t)
    return u,p,(2*np.pi**2*MU-1)*u+gp

class Patch:
    def __init__(self, k, points, inside, boundary, stencil, eps):
        self.k=k
        idx=np.argsort(np.linalg.norm(points-inside[k],axis=1),kind='stable')[:stencil]
        ip=cKDTree(inside).query(points[idx])[0]<1e-12
        vi=cKDTree(inside).query(points[idx[ip]])[1]
        bi=cKDTree(boundary).query(points[idx[~ip]])[1]
        pi=vi.copy(); qi=vi[vi!=k]
        self.vi,self.pi,self.bi,self.qi=vi,pi,bi,qi
        self.qpoints=inside[qi]
        pts=[]; typ=[]; glob=[]
        nv=len(inside)
        for ids, cloud, types, offset in [(vi,inside,[0,1],0),(pi,inside,[2],2*nv),(bi,boundary,[0,1],-1),(qi,inside,[3,4],-2)]:
            for i in ids:
                for t in types:
                    pts.append(cloud[i]);typ.append(t)
                    glob.append((2*i+t if t<2 else offset+i) if offset>=0 else offset)
        self.x=np.array(pts);self.typ=np.array(typ);self.glob=np.array(glob)
        self.ng=np.sum(self.glob>=0); self.nb=2*len(bi);self.nq=2*len(qi)
        self.eps=eps
        A=gram(self.x,self.typ,self.x,self.typ,eps)
        scale=1/np.sqrt(np.diag(A)); As=scale[:,None]*A*scale[None,:]
        L=cholesky(As,lower=True)
        self.L,self.scale=L,scale
        self.local_condition=float(np.linalg.cond(As))
        inv=scale[:,None]*cho_solve((L,True),np.diag(scale))
        q=(self.typ==2).astype(float);v=inv@q; den=q@v
        self.C=inv-np.outer(v,v)/den
        self.c=v/den
        # Whitened projection: stable native energy without forming A^{-1}.
        J=solve_triangular(L,np.diag(scale),lower=True)
        b=J@q
        self.E=J-np.outer(b,b@J)/(b@b)
        self.q,self.v,self.den=q,v,den
        self.V=self.weights(np.repeat(self.qpoints,2,axis=0),np.tile([0,1],len(qi)))
        self.U=self.weights(np.repeat(inside[[k]],2,axis=0),np.array([0,1]))
        self.M=self.weights(np.repeat(inside[[k]],2,axis=0),np.array([3,4]))
        self.constant_defect=float(max(np.max(abs(self.E@q)),np.max(abs(self.V@q))))
        self.interpolation_defect=float(np.max(abs(self.weights(self.x,self.typ)-np.eye(len(self.x)))))

    def weights(self,x,typ):
        H=gram(x,typ,self.x,self.typ,self.eps)
        W=cho_solve((self.L,True),(H*self.scale).T).T*self.scale
        return W+np.outer((typ==2)-W@self.q,self.v/self.den)

    def prepare(self,a,lam):
        nq=self.nq
        residual=a*self.V.copy()
        residual[:,-nq:]+=np.eye(nq)
        # Equal total weight for the auxiliary and target momentum residuals.
        self.F=np.vstack((residual/np.sqrt(nq), (self.M+a*self.U)/np.sqrt(2),np.sqrt(lam)*self.E))
        fq=self.F[:,-nq:]
        orth,_=qr(fq,mode='full')
        self.N=orth[:,nq:].T
        self.reduced=self.N@self.F[:,:self.ng]
        self.bound=self.F[:,self.ng:self.ng+self.nb]
        self.qfactor=qr(fq,mode='economic')

    def rhs(self,t,a,history):
        hq=exact(self.qpoints,t)[2].ravel()+history[0]
        ht=exact(self.x[[np.flatnonzero((self.glob==2*self.k))[0]]],t)[2].ravel()+history[1]
        g=exact(self.x[self.ng:self.ng+self.nb:2],t)[0].ravel()
        rhs=np.r_[hq/np.sqrt(self.nq),ht/np.sqrt(2),np.zeros(len(self.x))]-self.bound@g
        return rhs,g

def run(n,eps,lam,dt,T,stencil):
    grid=np.linspace(0,1,n);points=np.array([(x,y) for y in grid for x in grid])
    mask=np.all((points>0)&(points<1),axis=1);inside=points[mask];boundary=points[~mask]
    nv=len(inside); patches=[Patch(k,points,inside,boundary,min(stencil,len(points)),eps) for k in range(nv)]
    gauge=2*nv; size=3*nv-1
    prev=[(exact(p.qpoints,0)[0].ravel(),exact(inside[[p.k]],0)[0].ravel()) for p in patches]
    older=None; steps=round(T/dt);assert abs(steps*dt-T)<1e-12
    metrics={}; last=[]; maxres=0
    for step in range(1,steps+1):
        a=(1 if step==1 else 1.5)/dt;t=step*dt
        if step<=2:
            rows=[];cols=[];vals=[]; row0=0
            for p in patches:
                p.prepare(a,lam)
                ids=p.glob[:p.ng]; keep=ids!=gauge; ids=ids[keep]-(ids[keep]>gauge)
                block=p.reduced[:,keep];rr,cc=np.indices(block.shape)
                rows.extend((rr+row0).ravel());cols.extend(ids[cc].ravel());vals.extend(block.ravel());row0+=len(block)
            B=coo_matrix((vals,(rows,cols)),shape=(row0,size)).tocsr()
            K=(B.T@B).tocsc(); diagonal=1/np.sqrt(K.diagonal())
            Ks=K.multiply(diagonal[:,None]).multiply(diagonal[None,:]).tocsc();factor=splu(Ks)
            ev=np.linalg.eigvalsh(Ks.toarray())
            metrics.update(min_eigenvalue=float(ev[0]),condition=float(ev[-1]/ev[0]),unknowns=size,nnz=K.nnz)
            dense=Ks.toarray();vv=dense[:2*nv,:2*nv];vp=dense[:2*nv,2*nv:]
            schur=dense[2*nv:,2*nv:]-vp.T@np.linalg.solve(vv,vp)
            metrics['pressure_schur_min_eigenvalue']=float(np.linalg.eigvalsh(schur)[0])
        rhs=[]; local=[]
        for k,p in enumerate(patches):
            hist=tuple(v/dt for v in prev[k]) if step==1 else tuple((2*v-.5*w)/dt for v,w in zip(prev[k],older[k]))
            r,g=p.rhs(t,a,hist);rhs.extend(p.N@r);local.append((r,g,hist))
        rhs=np.array(rhs);load=B.T@rhs;z=diagonal*factor.solve(diagonal*load)
        maxres=max(maxres,float(np.linalg.norm(K@z-load)/max(np.linalg.norm(load),1e-30)))
        full=np.insert(z,gauge,0.);new=[];last=[];pder=[]
        for p,(r,g,hist) in zip(patches,local):
            zg=full[p.glob[:p.ng]];qmat,rmat=p.qfactor
            q=solve_triangular(rmat,qmat.T@(r-p.F[:,:p.ng]@zg))
            d=np.r_[zg,g,q];last.append(d);new.append((p.V@d,p.U@d))
            pder.extend(q+a*p.V@d-exact(p.qpoints,t)[2].ravel()-hist[0])
            pder.extend((p.M+a*p.U)@d-exact(inside[[p.k]],t)[2].ravel()-hist[1])
        older,prev=prev,new
    ue,pe,_=exact(inside,T);pe-=pe[0]
    un=full[:2*nv].reshape(-1,2);pn=full[2*nv:]
    def rms(v):return float(np.sqrt(np.mean(np.asarray(v)**2)))
    metrics.update(n=n,epsilon=eps,lam=lam,dt=dt,T=T,stencil=stencil,velocity_nodal_rms=rms(un-ue),pressure_nodal_rms=rms(pn-pe),velocity_nodal_relative=rms(un-ue)/rms(ue),pressure_nodal_relative=rms(pn-pe)/rms(pe),linear_residual=maxres,local_condition_max=max(p.local_condition for p in patches),constant_defect_max=max(p.constant_defect for p in patches))
    check=np.random.default_rng(20261030).uniform(.03,.97,(200,2)); owner=cKDTree(inside).query(check)[1]
    pred=np.zeros((len(check),3))
    for k,p in enumerate(patches):
        ids=np.flatnonzero(owner==k)
        if len(ids):pred[ids]=(p.weights(np.repeat(check[ids],3,axis=0),np.tile([0,1,2],len(ids)))@last[k]).reshape(-1,3)
    eu,ep,_=exact(check,T);ep-=exact(inside[[0]],T)[1][0]
    metrics.update(velocity_offnode_rms=rms(pred[:,:2]-eu),pressure_offnode_rms=rms(pred[:,2]-ep))
    metrics.update(momentum_residual_rms=rms(pder),interpolation_defect_max=max(p.interpolation_defect for p in patches))
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--n',type=int,nargs='+',default=[5,7,9]);ap.add_argument('--eps',type=float,nargs='+',default=[2.,4.]);ap.add_argument('--lam',type=float,nargs='+',default=[1e-6,1e-3,1.]);ap.add_argument('--dt',type=float,default=.02);ap.add_argument('--T',type=float,default=.1);ap.add_argument('--stencil',type=int,default=17);args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False);results=[]
    for n in args.n:
        for eps in args.eps:
            for lam in args.lam:
                start=time.time()
                try:result=run(n,eps,lam,args.dt,args.T,args.stencil)
                except Exception as exc:result=dict(n=n,epsilon=eps,lam=lam,error=repr(exc))
                result['seconds']=time.time()-start;results.append(result);print(json.dumps(result),flush=True)
                (dest/'results.json').write_text(json.dumps(results,indent=2))
