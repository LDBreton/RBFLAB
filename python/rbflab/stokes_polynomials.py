"""Normalized divergence-free velocity polynomials and pressure modulo constants."""
from math import factorial
import numpy as np
from .stencils import polynomial_powers


class StokesPolynomialBasis:
    def __init__(self,arithmetic,origin,scale,degree):
        self.a=arithmetic;self.origin=[arithmetic.number(v) for v in origin]
        self.scale=arithmetic.number(scale);self.degree=degree
        self.columns=[]
        # Curl of normalized stream monomials; no constant stream function.
        for x,y in polynomial_powers(2,degree+1)[1:]:
            terms=[]
            if y:terms.append((0,y,(x,y-1)))
            if x:terms.append((1,-x,(x-1,y)))
            self.columns.append(terms)
        # Constant pressure is invisible to velocity/momentum data and excluded.
        for power in polynomial_powers(2,max(1,degree-1))[1:]:
            self.columns.append([(2,1,power)])

    def matrix(self,points,functionals):
        a=self.a;out=a.zeros(len(points),len(self.columns))
        for i,(point,op) in enumerate(zip(points,functionals)):
            z=[(a.number(v)-o)/self.scale for v,o in zip(point,self.origin)]
            for j,terms in enumerate(self.columns):
                for component,coefficient,power in terms:
                    if component not in op:continue
                    for alpha,value in op[component].terms:
                        if any(k>p for k,p in zip(alpha,power)):continue
                        v=a.number(coefficient)*a.number(value)/self.scale**sum(alpha)
                        for coord,p,k in zip(z,power,alpha):
                            v*=factorial(p)//factorial(p-k)*coord**(p-k)
                        out[i,j]+=v
        return out


def append_columns(a,left,right):
    out=a.zeros(left.rows if a.ctx else left.shape[0],
                (left.cols+right.cols) if a.ctx else left.shape[1]+right.shape[1])
    n=left.cols if a.ctx else left.shape[1]
    out[:,:n]=left;out[:,n:]=right
    return out


def rank_ratio(matrix):
    """Float64 geometric/functional rank screen, also for MP assembly."""
    p=np.asarray(matrix.tolist(),dtype=float)
    if p.shape[0]<p.shape[1]:return 0.
    norms=np.linalg.norm(p,axis=0)
    if np.any(norms==0):return 0.
    values=np.linalg.svd(p/norms,compute_uv=False)
    return float(values[-1]/values[0])
