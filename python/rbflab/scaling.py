"""Kernel coordinate scaling with physical differential functionals."""
import numpy as np


def scaled_matrix(arithmetic,x,left,y,right,h):
    a=arithmetic
    out=a.zeros(len(x),len(y))
    if a.ctx:
        for i,point in enumerate(x):
            for j,center in enumerate(y):
                z=[(a.number(v)-a.number(w))/h for v,w in zip(point,center)]
                terms=[]
                for alpha,ca in left[i].terms:
                    for beta,cb in right[j].terms:
                        order=tuple(v+w for v,w in zip(alpha,beta))
                        terms.append(a.number(ca)*a.number(cb)*(-1)**sum(beta)
                            *a.backend.derivative(*z,order)/h**sum(order))
                out[i,j]=a.ctx.fsum(terms)
    else:
        z=(np.asarray(x)[:,None,:]-np.asarray(y)[None,:,:])/h
        cache={}
        for i,op in enumerate(left):
            for j,source in enumerate(right):
                for alpha,ca in op.terms:
                    for beta,cb in source.terms:
                        order=tuple(v+w for v,w in zip(alpha,beta))
                        if order not in cache:
                            cache[order]=a.kernel.derivative(z,order)/h**sum(order)
                        out[i,j]+=float(ca)*float(cb)*(-1)**sum(beta)*cache[order][i,j]
    return out
