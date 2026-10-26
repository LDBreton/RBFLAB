"""Overlapping explicit patches for scalar and independent vector data."""
import numpy as np
import rbflab as r


x=np.array(np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5),indexing='ij')).reshape(2,-1).T
cover=r.PatchCover([[-0.5,0],[0.5,0]],[1.7,1.7])
data=np.column_stack((np.sin(x[:,0])+x[:,1]**2,np.cos(x[:,1])-x[:,0]))
fit=r.interpolate(space=r.ScalarSpace(r.Gaussian(1)),centers=x,data=data,patches=cover)
q=np.array([[-0.1,0.2],[0.2,0.3]])
print('values',fit.evaluate(q))
print('x derivatives',fit.evaluate(q,r.Derivative(0)))
print('patch occupancy',fit.diagnostics['occupancy'])
