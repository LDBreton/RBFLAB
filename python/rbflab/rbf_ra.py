"""Experimental even vector rational approximation for flat-kernel quantities.

Shared-denominator least squares with numerator elimination, following the
mathematical RBF-RA construction of Wright & Fornberg (JCP 331, 2017).
This module does not enable a production LHI backend or certify PDE accuracy.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr, solve_triangular


@dataclass(frozen=True)
class EvenRationalFit:
    radius: float
    numerator: np.ndarray
    denominator: np.ndarray
    output_shape: tuple
    diagnostics: dict

    def __call__(self, epsilon):
        epsilon=complex(epsilon)
        if not np.isfinite(epsilon) or abs(epsilon)>self.radius*(1+1e-14):
            raise ValueError('Evaluation must lie inside the fitted contour')
        z=(epsilon/self.radius)**2
        den=np.polynomial.polynomial.polyval(z,self.denominator)
        scale=np.polynomial.polynomial.polyval(abs(z),abs(self.denominator))
        if abs(den)<1e-12*max(np.finfo(float).tiny,scale):
            raise FloatingPointError('Evaluation is too close to a fitted pole')
        value=np.polynomial.polynomial.polyval(z,self.numerator)/den
        return np.real_if_close(value.reshape(self.output_shape),tol=100)


def fit_even_rational(function, *, radius, samples=64, denominator_degree=12, normalization="constant"):
    """Fit a real-coefficient even vector function using complex128 samples.

    ``samples`` is the real equation count: samples/2 first-quadrant solves.
    Numerator has samples-denominator_degree coefficients in (epsilon/radius)^2.
    Matrix outputs are flattened jointly; all entries share one denominator.
    Fit residual is a diagnostic, NOT a bound on continuation error.
    """
    if not np.isfinite(radius) or radius<=0:raise ValueError('Positive finite radius required')
    if type(samples) is not int or samples<8 or samples%2:raise ValueError('samples must be even and >=8')
    if type(denominator_degree) is not int or not 1<=denominator_degree<samples//2:
        raise ValueError('Invalid denominator degree')
    if normalization not in ('constant','svd'):raise ValueError('Unknown denominator normalization')
    angles=(np.arange(samples//2)+.5)*np.pi/samples
    contour=radius*np.exp(1j*angles)
    values=[np.asarray(function(e),dtype=np.complex128) for e in contour]
    shape=values[0].shape
    if any(v.shape!=shape or not np.isfinite(v).all() for v in values):
        raise ValueError('Inconsistent or nonfinite contour samples')
    data=np.stack([v.ravel() for v in values]);row_scale=np.max(abs(data),axis=1)
    if np.any(row_scale==0):raise ValueError('Zero contour sample vector')
    z=(contour/radius)**2;n=denominator_degree;m=samples-n
    V=np.vander(z,N=m,increasing=True)/row_scale[:,None]
    B=-data[:,:,None]*(z[:,None,None]**np.arange(1,n+1)[None,None,:])/row_scale[:,None,None]
    rhs=data/row_scale[:,None]
    real=lambda v:np.concatenate((v.real,v.imag),axis=0)
    Q,R=qr(real(V),mode='full');b=np.einsum('ki,imj->kmj',Q.T,real(B));g=Q.T@real(rhs)
    tail=b[m:].reshape(-1,n);target=g[m:].reshape(-1)
    if normalization=='svd':
        # Homogeneous variant avoids imposing q(0)=1 when coefficients span many orders.
        C=np.concatenate((-target[:,None],tail),axis=1)
        _,singular,Vh=np.linalg.svd(C,full_matrices=False)
        coefficients=Vh[-1];denominator=coefficients[1:];constant=coefficients[0]
        rank=n
        target=target*constant
    else:
        constant=1.
        denominator,_,rank,singular=np.linalg.lstsq(tail,target,rcond=None)
    if rank<n:raise np.linalg.LinAlgError('Shared denominator fit is rank deficient')
    numerator=solve_triangular(R[:m,:],constant*g[:m]-np.einsum('imj,j->im',b[:m],denominator))
    residual=np.linalg.norm(tail@denominator-target)/max(np.linalg.norm(target),np.finfo(float).tiny)
    return EvenRationalFit(float(radius),numerator,np.r_[constant,denominator],shape,
        dict(normalization=normalization,samples=samples,complex_solves=len(contour),denominator_degree=n,denominator_rank=int(rank),
             projected_relative_residual=float(residual),denominator_fit_condition=float(singular[0]/singular[-1])))
