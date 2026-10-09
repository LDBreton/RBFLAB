"""Shared geometric neighborhoods; geometry remains Float64."""
import numpy as np
from scipy.spatial import cKDTree


def neighbors(tree, point, size):
    if type(size) is not int or not 1 <= size <= tree.n:
        raise ValueError("stencil_size must be between 1 and the point count")
    distance, ids = tree.query(point, k=size)
    distance, ids = np.atleast_1d(distance), np.atleast_1d(ids)
    return ids[np.lexsort((ids, distance))]


from dataclasses import dataclass
from itertools import product


def polynomial_powers(dimension, degree):
    """Total-degree exponents, independent of spatial dimension."""
    return [a for total in range(degree+1)
            for a in product(range(total+1), repeat=dimension) if sum(a)==total]


def geometry_quality(points, target, degree):
    points=np.asarray(points,dtype=float)
    displacement=points-np.asarray(target,dtype=float)
    radius=float(np.max(np.linalg.norm(displacement,axis=1)))
    if radius==0:
        return {"radius":0.,"separation_ratio":0.,"polynomial_ratio":0.}
    normalized=displacement/radius
    separation=float(np.min(cKDTree(normalized).query(normalized,k=2)[0][:,1]))
    powers=polynomial_powers(points.shape[1],degree)
    v=np.array([np.prod(normalized**a,axis=1) for a in powers]).T
    scales=np.linalg.norm(v,axis=0)
    v=v/np.where(scales>0,scales,1)
    singular=np.linalg.svd(v,compute_uv=False)
    ratio=float(singular[-1]/singular[0]) if len(points)>=len(powers) else 0.
    return {"radius":radius,"separation_ratio":separation,"polynomial_ratio":ratio}


@dataclass(frozen=True)
class StencilPolicy:
    """Choose neighborhood selection and kernel distance scaling.

    Args:
        scaling: ``"physical"`` (default) evaluates the kernel at physical
            displacements. ``"local"`` evaluates K((x-y)/h), where h is the
            maximum Euclidean distance from the first source center (one if
            zero). Physical derivative factors h**(-order) are included
            automatically; do not rescale returned weights. Shape parameters
            then refer to dimensionless distances, so switching policy with
            the same numeric parameter can change the approximation.
        selection: ``"nearest"`` keeps the requested stencil size.
            ``"quality"`` grows a nearest-neighbor set until polynomial-rank
            and node-separation thresholds are met, or raises an error.
        max_size: Largest stencil considered by quality selection. If None,
            use min(number of source points, 2 * stencil_size).
        min_polynomial_ratio: Minimum singular-value ratio of the normalized,
            column-scaled polynomial matrix in quality selection.
        min_separation_ratio: Minimum node separation divided by the radius
            measured from the target, used only in quality selection.

    Local scaling alone does not enable quality selection, move nodes, or
    change precision. Polynomials are normalized under both scaling policies.
    These options do not certify conditioning, accuracy, or PDE stability.
    """
    scaling: str = "physical"
    selection: str = "nearest"
    max_size: int | None = None
    min_polynomial_ratio: float = 1e-5
    min_separation_ratio: float = 1e-6

    def __post_init__(self):
        if self.scaling not in ("physical","local"):
            raise ValueError("scaling must be physical or local")
        if self.selection not in ("nearest","quality"):
            raise ValueError("selection must be nearest or quality")
        if self.max_size is not None and (type(self.max_size) is not int or self.max_size<1):
            raise ValueError("max_size must be a positive integer")
        for value in (self.min_polynomial_ratio,self.min_separation_ratio):
            if not np.isfinite(value) or not 0<value<=1:
                raise ValueError("Quality thresholds must be in (0, 1]")

    def select(self,tree,target,size,degree=None):
        selected=neighbors(tree,target,size)
        if self.selection=="nearest":
            return selected
        limit=self.max_size if self.max_size is not None else min(tree.n,2*size)
        if not size<=limit<=tree.n:
            raise ValueError("max_size must lie between stencil_size and point count")
        # Deterministic membership at distance ties in the opt-in path.
        candidates=np.asarray(tree.query_ball_point(target,
            np.nextafter(float(tree.query(target,k=limit)[0][-1]) if limit>1 else
                         float(tree.query(target,k=1)[0]),np.inf)),dtype=int)
        distances=np.linalg.norm(tree.data[candidates]-target,axis=1)
        candidates=candidates[np.lexsort((candidates,distances))][:limit]
        last=None
        for count in range(size,limit+1):
            selected=candidates[:count]
            last=geometry_quality(tree.data[selected],target,degree if degree is not None else 1)
            if (last["polynomial_ratio"]>=self.min_polynomial_ratio and
                last["separation_ratio"]>=self.min_separation_ratio):
                return selected
        raise np.linalg.LinAlgError(f"Stencil quality failed within max_size={limit}: {last}")
