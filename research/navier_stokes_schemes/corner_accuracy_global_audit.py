"""Independent nodal FD/weak-cell incompressibility audit; no RBF derivatives."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.sparse import lil_matrix
from threadpoolctl import threadpool_limits
from .corner_accuracy_continuation import transformed,SOURCE
from .pressure_lifting import load_problem


def derivative(cells,order):
    n=cells+1;D=np.zeros((n,n));width=order+1
    for i in range(n):
        start=min(max(i-width//2,0),n-width)
        ids=np.arange(start,start+width);offset=ids-i
        powers=np.array([offset.astype(float)**k for k in range(width)])
        rhs=np.zeros(width);rhs[1]=cells
        D[i,ids]=np.linalg.solve(powers,rhs)
    return D


def grid(p,U):
    V=np.zeros((p.cells+1,p.cells+1,2))
    for points,v in ((p.points,U.reshape(-1,2)),(p.boundary_points,p.g0.reshape(-1,2))):
        ij=np.rint(points*p.cells).astype(int);V[ij[:,0],ij[:,1]]=v
    return V


def fd(V,D):
    return D@V[:,:,0]+V[:,:,1]@D.T


def nodal_divergence_operator(p,order=4,exclude_layers=0):
    """Return independent sparse (D,b) with divergence = D@physical_U+b.

    Uses existing physical boundary g0 and existing interior sample locations.
    exclude_layers=2 retains indices 3..cells-3, for example. No RBF maps,
    affine reconstruction data or Ghia data are consulted.
    """
    cells=p.cells;one=derivative(cells,order)
    interior={tuple(ij):i for i,ij in enumerate(np.rint(p.points*cells).astype(int))}
    boundary={tuple(ij):i for i,ij in enumerate(np.rint(p.boundary_points*cells).astype(int))}
    targets=[ij for ij in interior if min(*ij,cells-ij[0],cells-ij[1])>exclude_layers]
    matrix=lil_matrix((len(targets),len(p.U0)));b=np.zeros(len(targets));g=p.g0.reshape(-1,2)
    for row,(i,j) in enumerate(targets):
        for component,siteindex in ((0,i),(1,j)):
            for k in np.flatnonzero(one[siteindex]):
                site=(k,j) if component==0 else (i,k);weight=one[siteindex,k]
                if site in interior:matrix[row,2*interior[site]+component]+=weight
                else:b[row]+=weight*g[boundary[site],component]
    return matrix.tocsr(),b


def cell_flux(V,h):
    # Exact perimeter flux of the bilinear interpolant of these four nodes.
    return h*.5*(V[1:,:-1,0]+V[1:,1:,0]-V[:-1,:-1,0]-V[:-1,1:,0]
                 +V[:-1,1:,1]+V[1:,1:,1]-V[:-1,:-1,1]-V[1:,:-1,1])


def stats(x):
    return dict(rms=float(np.sqrt(np.mean(x*x))),maximum=float(np.max(abs(x))),mean=float(np.mean(x)))


def run():
    out=Path('outputs/ns_corner_accuracy_global_audit_20261029');out.mkdir(parents=True,exist_ok=False)
    p=load_problem(12,SOURCE/'maps');h=1/p.cells;n=len(p.U0)
    D2,D4=derivative(12,2),derivative(12,4)
    # Homogeneous-boundary nodal operators have independently assessed rank.
    basegrid=grid(p,np.zeros(n));basefd=fd(basegrid,D4)[1:-1,1:-1].ravel()
    basecell=cell_flux(basegrid,h).ravel()/h**2
    fdmap=np.zeros((121,n));cellmap=np.zeros((144,n))
    for k in range(n):
        e=np.zeros(n);e[k]=1;v=grid(p,e)
        fdmap[:,k]=fd(v,D4)[1:-1,1:-1].ravel()-basefd
        cellmap[:,k]=cell_flux(v,h).ravel()/h**2-basecell
    left=null_space(p.M.toarray().T,rcond=1e-10)
    constraint=left.T@p.A.toarray()
    result=dict(configuration=dict(cells=12,total_nodes=169,interior_nodes=121,nu=.01,Re=100,scope='existing nodal velocities only; FD and bilinear weak-cell diagnostics'),
        rank=dict(momentum_mass_rank=int(np.linalg.matrix_rank(p.M.toarray(),tol=1e-10)),algebraic_velocity_constraints=int(np.linalg.matrix_rank(constraint,tol=1e-10)),fd4_nodal_divergence=int(np.linalg.matrix_rank(fdmap,tol=1e-10)),bilinear_cell_flux=int(np.linalg.matrix_rank(cellmap,tol=1e-10))),fields={})
    cases=[('original',SOURCE/'steady.npz',0.,.25),
           ('corner_a005_unstable',Path('outputs/ns_corner_accuracy_r025_20261029/root_a0.050.npz'),.05,.25),
           ('corner_r05_a005_unstable',Path('outputs/ns_corner_accuracy_r050_20261029/root_a0.050.npz'),.05,.5)]
    fields={}
    for name,path,a,radius in cases:
        z=np.load(path);U,L=z['U'],z['L'];V=grid(p,U);q,_=transformed(p,a,radius);g=q.boundary(0)
        dx=q.DxU@U+q.DxB@g+q.DxL@L;dy=q.DyU@U+q.DyB@g+q.DyL@L
        flux=cell_flux(V,h);row=dict(source=str(path),analytic_local_DF=stats(dx[::2]+dy[1::2]),fd2_all_interior=stats(fd(V,D2)[1:-1,1:-1]),fd4_all_interior=stats(fd(V,D4)[1:-1,1:-1]),fd2_inner7by7=stats(fd(V,D2)[3:-3,3:-3]),fd4_inner7by7=stats(fd(V,D4)[3:-3,3:-3]),bilinear_cell_divergence=stats(flux/h**2),absolute_cell_flux_sum=float(np.sum(abs(flux))),global_wall_flux=float(np.sum(flux)),constraint_inf=float(np.max(abs(q.A@U-q.M@L+q.B@g))))
        row['macrocell_divergence']={}
        for stride in (2,3,4,6,12):
            agg=flux.reshape(12//stride,stride,12//stride,stride).sum(axis=(1,3))
            row['macrocell_divergence'][str(stride)]=stats(agg/(stride*h)**2)
        result['fields'][name]=row
        fields[name+'_fd2']=fd(V,D2);fields[name+'_fd4']=fd(V,D4);fields[name+'_cell_divergence']=flux/h**2
    # Independent polynomial streamfunction control: fourth-order divergence is
    # exact at roundoff, while second order retains expected truncation error.
    x,y=np.meshgrid(np.linspace(0,1,13),np.linspace(0,1,13),indexing='ij')
    V=np.stack((x*x*(1-x)**2*2*y*(1-y)*(1-2*y),-2*x*(1-x)*(1-2*x)*y*y*(1-y)**2),axis=2)
    V*=.5/np.max(np.linalg.norm(V,axis=2))
    result['manufactured_control']=dict(max_speed=.5,fd2=stats(fd(V,D2)[1:-1,1:-1]),fd4=stats(fd(V,D4)[1:-1,1:-1]),cell_divergence=stats(cell_flux(V,h)/h**2))
    (out/'results.json').write_text(json.dumps(result,indent=2));np.savez(out/'fields.npz',**fields)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
