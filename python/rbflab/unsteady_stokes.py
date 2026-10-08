"""Fixed-basis global unsteady Stokes: M*c_dot + A*c = b(t).

The velocity mass rows act only at interior momentum sites. Boundary rows,
pressure gauge and pressure side constraint remain algebraic. No pressure
splitting or pressure boundary condition is introduced.
"""
from dataclasses import dataclass,field
from fractions import Fraction
import numpy as np
from .precision import Precision,PrecisionData,mp_number
from .operators import Identity
from .stokes import GlobalStokes,StokesProblem,StokesSystem,StokesSolution,_blocks


from .time_data import TimeData, at_time as _at


@dataclass(frozen=True)
class UnsteadyStokesProblem:
    forcing: tuple
    boundary_velocity: tuple
    initial_velocity: tuple
    viscosity: object = 1
    pressure_point: tuple = (0.,0.)
    pressure_value: object = 0

    def __post_init__(self):
        if len(self.initial_velocity)!=2:raise ValueError("Expected two initial velocity components")
        StokesProblem(self.forcing,self.boundary_velocity,self.viscosity,self.pressure_point,0)


class _InitialVelocity:
    def __init__(self,a,data,time):self.arithmetic,self.data,self.time=a,data,time

    def velocity(self,points):
        a=self.arithmetic;out=a.zeros(len(points),2)
        for component in (0,1):
            data=a.data(self.data[component],points)
            for i,v in enumerate(data):out[i,component]=v
        return out


@dataclass
class GlobalUnsteadyStokes:
    kernel: object
    precision: Precision = field(default_factory=Precision)

    def assemble(self,problem,cloud):
        if not isinstance(problem,UnsteadyStokesProblem):raise TypeError("Expected UnsteadyStokesProblem")
        base=GlobalStokes(self.kernel,self.precision).assemble(
            StokesProblem((0,0),(0,0),problem.viscosity,problem.pressure_point,0),cloud)
        a=base.arithmetic;n=len(base.points);ni=len(cloud.interior)
        mass=a.zeros(n+1,n+1)
        for component in (0,1):
            mass[component*ni:(component+1)*ni,:n]=_blocks(a,cloud.interior,
                [{component:Identity()} for _ in range(ni)],base.points,base.functionals)
        return UnsteadyStokesSystem(base,problem,mass)


class UnsteadyStokesSystem:
    def __init__(self,base,problem,mass):
        self.base,self.problem,self.mass=base,problem,mass
        self.arithmetic=base.arithmetic
        self._factors={}

    def _snapshot(self,time):
        a=self.arithmetic;p=self.problem;cloud=self.base.cloud
        forcing=tuple(_at(v,time) for v in p.forcing)
        boundary=tuple(_at(v,time) for v in p.boundary_velocity)
        gauge=a.data(_at(p.pressure_value,time),np.asarray([p.pressure_point]))[0]
        problem=StokesProblem(forcing,boundary,p.viscosity,p.pressure_point,gauge)
        data=[]
        for f in forcing:data.extend(a.data(f,cloud.interior))
        for g in boundary:data.extend(a.data(g,cloud.points[cloud.boundary_indices]))
        data.extend([gauge,a.number(0)])
        return StokesSystem(a,problem,cloud,self.base.points,self.base.functionals,
                            self.base.matrix,a.vector(data))

    def solve(self,dt,steps,*,scheme="bdf2",start_time=0):
        """Constant dt. BDF2 starts with one backward Euler step.

        Initial velocity must be solenoidal and compatible with boundary data.
        Initial pressure is not required and is not fabricated. Returned states
        correspond to positive steps; the initial object stores velocity only.
        """
        dt=Fraction(dt);start_time=Fraction(start_time)
        if dt<=0:raise ValueError("dt must be positive")
        if type(steps) is not int or steps<1:raise ValueError("steps must be a positive integer")
        if scheme not in ("backward_euler","bdf2"):raise ValueError("Unknown time scheme")
        a=self.arithmetic;cloud=self.base.cloud
        initial=_InitialVelocity(a,self.problem.initial_velocity,a.number(start_time))
        snapshot=self._snapshot(a.number(start_time))
        boundary=cloud.points[cloud.boundary_indices]
        iv=initial.velocity(boundary)
        tolerance=a.ctx.sqrt(a.ctx.eps) if a.ctx else 1e-10
        for c in (0,1):
            expected=a.data(snapshot.problem.boundary_velocity[c],boundary)
            if any(abs(iv[i,c]-v)>tolerance*(1+abs(v)) for i,v in enumerate(expected)):
                raise ValueError("Initial velocity is incompatible with boundary data at start_time")
        velocity=initial.velocity(cloud.interior);ni=len(cloud.interior)
        previous=a.vector([velocity[i,c] for c in (0,1) for i in range(ni)]+[a.number(0)]*(len(self.base.rhs)-2*ni))
        previous2=None;states=[];new_factors=0
        for step in range(1,steps+1):
            bdf2=scheme=="bdf2" and step>1
            alpha=Fraction(3,2)/dt if bdf2 else 1/dt
            history_coefficients=(2/dt,-Fraction(1,2)/dt) if bdf2 else (1/dt,)
            time=a.number(start_time+step*dt)
            snapshot=self._snapshot(time)
            rhs=snapshot.rhs+a.number(history_coefficients[0])*previous
            if bdf2:rhs+=a.number(history_coefficients[1])*previous2
            if alpha not in self._factors:
                matrix=self.base.matrix+a.number(alpha)*self.mass
                self._factors[alpha]=(matrix,a.factor(matrix));new_factors+=1
            matrix,factor=self._factors[alpha]
            coefficients=factor.solve(rhs)
            defect=(matrix*coefficients if a.ctx else matrix@coefficients)-rhs
            relative=a.norm(defect)/(a.norm(rhs) or a.number(1))
            history=([states[-1] if states else initial] if not bdf2 else
                     [states[-1],states[-2] if len(states)>1 else initial])
            state=UnsteadyStokesState(snapshot,coefficients,{
                "relative_residual":float(relative),"scaled_condition":factor.condition,
                "global_digits":a.ctx.dps if a.ctx else None,
                "scheme":"bdf2" if bdf2 else "backward_euler",},
                time,a.number(alpha),history,[a.number(v) for v in history_coefficients])
            states.append(state)
            previous2=previous
            previous=self.mass*coefficients if a.ctx else self.mass@coefficients
        return StokesTrajectory(initial,states,{"new_factorizations":new_factors,
            "cached_factorizations":len(self._factors),"scheme":scheme,"steps":steps,
            "dt":str(dt),"initial_pressure":"not prescribed"})


class UnsteadyStokesState(StokesSolution):
    def __init__(self,system,coefficients,diagnostics,time,alpha,history,weights):
        super().__init__(system,coefficients,diagnostics)
        self.time,self.alpha,self.history,self.history_weights=time,alpha,history,weights

    def velocity(self,points):
        return self.evaluate(points,extended=bool(self.system.arithmetic.ctx))[:,:2]

    def residuals(self,points,*,extended=False):
        """Discrete-in-time momentum residual (BE/BDF2), and divergence.

        This uses the numerical history, NOT an exact continuous time derivative.
        """
        from .methods import _query
        points=_query(points);a=self.system.arithmetic
        residual=super().residuals(points,extended=bool(a.ctx))
        derivative=self.alpha*self.velocity(points)
        for state,weight in zip(self.history,self.history_weights):
            derivative-=weight*state.velocity(points)
        for i in range(len(points)):
            for c in (0,1):residual[i,c]+=derivative[i,c]
        if extended:
            if not a.ctx:raise ValueError("extended requires global_digits")
            return residual
        return np.array(residual.tolist(),dtype=float) if a.ctx else residual


@dataclass
class StokesTrajectory:
    initial: object
    states: list
    diagnostics: dict

    @property
    def final(self):return self.states[-1]
