"""Independent solution, PDE and boundary centers for scalar stationary LHI."""
import numpy as np
import sympy as sp
import rbflab as rbf


def run():
    cloud = rbf.geometry.unit_box_grid(3)
    model = rbf.SymbolicScalar()
    u = model.field
    x, y = model.coordinates
    exact = 1 + x*x + y*y
    truth = model.data(exact)
    problem = model.stationary(
        sp.Eq(-model.laplacian(u), -4),
        boundary=[model.bc("boundary", sp.Eq(u, exact))],
    )
    pde_points = np.array([[.2, .3], [.7, .3], [.3, .7], [.7, .7], [.5, .5]])
    groups = {
        "values": rbf.CenterGroup(cloud.interior, role="solution", target="require"),
        "wall": rbf.CenterGroup(
            cloud.points[cloud.boundary_indices], role="boundary",
            operator=rbf.Identity(), data=truth,
        ),
        "forcing": rbf.CenterGroup(pde_points, role="pde", size=3, target="exclude"),
    }
    method = rbf.LHI(spaces={"u": rbf.ScalarSpace(rbf.PHS(5), 2)}, centers=groups)
    method.preflight(problem, cloud)
    system = method.assemble(problem, cloud)
    solution = system.solve()
    query = np.array([[.31, .47], [.68, .57]])
    error = float(np.max(np.abs(solution.evaluate(query) - truth(query))))
    print("Independent-center error:", error)
    print("First stencil groups:", system.stencils[0].groups)
    return error


if __name__ == "__main__":
    run()
