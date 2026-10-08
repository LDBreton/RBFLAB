"""Small shared utilities for curved-domain examples (not part of the public API)."""
from time import perf_counter
import numpy as np
import rbflab as rbf
from rbflab import meshgen


def local_method(backend="python", stencil=35, degree=3):
    """Select a local backend; global sparse solution still uses SciPy Float64."""
    backends = {"python": rbf.PythonBackend, "cpp": rbf.CppBackend, "torch": rbf.TorchBackend}
    if backend not in backends:
        raise ValueError("backend must be python, cpp or torch")
    options = {"compute_condition": False}
    if backend != "python": options["threads"] = 4
    return rbf.RBFFD(rbf.PHS(5), stencil, polynomial_degree=degree,
                      local_backend=backends[backend](**options),
                      stencil_policy=rbf.StencilPolicy(scaling="local"))


def solve(problem, cloud, method, *, dt=None, steps=None):
    """Report preparation, assembly and solve separately, including cold setup."""
    start=perf_counter()
    if isinstance(getattr(method, "local_backend", None), (rbf.CppBackend, rbf.TorchBackend)):
        method=method.prepare(problem, dimension=cloud.dimension)
    prepared=perf_counter()
    system=method.assemble(problem, cloud)
    assembled=perf_counter()
    solution=system.solve() if dt is None else system.solve(str(dt), steps)
    done=perf_counter()
    return solution, dict(prepare_s=prepared-start, assembly_s=assembled-prepared, solve_s=done-assembled)


def errors(solution, cloud, domain, exact, *, query_count=120):
    """Maximum nodal and independent off-node solution errors, not residuals."""
    query=meshgen.generate(domain, interior=query_count, boundary=36, seed=981).interior
    return {"nodal_max":float(np.max(np.abs(solution.evaluate(cloud.points)-exact(cloud.points)))),
            "off_node_max":float(np.max(np.abs(solution.evaluate(query)-exact(query))))}


def report(name, cloud, timings, error):
    """Print measured errors, timings and precision explicitly."""
    result={"example":name, "nodes":len(cloud.points), "precision":"float64", **timings, **error}
    print(result, flush=True)
    return result
