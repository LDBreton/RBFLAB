"""Define a runtime-parameter kernel and reuse it in interpolation and RBF-FD."""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf


def run(backend="python", compile_kernel=False):
    # --8<-- [start:family]
    s = sp.Symbol("s", nonnegative=True)  # squared distance, not radius
    c = sp.Symbol("c", positive=True)
    imq_family = rbf.Kernel((1+c*s)**sp.Rational(-1, 2), s, (c,))
    imq = imq_family(c="2")
    # --8<-- [end:family]
    # --8<-- [start:derivatives]
    offsets = np.array([[0., 0.], [.2, -.1]])
    kernel_values = imq.derivative(offsets, (0, 0))
    kernel_dx = imq.derivative(offsets, (1, 0))
    kernel_dxx = imq.derivative(offsets, (2, 0))
    reference_dxx = rbf.IMQ(2).derivative(offsets, (2, 0))
    # --8<-- [end:derivatives]
    np.testing.assert_allclose(kernel_dxx, reference_dxx, atol=1e-13)
    # --8<-- [start:mixture]
    beta = sp.Symbol("beta", nonnegative=True)
    family = rbf.Kernel((1+c*s)**sp.Rational(-1, 2) + beta*sp.exp(-c*s),
                        s, (c, beta))
    kernel = family(c="2", beta="0.1")
    another_kernel = family(c="3", beta="0.2")
    # --8<-- [end:mixture]
    if compile_kernel:
        # --8<-- [start:compile]
        artifact = family.compile(dimension=2, derivative_order=2, arithmetic="float64")
        compiled_kernel = artifact(c="2", beta="0.1")
        # --8<-- [end:compile]
        np.testing.assert_allclose(compiled_kernel.derivative(offsets, (2, 0)),
                                   kernel.derivative(offsets, (2, 0)), atol=1e-12)
        reused = family.compile(dimension=2, derivative_order=2, arithmetic="float64")
        assert artifact.directory == reused.directory
        print(f"Compiled derivative checked; cached artifact reused: {artifact.directory}")
    # --8<-- [start:interpolation]
    centers = rbf.unit_box_grid(4).points
    values = np.sin(centers[:, 0]) + np.cos(centers[:, 1])
    query = np.array([[.27, .42], [.61, .35]])
    interpolant = rbf.interpolate(kernel, centers, values, polynomial_degree=2)
    predicted = interpolant.evaluate(query)
    # --8<-- [end:interpolation]
    local = {"python": rbf.PythonBackend, "cpp": rbf.CppBackend,
             "torch": rbf.TorchBackend}[backend](compute_condition=False)
    # --8<-- [start:operators]
    method = rbf.RBFFD(kernel, stencil_size=20, polynomial_degree=2,
                       local_backend=local)
    ops = method.operators(source=centers, targets=query,
                           operators={"dx": rbf.Derivative(0), "lap": rbf.Laplacian()})
    field_dx = ops.dx @ values
    field_laplacian = ops.lap @ values
    # --8<-- [end:operators]
    truth = np.sin(query[:, 0])+np.cos(query[:, 1])
    error = float(np.max(np.abs(predicted-truth)))
    print(f"IMQ origin value/dx/dxx: {kernel_values[0]}, {kernel_dx[0]}, {kernel_dxx[0]}")
    print(f"bound parameters={kernel.values}, interpolation shape={predicted.shape}, error={error:.3e}")
    print(f"local derivative shapes: {field_dx.shape}, {field_laplacian.shape}; backend={backend}")
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    parser.add_argument("--compile", dest="compile_kernel", action="store_true")
    run(**vars(parser.parse_args()))
