# Symbolic equations and custom kernels

**You will learn:** to alter a symbolic operator independently from its kernel, give a kernel a runtime parameter, and inspect its origin value. Prerequisite: the [interpolation tutorial](interpolation.md).

The script declares an IMQ-like symbolic scalar kernel with a parameter and uses it in a small manufactured interpolation calculation. A runtime parameter can be changed without rewriting the PDE declaration. For compiled execution, supported kernel expressions may be cached by their structure while parameters remain inputs; compilation is an optional backend path, not part of the default `pip install rbflab`.

For squared distance \(s=\|x-y\|^2\), the family is
\(\phi(s;c)=(1+cs)^{-1/2}\) with \(c>0\). The origin limit is
\(\phi(0;c)=1\), so coincident nodes are evaluated at the mathematical
limit rather than by substituting an artificial radius. The script
declares the expression and parameter with SymPy, binds \(c=2\), solves
an interpolation problem, and checks the kernel origin value.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/custom_kernel.py"
    ```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/custom_kernel.py).

Run `python -m examples.tutorials.custom_kernel`. The example evaluates the kernel's known \(r\to0\) value, then reports a sampled maximum error of \(3.736\times10^{-3}\) at two query points. Test a different parameter and compare conditioning as well as error. A piecewise compact-support kernel needs validated branches and correct one-sided/origin derivatives; see the [kernel API](../api/kernels-geometry.md) and [conditioning](../theory/conditioning.md).
