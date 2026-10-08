"""Run a short cavity calculation and inspect the discrete flow diagnostics.

The detailed assembly lives in examples/navier_stokes_cavity.py.
Run: python -m examples.tutorials.cavity
"""
import argparse
from pathlib import Path

from examples.navier_stokes_cavity import solve
from rbflab import viz


def run(cells=6, end=0.005, output=None):
    points, times, velocity, diagnostics = solve(cells=cells, end=end, frames=5)
    print(f"saved_times={len(times)}, velocity_array={velocity.shape}")
    print(diagnostics)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        viz.animate_velocity_samples(points, times, velocity, output,
                                     title="RBF-FD cavity startup",
                                     poster=output.with_suffix(".png"))
        print(f"animation={output}")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--end", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.cells, args.end, args.output)
