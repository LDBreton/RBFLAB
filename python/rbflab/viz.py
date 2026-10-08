"""Optional, lightweight Matplotlib views of numerical RBFLAB solutions.

Install with ``pip install rbflab[examples]``. Matplotlib is imported only
when a plotting function is called, so ordinary PDE use stays Python-only.
The regular display grid does not enter the numerical discretization.
"""
from pathlib import Path
import numpy as np

_BACKGROUND = "#0b1220"
_TEXT = "#e8f1ff"
_MUTED = "#9fb3ce"


def _grid(bounds, resolution):
    if type(resolution) is not int or resolution < 16:
        raise ValueError("resolution must be an integer >= 16")
    box = np.asarray(bounds, dtype=float)
    if box.shape != (2, 2) or not np.isfinite(box).all() or np.any(box[:, 1] <= box[:, 0]):
        raise ValueError("bounds must be two increasing (minimum, maximum) pairs")
    x = np.linspace(*box[0], resolution)
    y = np.linspace(*box[1], resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, np.column_stack((X.ravel(), Y.ravel()))


def _axes(ax, title):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), layout="constrained")
    else:
        fig = ax.figure
    fig.patch.set_facecolor(_BACKGROUND)
    ax.set_facecolor(_BACKGROUND)
    ax.set_title(title, loc="left", color=_TEXT, fontsize=13, weight="bold", pad=14)
    ax.set_xlabel("x", color=_MUTED)
    ax.set_ylabel("y", color=_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_color("#394c68")
    ax.set_aspect("equal")
    return fig, ax


def plot_scalar(solution, *, bounds=None, domain=None, resolution=100,
                ax=None, title="Scalar field", cmap="magma", vmin=None,
                vmax=None, colorbar=True, cloud=None):
    """Plot a 2D solution using its ``evaluate(points)`` method.

    Return ``(figure, axes)``. Pass ``cloud`` to show collocation nodes.
    Sampling here is exclusively for display, independent of PDE assembly.
    Pass domain to mask holes/exterior points and use its bounds. Only interior
    display points are evaluated; no triangulation is inferred.
    """
    x, y, points = _grid(bounds if bounds is not None else (domain.bounds if domain is not None else ((0, 1), (0, 1))), resolution)
    values = _sample_domain(solution.evaluate, points, domain).reshape(resolution, resolution)
    if not np.isfinite(values).all():
        raise ValueError("Cannot plot nonfinite solution values")
    fig, ax = _axes(ax, title)
    image = ax.imshow(values, extent=(x[0], x[-1], y[0], y[-1]), origin="lower",
                      interpolation="bicubic", cmap=cmap, vmin=vmin, vmax=vmax)
    if cloud is not None:
        nodes = np.asarray(cloud.points, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("node overlay requires a 2D cloud")
        ax.scatter(nodes[:, 0], nodes[:, 1], s=6, c="white", alpha=.28, linewidths=0)
    if colorbar:
        bar = fig.colorbar(image, ax=ax, shrink=.8, pad=.025)
        bar.ax.tick_params(colors=_MUTED, labelsize=8)
        bar.outline.set_edgecolor("#394c68")
    return fig, ax


def plot_velocity(solution, *, bounds=None, domain=None, resolution=70,
                  ax=None, title="Divergence-free velocity", cmap="viridis"):
    """Plot speed and streamlines, optionally masked by a 2D domain."""
    x, y, points = _grid(bounds if bounds is not None else (domain.bounds if domain is not None else ((0, 1), (0, 1))), resolution)
    velocity = _sample_domain(solution.velocity, points, domain, components=2)
    if velocity.shape != (len(points), 2) or not np.isfinite(velocity).all():
        raise ValueError("velocity must be finite with shape (N, 2)")
    u = velocity[:, 0].reshape(resolution, resolution)
    v = velocity[:, 1].reshape(resolution, resolution)
    fig, ax = _axes(ax, title)
    speed = np.hypot(u, v)
    image = ax.imshow(speed, extent=(x[0], x[-1], y[0], y[-1]), origin="lower",
                      interpolation="bicubic", cmap=cmap)
    if speed.max() > 0:
        ax.streamplot(x, y, u, v, color="#eaf6ff", density=1.1,
                      linewidth=.8, arrowsize=.75, minlength=.15)
    bar = fig.colorbar(image, ax=ax, shrink=.8, pad=.025)
    bar.set_label("speed", color=_MUTED)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_edgecolor("#394c68")
    return fig, ax


def animate_scalar(trajectory, path, *, bounds=None, domain=None,
                   resolution=90, fps=12, every=1, title="Heat diffusion"):
    """Save a 2D trajectory with fixed color limits and optional domain mask."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt

    path = Path(path)
    if path.suffix.lower() != ".gif":
        raise ValueError("animation path must end in .gif")
    if type(every) is not int or every < 1 or fps <= 0:
        raise ValueError("every and fps must be positive")
    x, y, points = _grid(bounds if bounds is not None else (domain.bounds if domain is not None else ((0, 1), (0, 1))), resolution)
    states = [trajectory.initial, *trajectory.states[::every]]
    if states[-1] is not trajectory.final:
        states.append(trajectory.final)
    frames = []
    for state in states:
        values = _sample_domain(state.evaluate, points, domain).reshape(resolution, resolution)
        if not np.isfinite(values).all():
            raise ValueError("Cannot animate nonfinite solution values")
        frames.append(values)
    fig, ax = _axes(None, title)
    image = ax.imshow(frames[0], extent=(x[0], x[-1], y[0], y[-1]), origin="lower",
                      interpolation="bicubic", cmap="magma",
                      vmin=float(min(v.min() for v in frames)), vmax=float(max(v.max() for v in frames)))
    bar = fig.colorbar(image, ax=ax, shrink=.8, pad=.025)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_edgecolor("#394c68")
    caption = ax.text(.03, .04, "t = 0", transform=ax.transAxes, color=_TEXT,
                      fontsize=10, weight="bold", bbox=dict(facecolor=_BACKGROUND,
                      edgecolor="none", alpha=.75, pad=6))

    def update(index):
        image.set_data(frames[index])
        stamp = getattr(states[index], "time", 0)
        caption.set_text(f"t = {float(stamp):.3f}")
        return image, caption

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps,
                              blit=False, repeat=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return path


def animate_velocity_samples(points, times, velocities, path, *,
                             bounds=((0, 1), (0, 1)), resolution=85,
                             fps=10, title="Velocity", poster=None):
    """Animate nodal 2D velocity as speed and arrows; optionally save a PNG.

    Linear interpolation onto a display grid does not change the numerical
    solution. ``velocities`` has shape (frames, nodes, 2).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    points = np.asarray(points, dtype=float)
    times = np.asarray(times, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    path = Path(path)
    if path.suffix.lower() != ".gif":
        raise ValueError("animation path must end in .gif")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be finite 2D coordinates")
    if (velocities.shape != (len(times), len(points), 2) or len(times) < 2
            or not np.isfinite(velocities).all() or not np.isfinite(times).all()):
        raise ValueError("times and velocities must contain finite matching frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    x, y, grid = _grid(bounds, resolution)
    tri = Delaunay(points)
    frames = []
    for velocity in velocities:
        field = LinearNDInterpolator(tri, velocity, fill_value=0.)(grid)
        frames.append(field.reshape(resolution, resolution, 2))
    speed_max = max(1e-12, max(np.linalg.norm(frame, axis=2).max() for frame in frames))
    fig, ax = _axes(None, title)
    extent = (x[0], x[-1], y[0], y[-1])
    image = ax.imshow(np.linalg.norm(frames[0], axis=2), extent=extent,
                      origin="lower", cmap="viridis", vmin=0, vmax=speed_max)
    skip = (slice(4, -4, 8), slice(4, -4, 8))
    X, Y = np.meshgrid(x, y)
    arrows = ax.quiver(X[skip], Y[skip], frames[0][:, :, 0][skip],
                       frames[0][:, :, 1][skip], color="#eaf6ff",
                       scale=8, width=.003, alpha=.85)
    bar = fig.colorbar(image, ax=ax, shrink=.8, pad=.025)
    bar.set_label("speed / lid speed", color=_MUTED)
    bar.ax.tick_params(colors=_MUTED, labelsize=8)
    bar.outline.set_edgecolor("#394c68")
    caption = ax.text(.03, .04, "", transform=ax.transAxes, color=_TEXT,
                      fontsize=10, weight="bold", bbox=dict(facecolor=_BACKGROUND,
                      edgecolor="none", alpha=.75, pad=6))

    def update(index):
        field = frames[index]
        image.set_data(np.linalg.norm(field, axis=2))
        arrows.set_UVC(field[:, :, 0][skip], field[:, :, 1][skip])
        caption.set_text(f"t = {times[index]:.2f}")
        return image, arrows, caption

    path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(frames),
                              interval=1000 / fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=110)
    if poster is not None:
        poster = Path(poster)
        poster.parent.mkdir(parents=True, exist_ok=True)
        update(-1)
        arrows.remove()
        final = frames[-1]
        ax.streamplot(x, y, final[:, :, 0], final[:, :, 1],
                      color="#eaf6ff", density=1.25, linewidth=.7,
                      arrowsize=.7, minlength=.15)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        fig.savefig(poster, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _sample_domain(evaluate, points, domain, components=None):
    """Evaluate only physical points; mask holes before rendering/interpolation."""
    inside = np.ones(len(points), dtype=bool) if domain is None else domain.contains_points(points)
    if not np.any(inside):
        raise ValueError("Display grid does not intersect the domain")
    shape = (len(points),) if components is None else (len(points), components)
    result = np.ma.masked_all(shape, dtype=float)
    values = np.asarray(evaluate(points[inside]), dtype=float)
    expected = (int(inside.sum()),) if components is None else (int(inside.sum()), components)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("Display evaluator must return finite values with matching shape")
    result[inside] = values
    return result


def plot_cloud(cloud, *, ax=None, title="Nodes and boundary groups", normals=False, stencil=None):
    """Plot a 2D/3D cloud with boundary labels and optional normals/stencil IDs.

    Connectivity is never fabricated. For 3D, pass a Matplotlib 3D axes or
    leave ax unset. Returns (figure, axes); does not call show().
    """
    import matplotlib.pyplot as plt
    if cloud.dimension == 3:
        if ax is None:
            fig = plt.figure(figsize=(6, 5), layout="constrained")
            ax = fig.add_subplot(projection="3d")
        else:
            fig = ax.figure
        fig.patch.set_facecolor(_BACKGROUND); ax.set_facecolor(_BACKGROUND)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=title)
        ax.set_box_aspect(np.ptp(cloud.points, axis=0))
        ax.tick_params(colors=_MUTED)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.label.set_color(_MUTED)
        ax.title.set_color(_TEXT)
    else:
        fig, ax = _axes(ax, title)
    p = cloud.points
    ax.scatter(*p[cloud.interior_indices].T, s=5, color="#748aa5", alpha=.55, label="interior")
    colors = ["#53d7cc", "#ffbc6e", "#af9bff", "#f487b5", "#8cdf89"]
    for k, (label, ids) in enumerate(cloud.boundary.items()):
        color=colors[k % len(colors)]
        ax.scatter(*p[ids].T, s=13, color=color, label=str(label))
        if normals and label in cloud.normals:
            take = slice(None, None, max(1, len(ids)//18))
            v = cloud.normals[label][take]; q=p[ids][take]
            if cloud.dimension == 2:
                ax.quiver(*q.T, *v.T, color=color, scale=20, width=.003)
            else:
                ax.quiver(*q.T, *v.T, color=color, length=.12)
    if stencil is not None:
        ax.scatter(*p[np.asarray(stencil, dtype=int)].T, s=42, facecolors="none", edgecolors="white", label="stencil")
    ax.legend(facecolor=_BACKGROUND, edgecolor="#394c68", labelcolor=_TEXT, fontsize=8)
    return fig, ax


def plot_slice(solution, domain, *, axis=2, coordinate=0., resolution=85,
               ax=None, title="3D solution / cross-section", cmap="magma"):
    """Plot a scalar 3D evaluator on one Cartesian slice, masked by domain.

    axis is the fixed coordinate (0, 1 or 2); coordinate is its physical value.
    The slice samples the numerical reconstruction, not an extra PDE grid.
    """
    if axis not in (0, 1, 2) or not np.isfinite(coordinate):
        raise ValueError("axis must be 0, 1 or 2 and coordinate finite")
    free = [i for i in range(3) if i != axis]
    bounds = np.asarray(domain.bounds)[free]
    x, y, flat = _grid(bounds, resolution)
    points = np.empty((len(flat),3)); points[:,free] = flat; points[:,axis] = coordinate
    values = _sample_domain(solution.evaluate, points, domain).reshape(resolution,resolution)
    fig, ax = _axes(ax, title)
    image=ax.imshow(values, extent=(x[0],x[-1],y[0],y[-1]), origin="lower", cmap=cmap)
    ax.set_xlabel("xyz"[free[0]], color=_MUTED);ax.set_ylabel("xyz"[free[1]], color=_MUTED)
    bar=fig.colorbar(image, ax=ax, shrink=.8, pad=.025);bar.ax.tick_params(colors=_MUTED)
    return fig, ax
