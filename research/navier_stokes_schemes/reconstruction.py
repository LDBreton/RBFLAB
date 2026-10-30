"""Moving-point evaluation with frozen mixed Hermite observation stencils.

Uses public LocalApproximation/Samples APIs only. Each query batch constructs
new weights; the public API currently exposes no reusable local fitted field.
An exact-coordinate LRU cache avoids rebuilding repeated trajectory queries.
"""
from collections import OrderedDict
from dataclasses import replace

import numpy as np


class FixedStencilReconstruction:
    """Evaluate the same local interpolants used at the arrival nodes.

    ``local`` is the original mixed LocalApproximation, ``reference_map`` any
    one of its maps at the arrival nodes, and ``velocity`` its velocity field
    expression. The reference map's public ``local(i).groups`` owns the exact
    original memberships, including original-target momentum exclusion.

    Queries must retain arrival-node order. An optional ``rows`` index vector
    allows an explicitly identified subset; membership is never reselected by
    distance to a departing point. All local/native options are preserved.
    """

    def __init__(self, local, reference_map, velocity, *, cache_size=8):
        if local.kernel_scaling != "physical":
            raise ValueError("Fixed interpolants require physical kernel scaling")
        self.local = local
        self.velocity = velocity
        count = len(reference_map.output_layout["points"])
        self.memberships = tuple(reference_map.local(i).groups for i in range(count))
        self.cache_size = max(0, int(cache_size))
        self.cache = OrderedDict()
        self.builds = 0
        self.cache_hits = 0

    def maps(self, points, *, rows=None):
        points = np.ascontiguousarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected an (N,2) moving query array")
        if rows is None:
            if len(points) != len(self.memberships):
                raise ValueError("A subset query must identify its arrival-node rows")
            rows = np.arange(len(points), dtype=int)
        rows = np.asarray(rows, dtype=int)
        if rows.shape != (len(points),) or np.any(rows < 0) or np.any(rows >= len(self.memberships)):
            raise ValueError("Invalid arrival-node row indices")
        key = (points.shape, points.tobytes(), rows.tobytes())
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        source = {
            name: replace(group, indices=[self.memberships[i][name] for i in rows],
                          size=None, within=None, policy=None, target="allow")
            for name, group in self.local.source.items()
        }
        # Recreate matching representers from the same space and frozen groups.
        # The target policy is now allow: exclusion was already applied at the
        # original arrival point and must not be reinterpreted at the new foot.
        frozen = replace(self.local, source=source,
                         trial=self.local.trial.space.representers(source))
        result = frozen.operators(targets=points, operators={"value": self.velocity}).value
        self.builds += 1
        if self.cache_size:
            self.cache[key] = result
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result

    def evaluate(self, U, L, boundary, points, *, rows=None):
        """Evaluate nodal observations; outputs have node-major (N,2) shape."""
        values = {"velocity": np.asarray(U).reshape(-1, 2),
                  "boundary": np.asarray(boundary).reshape(-1, 2),
                  "momentum": np.asarray(L).reshape(-1, 2)}
        if set(self.local.source) != set(values):
            raise ValueError("Expected velocity, boundary and momentum sample groups")
        return np.asarray(self.maps(points, rows=rows) @ values).reshape(-1, 2)
