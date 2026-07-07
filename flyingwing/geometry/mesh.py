"""Turn the per-half upper/lower surface grids from `aircraft.py` into a
single watertight, manifold triangle mesh of the complete (both-sides)
aircraft -- vertices + face indices, suitable for Plotly Mesh3d, STL export,
or any other downstream consumer.

The upper and lower surface grids are numerically coincident along the
leading edge (j=0) and trailing edge (j=n_chord-1) columns, since the base
MH64 shape has exactly zero thickness there (see `airfoil_family._base_shape`,
which asserts this). Those columns are welded to a single shared vertex
index rather than left as separate-but-coincident points, so the mesh is a
genuine closed manifold rather than one with a zero-width degenerate gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aircraft import Aircraft


@dataclass
class TriangleMesh:
    vertices: np.ndarray  # (Nv, 3)
    faces: np.ndarray  # (Nf, 3) int indices into vertices


def _mirror_full_span(half: np.ndarray) -> np.ndarray:
    """(N, M, 3) for y in [0, half_span] -> (2N-1, M, 3) for y in
    [-half_span, half_span], sharing a single row at y=0 (no duplicate
    vertices at the symmetry plane)."""
    mirrored = half.copy()
    mirrored[:, :, 1] *= -1.0
    # negative side: reversed order, excluding the shared y=0 row (index 0)
    negative_side = mirrored[1:][::-1]
    return np.concatenate([negative_side, half], axis=0)


def build_watertight_mesh(aircraft: Aircraft) -> TriangleMesh:
    upper_full = _mirror_full_span(aircraft.upper_surface_m)
    lower_full = _mirror_full_span(aircraft.lower_surface_m)

    n_span, n_chord, _ = upper_full.shape
    n_upper_vertices = n_span * n_chord

    # Lower-surface vertex index for (i, j), welding j=0 (LE) and
    # j=n_chord-1 (TE) onto the upper surface's vertex at the same station.
    def lower_index(i, j):
        i = np.asarray(i)
        j = np.asarray(j)
        on_seam = (j == 0) | (j == n_chord - 1)
        upper_idx = i * n_chord + j
        lower_idx = n_upper_vertices + i * n_chord + j
        return np.where(on_seam, upper_idx, lower_idx)

    def upper_index(i, j):
        return i * n_chord + j

    ii, jj = np.meshgrid(np.arange(n_span - 1), np.arange(n_chord - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()

    # Upper surface faces (outward normal roughly +z)
    a, b, c, d = upper_index(ii, jj), upper_index(ii, jj + 1), upper_index(ii + 1, jj), upper_index(ii + 1, jj + 1)
    upper_faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([b, d, c], axis=1)], axis=0
    )

    # Lower surface faces (welded at LE/TE seams, outward normal roughly -z)
    a, b, c, d = lower_index(ii, jj), lower_index(ii, jj + 1), lower_index(ii + 1, jj), lower_index(ii + 1, jj + 1)
    lower_faces = np.concatenate(
        [np.stack([a, c, b], axis=1), np.stack([b, c, d], axis=1)], axis=0
    )

    # Tip caps: with LE/TE welded, upper_index and lower_index already agree
    # at j=0 and j=n_chord-1, so the tip "cap" is just the interior columns
    # (j=1..n_chord-2) connecting the upper and lower rows at each tip.
    def tip_cap(i_row, flip):
        j = np.arange(n_chord - 1)
        u0, u1 = upper_index(i_row, j), upper_index(i_row, j + 1)
        l0, l1 = lower_index(i_row, j), lower_index(i_row, j + 1)
        if not flip:
            tri1 = np.stack([u0, u1, l0], axis=1)
            tri2 = np.stack([u1, l1, l0], axis=1)
        else:
            tri1 = np.stack([u0, l0, u1], axis=1)
            tri2 = np.stack([u1, l0, l1], axis=1)
        return np.concatenate([tri1, tri2], axis=0)

    tip_neg = tip_cap(0, flip=True)
    tip_pos = tip_cap(n_span - 1, flip=False)

    vertices = np.concatenate([upper_full.reshape(-1, 3), lower_full.reshape(-1, 3)], axis=0)
    faces = np.concatenate([upper_faces, lower_faces, tip_neg, tip_pos], axis=0)

    # Welding at the LE/TE seams and at the tip fans produces a few
    # zero-area degenerate triangles (two of whose three vertex indices
    # coincide) -- harmless for rendering, but drop them so the mesh is a
    # clean manifold.
    degenerate = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    faces = faces[~degenerate]

    return TriangleMesh(vertices=vertices, faces=faces)
