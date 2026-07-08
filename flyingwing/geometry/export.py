"""Export a `TriangleMesh` (see `geometry/mesh.py`) as a binary STL file, for
external CFD surface meshing (snappyHexMesh, ANSA, etc.) or any other
downstream tool that reads STL. Hand-rolled rather than a new dependency --
binary STL is a simple fixed format: an 80-byte header, a uint32 triangle
count, then per triangle a normal vector + 3 vertices + a 0 attribute byte
count, all little-endian float32/uint16.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .mesh import TriangleMesh


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1)
    lengths_safe = np.where(lengths > 1e-12, lengths, 1.0)
    return normals / lengths_safe[:, None]


def write_stl(mesh: TriangleMesh, path: str | Path, name: bytes = b"flying_wing") -> Path:
    """Vectorized (not a per-face Python loop -- this project's watertight
    mesh runs to 100k+ faces at the default 200-station resolution)."""
    path = Path(path)
    faces = mesh.faces
    n_faces = len(faces)
    normals = _face_normals(mesh.vertices, faces).astype("<f4")
    tri_vertices = mesh.vertices[faces].astype("<f4")  # (n_faces, 3, 3)

    record_dtype = np.dtype([
        ("normal", "<f4", 3),
        ("vertices", "<f4", (3, 3)),
        ("attribute_byte_count", "<u2"),
    ])
    records = np.zeros(n_faces, dtype=record_dtype)
    records["normal"] = normals
    records["vertices"] = tri_vertices

    header = name[:80].ljust(80, b"\0")
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_faces))
        f.write(records.tobytes())
    return path
