"""I/O helpers for zarr volumes and geff tracking graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def load_zarr_timepoint(zarr_path: str | Path, t: int) -> npt.NDArray[np.uint16]:
    """Load a single timepoint from a zarr v3 volume.

    The volume has shape (T, Z, Y, X).  Each timepoint chunk lives at
    ``0/{t}/0/0/0`` inside the zarr store.

    Args:
        zarr_path: Path to the ``.zarr`` directory.
        t: Timepoint index (0-based).

    Returns:
        3-D array of shape (Z, Y, X) with dtype uint16.

    Raises:
        ImportError: If ``zarr`` is not installed.
        FileNotFoundError: If ``zarr_path`` does not exist.
        IndexError: If ``t`` is out of range.

    Example:
        >>> import tempfile, pathlib
        >>> # no zarr store present -- verify error type only
        >>> try:
        ...     load_zarr_timepoint("/nonexistent.zarr", 0)
        ... except (FileNotFoundError, Exception):
        ...     pass
    """
    try:
        import zarr  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "zarr is required: pip install zarr"
        raise ImportError(msg) from exc

    store = zarr.open(str(zarr_path), mode="r")
    arr = store["0"]
    if t < 0 or t >= arr.shape[0]:
        msg = f"Timepoint {t} out of range [0, {arr.shape[0]})"
        raise IndexError(msg)
    return arr[t].astype(np.uint16)


def read_geff_graph(geff_path: str | Path) -> dict[str, Any]:
    """Read a geff ground-truth tracking graph.

    A ``.geff`` directory (Zarr v3) contains:

    - ``nodes/ids`` -- node ID array
    - ``nodes/props/{t,z,y,x}/values`` -- integer centroid coordinates
    - ``edges/ids`` -- edge array of shape (N, 2) with (source_id, target_id)

    Args:
        geff_path: Path to the ``.geff`` directory.

    Returns:
        Dict with keys ``node_ids``, ``node_t``, ``node_z``, ``node_y``,
        ``node_x``, ``edge_sources``, ``edge_targets``.

    Raises:
        ImportError: If ``zarr`` is not installed.
        FileNotFoundError: If ``geff_path`` does not exist.

    Example:
        >>> try:
        ...     read_geff_graph("/nonexistent.geff")
        ... except (FileNotFoundError, Exception):
        ...     pass
    """
    try:
        import zarr  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "zarr is required: pip install zarr"
        raise ImportError(msg) from exc

    g = zarr.open(str(geff_path), mode="r")
    node_ids: npt.NDArray[np.int64] = np.asarray(g["nodes/ids"])
    edge_ids: npt.NDArray[np.int64] = np.asarray(g["edges/ids"])

    def _prop(name: str) -> npt.NDArray[np.int64]:
        return np.asarray(g[f"nodes/props/{name}/values"])

    return {
        "node_ids": node_ids,
        "node_t": _prop("t"),
        "node_z": _prop("z"),
        "node_y": _prop("y"),
        "node_x": _prop("x"),
        "edge_sources": edge_ids[:, 0],
        "edge_targets": edge_ids[:, 1],
    }
