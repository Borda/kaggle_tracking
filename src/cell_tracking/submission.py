"""Submission CSV builder and validator for the cell tracking competition."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_SUBMISSION_COLUMNS = ("id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")


@dataclass(frozen=True, slots=True)
class NodeRow:
    """Single detected cell at one timepoint.

    Args:
        dataset: Dataset folder name (without ``.zarr`` extension).
        node_id: Unique integer ID for this detection.
        t: Timepoint (frame index).
        z: Z centroid in voxels.
        y: Y centroid in voxels.
        x: X centroid in voxels.

    Example:
        >>> NodeRow(dataset="44b6", node_id=1, t=0, z=32, y=128, x=128)
        NodeRow(dataset='44b6', node_id=1, t=0, z=32, y=128, x=128)
    """

    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int


@dataclass(frozen=True, slots=True)
class EdgeRow:
    """Temporal link between two detected cells.

    Args:
        dataset: Dataset folder name (without ``.zarr`` extension).
        source_id: ``node_id`` of the cell at time t.
        target_id: ``node_id`` of the cell at time t+1 (or daughter at division).

    Example:
        >>> EdgeRow(dataset="44b6", source_id=1, target_id=2)
        EdgeRow(dataset='44b6', source_id=1, target_id=2)
    """

    dataset: str
    source_id: int
    target_id: int


def build_submission(
    nodes: Sequence[NodeRow],
    edges: Sequence[EdgeRow],
    output_path: str | Path,
) -> Path:
    """Write nodes and edges to a competition submission CSV.

    Rows are written nodes-first, then edges.  The ``id`` column is filled
    with consecutive integers starting at 0.

    Args:
        nodes: Detected cell centroids, one per detection per timepoint.
        edges: Temporal links between detections.
        output_path: Destination path for the CSV file.

    Returns:
        Resolved path to the written file.

    Raises:
        ValueError: If ``nodes`` and ``edges`` are both empty.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = Path(tmp) / "submission.csv"
        ...     n = NodeRow(dataset="44b6", node_id=1, t=0, z=32, y=128, x=128)
        ...     e = EdgeRow(dataset="44b6", source_id=1, target_id=2)
        ...     path = build_submission([n], [e], out)
        ...     path.exists()
        True
    """
    if not nodes and not edges:
        msg = "At least one node or edge row is required."
        raise ValueError(msg)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_SUBMISSION_COLUMNS)
        row_id = 0
        for node in nodes:
            writer.writerow([row_id, node.dataset, "node", node.node_id, node.t, node.z, node.y, node.x, -1, -1])
            row_id += 1
        for edge in edges:
            writer.writerow([row_id, edge.dataset, "edge", -1, -1, -1, -1, -1, edge.source_id, edge.target_id])
            row_id += 1

    return out.resolve()


def validate_submission(path: str | Path) -> dict[str, int]:
    """Check a submission CSV for required columns and row types.

    Args:
        path: Path to the submission CSV.

    Returns:
        Dict with ``total_rows``, ``node_rows``, and ``edge_rows`` counts.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing or unknown ``row_type`` found.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = Path(tmp) / "submission.csv"
        ...     n = NodeRow(dataset="44b6", node_id=1, t=0, z=32, y=128, x=128)
        ...     build_submission([n], [], out)  # doctest: +ELLIPSIS
        ...     validate_submission(out)["node_rows"]
        PosixPath(...)
        1
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        actual = tuple(reader.fieldnames or [])
        if actual != _SUBMISSION_COLUMNS:
            msg = f"Expected columns {_SUBMISSION_COLUMNS}, got {actual}"
            raise ValueError(msg)
        node_rows = edge_rows = 0
        for row in reader:
            rt = row["row_type"]
            if rt == "node":
                node_rows += 1
            elif rt == "edge":
                edge_rows += 1
            else:
                msg = f"Unknown row_type: {rt!r}"
                raise ValueError(msg)

    return {"total_rows": node_rows + edge_rows, "node_rows": node_rows, "edge_rows": edge_rows}
