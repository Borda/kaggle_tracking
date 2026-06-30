"""Tests for the submission builder and validator."""

from pathlib import Path

import pytest

from cell_tracking.submission import EdgeRow, NodeRow, build_submission, validate_submission


def _make_node(dataset: str = "44b6", node_id: int = 1, t: int = 0) -> NodeRow:
    return NodeRow(dataset=dataset, node_id=node_id, t=t, z=32, y=128, x=128)


def _make_edge(dataset: str = "44b6", source_id: int = 1, target_id: int = 2) -> EdgeRow:
    return EdgeRow(dataset=dataset, source_id=source_id, target_id=target_id)


class TestBuildSubmission:
    """build_submission writes a valid CSV with correct row counts."""

    def test_nodes_only(self, tmp_path: Path) -> None:
        """Node-only submission writes one data row."""
        out = build_submission([_make_node()], [], tmp_path / "sub.csv")
        assert out.exists()
        stats = validate_submission(out)
        assert stats["node_rows"] == 1
        assert stats["edge_rows"] == 0

    def test_edges_only(self, tmp_path: Path) -> None:
        """Edge-only submission writes one data row."""
        out = build_submission([], [_make_edge()], tmp_path / "sub.csv")
        stats = validate_submission(out)
        assert stats["edge_rows"] == 1
        assert stats["node_rows"] == 0

    def test_mixed_rows(self, tmp_path: Path) -> None:
        """Mixed submission counts nodes and edges separately."""
        nodes = [_make_node(node_id=i) for i in range(3)]
        edges = [_make_edge(source_id=i, target_id=i + 1) for i in range(2)]
        out = build_submission(nodes, edges, tmp_path / "sub.csv")
        stats = validate_submission(out)
        assert stats["node_rows"] == 3
        assert stats["edge_rows"] == 2
        assert stats["total_rows"] == 5

    def test_empty_raises(self, tmp_path: Path) -> None:
        """Empty nodes and edges raises ValueError."""
        with pytest.raises(ValueError, match="required"):
            build_submission([], [], tmp_path / "sub.csv")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """build_submission creates missing parent directories."""
        out = tmp_path / "nested" / "deep" / "sub.csv"
        build_submission([_make_node()], [], out)
        assert out.exists()


class TestValidateSubmission:
    """validate_submission checks columns and row types."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when path does not exist."""
        with pytest.raises(FileNotFoundError):
            validate_submission(tmp_path / "missing.csv")

    def test_wrong_columns_raises(self, tmp_path: Path) -> None:
        """ValueError raised when column names do not match."""
        bad = tmp_path / "bad.csv"
        bad.write_text("id,wrong_col\n1,x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="columns"):
            validate_submission(bad)
