"""Tests for cell_tracking package imports and metadata."""

import cell_tracking


def test_package_version_is_string() -> None:
    """Package version attribute is a non-empty string."""
    assert isinstance(cell_tracking.__version__, str)
    assert len(cell_tracking.__version__) > 0


def test_public_api_exports() -> None:
    """All names declared in __all__ are importable from the package."""
    for name in cell_tracking.__all__:
        assert hasattr(cell_tracking, name), f"Missing from package: {name}"
