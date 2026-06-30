# 🔬 Biohub – Cell Tracking During Development

[![CI complete testing](https://github.com/Borda/kaggle_tracking/actions/workflows/ci_testing.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_tracking/actions/workflows/ci_testing.yml)
[![codecov](https://codecov.io/gh/Borda/kaggle_tracking/branch/main/graph/badge.svg)](https://codecov.io/gh/Borda/kaggle_tracking)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/kaggle_tracking/main.svg)](https://results.pre-commit.ci/latest/github/Borda/kaggle_tracking/main)

Kaggle competition: **[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)**

Detect, track, and link zebrafish cells through 3D space and time. Given time-lapse 3D fluorescence microscopy of developing zebrafish embryos, the task is to:

1. **Detect** cells per timepoint (nodes in a tracking graph)
2. **Link** cells across time (edges in the graph)
3. **Identify divisions** (nodes with ≥2 outgoing edges)

Submissions are evaluated with a combined **Edge Jaccard + Division Jaccard** metric.

## 🗂️ Project Layout

```text
.
├── src/cell_tracking/  # shared helpers (zarr I/O, geff reader, submission builder)
├── notebooks/          # percent-format notebook scripts with # %% cell markers
│   └── 00_baseline.py  # EDA + blob detection + nearest-neighbour tracking baseline
├── resources/          # competition overview, data schema, welcome notes
│   ├── Overview.md     # competition description and evaluation metric
│   ├── data.md         # data format (zarr v3, geff, submission CSV schema)
│   └── welcome.md      # organiser intro + useful packages + competitive methods
├── tests/              # doctests and regression tests for shared helpers
├── data/               # local competition files — ignored by git
└── outputs/            # local generated submissions — ignored by git
```

## ⚡ Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade uv
uv sync --group dev --extra kaggle
```

Install competition-specific packages locally:

```bash
pip install zarr geff zarr-developers tracksdata scipy scikit-image
# optional but recommended for competitive methods:
pip install ultrack motile laptrack
```

On Kaggle, attach this repository as a resource, then install the helper package:

```bash
pip install /kaggle/input/<resource-name>
```

Import shared helpers from notebook cells:

```python
from cell_tracking import build_submission, load_zarr_timepoint, read_geff_graph
```

## 📐 Data Format

| Item         | Format                     | Shape                                         |
| ------------ | -------------------------- | --------------------------------------------- |
| Image volume | `.zarr` v3                 | `(T, Z, Y, X)` uint16, ~`(100, 64, 256, 256)` |
| Ground truth | `.geff` (Zarr-based graph) | sparse nodes + edges                          |
| Submission   | `.csv`                     | node rows + edge rows                         |

Physical voxel scale: z=1.625, y=0.40625, x=0.40625 µm/voxel.

**Submission CSV columns:** `id, dataset, row_type, node_id, t, z, y, x, source_id, target_id`

- Node row: `row_type=node`, fill `node_id, t, z, y, x`, set `source_id=target_id=-1`
- Edge row: `row_type=edge`, fill `source_id, target_id`, set `node_id=t=z=y=x=-1`

## 🔗 Useful Resources

- [Official competition helper library](https://github.com/royerlab/kaggle-cell-tracking-competition) – baseline training, inference, metric computation
- [geff](https://github.com/live-image-tracking-tools/geff) – graph exchange format over Zarr
- [tracksdata](https://github.com/royerlab/tracksdata) – tracking graph utilities
- [ultrack](https://github.com/royerlab/ultrack) – competitive cell tracking solver
- [motile](https://github.com/funkelab/motile) – ILP-based tracking with custom costs
- [laptrack](https://github.com/yfukai/laptrack) – LAP-based tracking
- [trackastra](https://github.com/weigertlab/trackastra) – transformer-based tracking
- [Cell Tracking Challenge](https://celltrackingchallenge.net/) – field benchmark datasets and SOTA methods

## ⚙️ Development Checks

```bash
uv run --group dev ruff check .
uv run --group dev pytest -q
uv build
```

For a full local pre-commit pass:

```bash
uv run --group dev pre-commit run --all-files
```

## 📋 Competition Timeline

| Date               | Milestone                    |
| ------------------ | ---------------------------- |
| June 29, 2026      | Start Date                   |
| September 22, 2026 | Entry & Team Merger Deadline |
| September 29, 2026 | Final Submission Deadline    |

Prizes: $60,000 total (1st: $18k, 2nd: $12k, 3rd: $8k, …).

______________________________________________________________________

> **Template origin:** forked from [`Borda/kaggle_sandbox`](https://github.com/Borda/kaggle_sandbox) – a minimal Python 3.10+ Kaggle project template with installable helper package, pre-commit, ruff, pytest, and CI.
> See the sandbox for reusable scaffolding and prior competition showcases.
