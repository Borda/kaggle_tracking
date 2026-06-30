# %% [markdown]
# # Biohub - Cell Tracking During Development: EDA + Baseline

# **Competition:** [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
# **Task:** Detect zebrafish cells in 3D+time fluorescence microscopy, link across frames, and identify divisions. Output: node rows (centroids) + edge rows (links).

# **Metric:** Edge Jaccard + Division Jaccard (higher = better). **Baseline strategy:**
# 1. Anisotropy correction (block-mean XY/4 → isotropic ~1.625 µm/voxel grid)
# 2. Peak detection on isotropic volume (Gaussian smooth + Otsu + peak_local_max)
# 3. Frame-pair Hungarian linking in physical µm space (max 12 µm)
# 4. Division post-processing (unmatched daughters → find eligible parent)

# > All helpers inlined — self-contained, no package imports from `cell_tracking`. Do not modify after Kaggle validation.

# %% [markdown]
# ## Setup
# **Packages used:**
# - `zarr` — reads Zarr v3 volumes (.zarr) and GEFF graph stores (.geff).
# - `blosc2` — direct chunk decompression, ~3× faster than zarr array slicing for single-timepoint loads.
# - `scikit-image` — `peak_local_max` for cell centroid detection after Gaussian smoothing.
# - `scipy` — `linear_sum_assignment` (Hungarian algorithm) for optimal frame-pair assignment.
# - `numpy / pandas / matplotlib / tqdm` — data manipulation, CSV writing, visualisation, progress bars.
# <!-- -->
# **Offline setup** — Kaggle commit/submission runs have no internet access. Download wheels once
# (with internet enabled), then install from the local folder during submission:
# <!-- -->
# **Environment variables** let you override paths for local development without editing code:
# - `KAGGLE_DATA_DIR` — defaults to the standard Kaggle input path for this competition.
# - `KAGGLE_OUTPUT_DIR` — where EDA images and the sanity CSV are written (not the final submission).

# %%
# Step 1 — run once with internet enabled to cache wheels locally:
# !pip download zarr blosc2 scikit-image scipy numpy pandas matplotlib tqdm ipywidgets -d packages/

# %%
# Step 2 — install from local cache (offline-safe); falls back to PyPI if cache missing:
# !pip install --no-index --find-links packages/ zarr blosc2 scikit-image scipy numpy pandas matplotlib tqdm ipywidgets || pip install zarr blosc2 scikit-image scipy numpy pandas matplotlib tqdm ipywidgets

# %%
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
DATA_DIR = Path(os.environ.get("KAGGLE_DATA_DIR", "/kaggle/input/competitions/biohub-cell-tracking-during-development"))
OUTPUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# %% [markdown]
# ## EDA - Dataset Overview
# **Zarr volume — shape `(T, Z, Y, X)`:**
# | Axis | Meaning | Typical size |
# |------|---------|-------------|
# | T | Timepoints (frames in the time series) | 100 |
# | Z | Depth slices (along the optical axis) | 64 |
# | Y | Height — rows of each 2D plane | 256 |
# | X | Width — columns of each 2D plane | 256 |
# dtype: `uint16` (fluorescence intensity, 0–65535). One file per recording.
# Chunk layout: `(1, Z, Y, X)` — one full 3D stack per timepoint chunk, stored at
# `0/c/{t}/0/0/0`. Reading timepoint `t` = reading exactly one file from disk. **GEFF graph — shape of key arrays:**
# | Path in .geff | Shape | Contents |
# |---------------|-------|----------|
# | `nodes/ids` | `(N,)` | integer node ID per detected cell |
# | `nodes/props/{t,z,y,x}/values` | `(N,)` each | centroid coordinates in voxels |
# | `edges/ids` | `(E, 2)` | `[source_id, target_id]` parent→child pairs |
# <!-- -->
# Annotations are **sparse** — not every cell in every frame is labelled.
# Only available for training data; test data has `.zarr` volumes only.
# We first list what's available, then inspect the sample submission to understand the exact output format required.

# %%
def get_volume_shape(zarr_path: Path) -> tuple[int, int, int, int]:
    """Return (T, Z, Y, X) from zarr.json — stdlib only, no zarr import."""
    with (zarr_path / "0" / "zarr.json").open() as f:
        return tuple(json.load(f)["shape"])

# %%
train_zarr = sorted(TRAIN_DIR.glob("*.zarr"))
train_geff = sorted(TRAIN_DIR.glob("*.geff"))
test_zarr = sorted(TEST_DIR.glob("*.zarr"))

assert train_zarr, f"No .zarr files in {TRAIN_DIR}"
assert train_geff, f"No .geff files in {TRAIN_DIR}"

print(f"Train zarr: {len(train_zarr)}, geff: {len(train_geff)}")
print(f"Test  zarr: {len(test_zarr)}")

# %%
from collections import Counter

# Shape audit — count occurrences of each (T, Z, Y, X) per split
for split, paths in [("train", train_zarr), ("test", test_zarr)]:
    if not paths:
        continue
    counts = Counter(get_volume_shape(p) for p in paths)
    print(f"{split} ({len(paths)} volumes):")
    for shape, n in counts.most_common():
        print(f"  {shape}  ×{n}")

# %% [markdown]
# ### Sample submission columns
# Inspect `sample_submission.csv` to confirm column names and sentinel values before
# writing any predictions. A mismatch in column names or types causes a silent score of 0.

# %%
df_sample = pd.read_csv(SAMPLE_SUBMISSION)
print(df_sample.head(6).to_string())
print(f"\n{df_sample['row_type'].value_counts()}")

# %% [markdown]
# ### Voxel scale and the anisotropy problem
# The microscope captures voxels at very different resolutions along each axis:
# | Axis | Resolution |
# |------|-----------|
# | Z    | 1.625 µm/voxel |
# | Y    | 0.406 µm/voxel |
# | X    | 0.406 µm/voxel |
# XY is **4× finer** than Z. If we detect cell centroids directly on the raw volume,
# the cost matrix for linking uses Euclidean distance in *voxel space*, not physical space.
# A centroid error of just 4 voxels in Z = 4 × 1.625 = **6.5 µm** — already close to the
# 7 µm metric matching window. Small detection noise in Z causes missed matches.
# **Fix:** block-average XY by 4 before any detection step. The result is an isotropic
# ~1.625 µm/voxel grid in all three axes. Detection now sees round cells instead of
# pancake-shaped blobs, and centroid errors shrink proportionally.

# %%
# Physical voxel scale (µm/voxel) — raw volume
VOXEL_Z = 1.625
VOXEL_Y = 0.40625
VOXEL_X = 0.40625

# After XY block-mean /4: isotropic ~1.625 µm/voxel everywhere
XY_DOWNSAMPLE = 4

# %% [markdown]
# ### Detection and linking parameters
# - `GAUSS_SIGMA` — light Gaussian smoothing before thresholding; suppresses shot noise without blurring cell boundaries.
# - `THRESH_REL` — relative threshold between background and peak intensity. Otsu splits foreground/background;
#   we use `bg + rel × (peak - bg)` so the threshold adapts per-frame to varying illumination.
# - `MIN_PEAK_DIST` — minimum centroid separation in isotropic voxels. 3 isotropic voxels ≈ 4.9 µm — keeps
#   one detection per cell even in dense regions.
# - `MAX_LINK_DIST_UM` — **12 µm**, not 7 µm. The 7 µm is the *metric matching window* (how far off a
#   prediction can be and still score). The linking budget can be looser; 12 µm handles fast-moving cells
#   without introducing too many false links.
# - `DIV_PARENT_DIST_UM` / `DIV_SISTER_DIST_UM` — gates for division detection (see Division section).

# %%
# Detection on isotropic volume
GAUSS_SIGMA = 1.0    # smoothing before threshold
THRESH_REL = 0.30    # Otsu multiplier: threshold = background + rel*(peak-background)
MIN_PEAK_DIST = 3    # minimum separation in isotropic voxels (~4.9 µm)

# Tracking
MAX_LINK_DIST_UM = 12.0    # Hungarian assignment cutoff (µm)
DIV_PARENT_DIST_UM = 12.0  # max parent->daughter distance for division
DIV_SISTER_DIST_UM = 7.0   # max sister-sister distance for division

print(f"Train: {TRAIN_DIR.exists()}, Test: {TEST_DIR.exists()}")

# %% [markdown]
# ## Helpers
# All pipeline helpers are **inlined here** — no imports from the `cell_tracking` package.
# This keeps the notebook fully self-contained for Kaggle submission.
# After validating this notebook on Kaggle, reusable helpers can be distilled into
# `src/cell_tracking/` and imported in subsequent notebooks (*snowball* pattern).
# Never modify this baseline after it produces a validated score.

# %% [markdown]
# ### Data loading
# **Zarr v3 layout:** `<dataset>.zarr/0/c/<t>/0/0/0` — one chunk per timepoint.
# The entire volume fits in one chunk along T, so reading a single timepoint = reading one file.
# **blosc2 direct decompression** is faster than `zarr.open()[t]` because it skips the zarr
# array machinery and reads the compressed chunk bytes directly from disk. The fallback to
# `zarr.open()` handles edge cases (different codecs, missing blosc2 install).

# %%


def load_timepoint_blosc2(zarr_path: Path, t: int) -> np.ndarray:
    """Load one timepoint (Z, Y, X) directly from blosc2 chunk.

    Faster than zarr array slicing - avoids library overhead.
    Chunk path: <zarr>/0/c/<t>/0/0/0
    """
    import blosc2

    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    raw = chunk_path.read_bytes()
    arr = blosc2.decompress(raw)
    return np.frombuffer(arr, dtype=np.uint16).reshape(-1)  # caller reshapes


def load_timepoint(zarr_path: Path, t: int, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Load one timepoint with automatic fallback (blosc2 -> zarr)."""
    try:
        flat = load_timepoint_blosc2(zarr_path, t)
        return flat[: shape_zyx[0] * shape_zyx[1] * shape_zyx[2]].reshape(shape_zyx).astype(np.float32)
    except Exception:
        import zarr
        store = zarr.open(str(zarr_path), mode="r")
        return store["0"][t].astype(np.float32)


# %% [markdown]
# ### Anisotropy correction
# `make_isotropic` reduces XY resolution by a factor of 4 using **block-mean pooling**
# (not stride/decimation). Block-mean averages each 4×4 XY patch, which:
# - preserves signal from dim cells that might be missed by stride sampling
# - reduces shot noise (averaging 16 pixels)
# - produces ~1.625 µm/voxel isotropic grid, matching the Z resolution
# <!-- -->
# All downstream detection operates in this isotropic space.
# Coordinates are converted back to original voxel space after detection.

# %%

def make_isotropic(vol: np.ndarray) -> np.ndarray:
    """Block-average XY by XY_DOWNSAMPLE to get isotropic voxels.

    Raw: z=1.625, y=x=0.406 µm. After /4: z=1.625, y=x=1.625 µm.
    Uses block-mean (not stride) to preserve signal.
    """
    Z, Y, X = vol.shape
    Y2 = Y // XY_DOWNSAMPLE
    X2 = X // XY_DOWNSAMPLE
    vol_crop = vol[:, : Y2 * XY_DOWNSAMPLE, : X2 * XY_DOWNSAMPLE]
    return vol_crop.reshape(Z, Y2, XY_DOWNSAMPLE, X2, XY_DOWNSAMPLE).mean(axis=(2, 4))

# %% [markdown]
# ### Cell detection
# Pipeline per timepoint on the isotropic volume:
# 1. **Gaussian smooth** (`sigma=1.0`) — removes single-pixel noise while preserving cell-scale structures.
# 2. **Otsu threshold** — automatically splits foreground (cells) from background per frame.
#    We use `bg + THRESH_REL × (peak - bg)` rather than the raw Otsu value; this is more robust
#    when cells are bright and the background is near-zero.
# 3. **`peak_local_max`** — finds local intensity maxima above the threshold, with a minimum
#    separation of `MIN_PEAK_DIST` isotropic voxels. Each maximum = one detected cell centroid.
# Detected coordinates are in isotropic space; `detect_cells` converts Y and X back to
# original voxel space by multiplying by `XY_DOWNSAMPLE`.

# %%

from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu


def detect_peaks(vol_iso: np.ndarray) -> np.ndarray:
    """Detect cell centroids on isotropic volume.

    Returns array of shape (N, 3) with columns (z_iso, y_iso, x_iso).
    Coordinates are in isotropic-voxel space (multiply y,x by XY_DOWNSAMPLE to
    get back to original voxel coordinates).
    """
    smoothed = gaussian_filter(vol_iso.astype(np.float32), sigma=GAUSS_SIGMA)
    try:
        thresh = threshold_otsu(smoothed)
    except Exception:
        thresh = smoothed.mean()
    bg = smoothed[smoothed < thresh].mean() if (smoothed < thresh).any() else 0.0
    pk = smoothed.max()
    cutoff = bg + THRESH_REL * (pk - bg)
    peaks = peak_local_max(smoothed, min_distance=MIN_PEAK_DIST, threshold_abs=float(cutoff))
    return peaks  # (N, 3) z_iso, y_iso, x_iso


def detect_cells(zarr_path: Path) -> list[dict]:
    """Detect cells in all timepoints of one zarr volume.

    Returns list of dicts: {t, z, y, x} in *original* voxel coordinates.
    """
    from tqdm.auto import tqdm

    T, Z, Y, X = get_volume_shape(zarr_path)
    detections = []
    for t in tqdm(range(T), desc=zarr_path.stem[:20], leave=False):
        vol = load_timepoint(zarr_path, t, (Z, Y, X))
        vol_iso = make_isotropic(vol)
        peaks = detect_peaks(vol_iso)
        for p in peaks:
            # Scale isotropic y,x back to original voxel space
            detections.append({
                "t": t,
                "z": int(p[0]),
                "y": int(p[1]) * XY_DOWNSAMPLE,
                "x": int(p[2]) * XY_DOWNSAMPLE,
            })
    return detections

# %% [markdown]
# ### Frame-pair linking — Hungarian algorithm
# We link detections between consecutive timepoints using the **Hungarian algorithm**
# (`scipy.optimize.linear_sum_assignment`), which finds the globally optimal one-to-one
# assignment minimising total travel distance. Greedy nearest-neighbour linking can assign
# the same target to multiple sources or miss globally cheaper swaps.
# Cost matrix: physical Euclidean distance in µm using the actual voxel scales
# (`VOXEL_Z`, `VOXEL_Y`, `VOXEL_X`) — **not** voxel-space distance, which would be misleading due to anisotropy.
# Pairs where the optimal assignment distance exceeds `MAX_LINK_DIST_UM` are discarded
# (cell appeared/disappeared or moved too far). This is a greedy per-frame approach;
# global ILP tracking (e.g. `motile`) handles multi-frame occlusions better but requires more setup.

# %%

from scipy.optimize import linear_sum_assignment


def _dist_um(a: dict, b: dict) -> float:
    dz = (a["z"] - b["z"]) * VOXEL_Z
    dy = (a["y"] - b["y"]) * VOXEL_Y
    dx = (a["x"] - b["x"]) * VOXEL_X
    return float((dz**2 + dy**2 + dx**2) ** 0.5)


def link_detections(detections: list[dict]) -> list[tuple[int, int]]:
    """Hungarian nearest-neighbour linking in physical µm space.

    Returns list of (source_idx, target_idx) pairs referencing positions in
    detections.
    """
    if not detections:
        return []
    by_t: dict[int, list[tuple[int, dict]]] = {}
    for idx, det in enumerate(detections):
        by_t.setdefault(det["t"], []).append((idx, det))

    edges: list[tuple[int, int]] = []
    for t_cur in sorted(by_t)[:-1]:
        t_nxt = t_cur + 1
        if t_nxt not in by_t:
            continue
        srcs = by_t[t_cur]
        tgts = by_t[t_nxt]
        cost = np.array([[_dist_um(s, td) for _, td in tgts] for _, s in srcs], dtype=np.float32)
        row_ind, col_ind = linear_sum_assignment(cost)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] <= MAX_LINK_DIST_UM:
                edges.append((srcs[ri][0], tgts[ci][0]))
    return edges

# %% [markdown]
# ### Division detection
# The official baseline ignores cell divisions entirely, leaving Division Jaccard = 0.
# A simple post-processing pass recovers most of those points nearly for free.
# **Strategy** — after frame-pair linking, for each *unmatched* detection in frame `t+1`
# (no incoming edge assigned by the Hungarian step):
# 1. Find all nodes in frame `t` that already have **exactly one** outgoing edge (i.e. one
#    child) — these are candidate parents that could have divided.
# 2. Of those, keep only candidates within `DIV_PARENT_DIST_UM` of the unmatched daughter.
# 3. Additionally require the two sisters (existing child + new daughter) to be within
#    `DIV_SISTER_DIST_UM` of each other — divisions produce two nearby daughters.
# 4. Assign the closest qualifying candidate as the parent; add a second outgoing edge.
# This adds Division Jaccard without touching the node/edge detection, at the cost of
# a second pass over the detections (milliseconds).

# %%

def detect_divisions(
    detections: list[dict],
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Add division edges for unmatched daughters.

    Strategy: for each detection in frame t+1 with no incoming edge,
    find a node in frame t that already has exactly one outgoing edge
    and is within DIV_PARENT_DIST_UM. Both sisters must be within
    DIV_SISTER_DIST_UM of each other.

    Returns additional (source_idx, target_idx) division edges to append.
    """
    if not detections or not edges:
        return []

    edge_set = set(edges)
    targets_with_edge = {ti for _, ti in edge_set}
    by_t: dict[int, list[tuple[int, dict]]] = {}
    for idx, det in enumerate(detections):
        by_t.setdefault(det["t"], []).append((idx, det))

    # children_of[parent_idx] = list of child indices already linked
    from collections import defaultdict
    children_of: dict[int, list[int]] = defaultdict(list)
    for si, ti in edge_set:
        children_of[si].append(ti)

    div_edges: list[tuple[int, int]] = []

    for t_cur in sorted(by_t)[:-1]:
        t_nxt = t_cur + 1
        if t_nxt not in by_t:
            continue

        # unmatched daughters: nodes in t_nxt with no incoming edge
        unmatched = [(idx, det) for idx, det in by_t[t_nxt] if idx not in targets_with_edge]

        # eligible parents: nodes in t_cur with exactly 1 child
        parents = [(idx, det) for idx, det in by_t[t_cur] if len(children_of[idx]) == 1]

        for d_idx, daughter in unmatched:
            best_parent: tuple[int, dict] | None = None
            best_dist = DIV_PARENT_DIST_UM

            for p_idx, parent in parents:
                dist_pd = _dist_um(parent, daughter)
                if dist_pd > best_dist:
                    continue
                # existing sister
                sister_idx = children_of[p_idx][0]
                sister = detections[sister_idx]
                dist_sisters = _dist_um(daughter, sister)
                if dist_sisters <= DIV_SISTER_DIST_UM:
                    best_parent = (p_idx, parent)
                    best_dist = dist_pd

            if best_parent is not None:
                p_idx = best_parent[0]
                div_edges.append((p_idx, d_idx))
                children_of[p_idx].append(d_idx)
                targets_with_edge.add(d_idx)

    return div_edges

# %% [markdown]
# ### Submission format
# The competition CSV has **two row types** interleaved in the same file:
# | `row_type` | Required fields | Sentinel values |
# |------------|----------------|-----------------|
# | `node`     | `node_id, t, z, y, x` | `source_id = target_id = -1` |
# | `edge`     | `source_id, target_id` | `node_id = t = z = y = x = -1` |
# `source_id` and `target_id` reference `node_id` values in the same submission file,
# so node IDs must be globally unique across all datasets in the file.
# We use a running `offset` counter when concatenating multiple datasets.

# %%

_SUBMISSION_COLS = ("id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")


@dataclass
class NodeRow:
    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int


@dataclass
class EdgeRow:
    dataset: str
    source_id: int
    target_id: int


def build_submission(nodes: list, edges: list, output_path: Path) -> Path:
    """Write nodes + edges to competition submission CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_SUBMISSION_COLS)
        for row_id, n in enumerate(nodes):
            w.writerow([row_id, n.dataset, "node", n.node_id, n.t, n.z, n.y, n.x, -1, -1])
        for row_id, e in enumerate(edges, start=len(nodes)):
            w.writerow([row_id, e.dataset, "edge", -1, -1, -1, -1, -1, e.source_id, e.target_id])
    return output_path

# %% [markdown]
# ## EDA - 3D Slice Viewer
# Interactive viewer showing three orthogonal slices through the volume simultaneously, plus a 3D diagram
# of the cutting planes. Crosshair lines on each panel show where the other two planes intersect.
# <!-- -->
# Run the notebook interactively on Kaggle to use sliders. In batch/commit mode the viewer falls back
# to static middle-slice snapshots saved as PNG.
# <!-- -->
# **Why three planes:** a single XY slice hides anisotropy. Browsing XZ or YZ slices reveals the
# difference between raw (cells appear elongated in Z) and isotropic (cells appear round everywhere).

# %%
from ipywidgets import interact, IntSlider


def show_volume(vol: np.ndarray, z: int, y: int, x: int, title: str = "", fig_size: tuple = (14, 12), save_path: Path | None = None) -> None:
    """Show three orthogonal slices through vol at (z, y, x) plus a 3D cutting-plane diagram.

    Args:
        vol: 3D float array shape (Z, Y, X).
        z: Z-index for axial (XY) slice.
        y: Y-index for coronal (XZ) slice.
        x: X-index for sagittal (YZ) slice.
        title: Figure suptitle.
        fig_size: Figure size in inches.
        save_path: If given, save figure to this path before showing.
    """
    vz, vy, vx = vol.shape
    fig = plt.figure(figsize=fig_size)
    ax00 = fig.add_subplot(2, 2, 1)
    ax01 = fig.add_subplot(2, 2, 2)
    ax10 = fig.add_subplot(2, 2, 3)
    ax11 = fig.add_subplot(2, 2, 4, projection="3d")
    kw = dict(cmap="gray", interpolation="nearest")

    ax00.imshow(vol[z], **kw)
    ax00.axvline(x=x, color="g", linewidth=1.5, alpha=0.8, label=f"x={x} (sagittal)")
    ax00.axhline(y=y, color="b", linewidth=1.5, alpha=0.8, label=f"y={y} (coronal)")
    ax00.set_title(f"Axial XY  z={z}")
    ax00.set_xlabel("X (voxel)")
    ax00.set_ylabel("Y (voxel)")
    ax00.legend(fontsize=8, loc="upper right")
    ax00.grid(True, alpha=0.2, color="white", linewidth=0.5)

    ax01.imshow(vol[:, :, x].T, **kw)
    ax01.axvline(x=z, color="r", linewidth=1.5, alpha=0.8, label=f"z={z} (axial)")
    ax01.axhline(y=y, color="b", linewidth=1.5, alpha=0.8, label=f"y={y} (coronal)")
    ax01.set_title(f"Sagittal YZ  x={x}")
    ax01.set_xlabel("Z (voxel)")
    ax01.set_ylabel("Y (voxel)")
    ax01.legend(fontsize=8, loc="upper right")
    ax01.grid(True, alpha=0.2, color="white", linewidth=0.5)

    ax10.imshow(vol[:, y, :], **kw, aspect="auto")
    ax10.axvline(x=x, color="g", linewidth=1.5, alpha=0.8, label=f"x={x} (sagittal)")
    ax10.axhline(y=z, color="r", linewidth=1.5, alpha=0.8, label=f"z={z} (axial)")
    ax10.set_title(f"Coronal XZ  y={y}")
    ax10.set_xlabel("X (voxel)")
    ax10.set_ylabel("Z (voxel)")
    ax10.legend(fontsize=8, loc="upper right")
    ax10.grid(True, alpha=0.2, color="white", linewidth=0.5)

    Y_g, X_g = np.meshgrid(np.arange(vy), np.arange(vx), indexing="ij")
    Z_g, X_g2 = np.meshgrid(np.arange(vz), np.arange(vx), indexing="ij")
    Z_g2, Y_g2 = np.meshgrid(np.arange(vz), np.arange(vy), indexing="ij")
    ax11.plot_surface(X_g, Y_g, np.full_like(X_g, z, dtype=float), color="r", alpha=0.25)
    ax11.plot_surface(X_g2, np.full_like(X_g2, y, dtype=float), Z_g, color="b", alpha=0.25)
    ax11.plot_surface(np.full_like(Y_g2, x, dtype=float), Y_g2, Z_g2, color="g", alpha=0.25)
    ax11.set_xlabel("X")
    ax11.set_ylabel("Y")
    ax11.set_zlabel("Z")
    ax11.set_xlim([0, vx])
    ax11.set_ylim([0, vy])
    ax11.set_zlim([0, vz])
    ax11.view_init(elev=20, azim=45)
    import matplotlib.patches as mpatches
    ax11.legend(
        handles=[mpatches.Patch(color="r", label=f"axial z={z}"), mpatches.Patch(color="b", label=f"coronal y={y}"), mpatches.Patch(color="g", label=f"sagittal x={x}")],
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=80)
    plt.show()


def interactive_show(vol: np.ndarray, title: str = "", save_path: Path | None = None) -> None:
    """Interactive 3D slice viewer; falls back to static middle slices in batch/commit mode.

    Args:
        vol: 3D float array shape (Z, Y, X).
        title: Viewer title passed to show_volume.
        save_path: Saved only in batch/commit mode (interactive mode never saves).
    """
    vz, vy, vx = vol.shape
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive") == "Interactive":
        interact(
            lambda z, y, x: show_volume(vol, z, y, x, title=title),
            z=IntSlider(min=0, max=vz - 1, step=1, value=vz // 2, description="Z-slice"),
            y=IntSlider(min=0, max=vy - 1, step=1, value=vy // 2, description="Y-slice"),
            x=IntSlider(min=0, max=vx - 1, step=1, value=vx // 2, description="X-slice"),
        )
    else:
        show_volume(vol, vz // 2, vy // 2, vx // 2, title=f"{title} [batch — middle slices]", save_path=save_path)


# %%
T, Z, Y, X = get_volume_shape(train_zarr[0])
t_mid, z_mid = T // 2, Z // 2
vol_raw = load_timepoint(train_zarr[0], t_mid, (Z, Y, X))
vol_iso = make_isotropic(vol_raw)
print(f"Raw shape: {vol_raw.shape},  Isotropic shape: {vol_iso.shape}")

# %%
interactive_show(vol_raw, title=f"Raw volume  t={t_mid}", save_path=OUTPUT_DIR / "eda_raw.png")

# %%
interactive_show(vol_iso, title=f"Isotropic volume  t={t_mid}", save_path=OUTPUT_DIR / "eda_isotropic.png")

# %% [markdown]
# ## EDA - Ground Truth Graph
# The `.geff` file stores the ground-truth lineage as a sparse graph:
# - `nodes/ids` — integer ID per cell instance
# - `nodes/props/{t,z,y,x}/values` — spatial coordinates per node
# - `edges/ids` — pairs of node IDs representing parent→child links (including divisions)
# We visualise the distribution of GT nodes over time and Z depth to understand:
# - **Time distribution:** are all timepoints annotated, or only a subset?
# - **Z distribution:** are cells spread throughout the volume or concentrated at certain depths?
#   Concentrated Z distributions can bias detection thresholds.

# %%
import zarr
g = zarr.open(str(train_geff[0]), mode="r")
node_ids = np.asarray(g["nodes/ids"])
node_t = np.asarray(g["nodes/props/t/values"])
node_z = np.asarray(g["nodes/props/z/values"])
edge_ids = np.asarray(g["edges/ids"])
print(f"GT nodes: {len(node_ids)}, edges: {len(edge_ids)}")
print(f"Annotated timepoints: {np.unique(node_t)}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(node_t, bins=30, color="steelblue")
axes[0].set_title("GT nodes per timepoint")
axes[0].set_xlabel("Timepoint")
axes[0].set_ylabel("Node count")
axes[0].grid(True, alpha=0.3)
axes[1].hist(node_z, bins=30, color="coral")
axes[1].set_title("GT node z-depth distribution")
axes[1].set_xlabel("Z depth (voxels)")
axes[1].set_ylabel("Node count")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_gt_distribution.png", dpi=80)
plt.show()

# %% [markdown]
# ## EDA - Detection Sanity Check (one timepoint)
# Before running the full pipeline we verify that `detect_peaks` finds a plausible number
# of cells at a single mid-timepoint on the isotropic volume.
# We overlay detected peak positions (red dots) on the isotropic XY slice at `z=z_mid`.
# Because peaks span all Z layers, we show only peaks within ±2 isotropic voxels of `z_mid` to avoid overplotting.
# **What to check:**
# - Peaks align with bright spots in the image (not background noise).
# - Cell count is in a plausible range (typically hundreds per timepoint for zebrafish data).
# - No large clusters of spurious detections in empty regions.

# %%
peaks = detect_peaks(vol_iso)
print(f"Detected {len(peaks)} peaks at t={t_mid}")

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(vol_iso[z_mid], cmap="gray")
near = peaks[np.abs(peaks[:, 0] - z_mid) < 2]
ax.scatter(near[:, 2], near[:, 1], s=6, c="red", alpha=0.7, label=f"Detected peaks (n={len(near)})")
ax.set_title(f"Isotropic peaks at t={t_mid} z={z_mid}")
ax.set_xlabel("X (isotropic px)")
ax.set_ylabel("Y (isotropic px)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, color="white", linewidth=0.5)
plt.savefig(OUTPUT_DIR / "eda_peaks.png", dpi=80)
plt.show()

# %% [markdown]
# ## Sanity Check: One Train Sample (detection + linking + division)
# We run the full pipeline on one training sample before touching test data.
# Training data has ground-truth `.geff` files, so we can manually compare
# detected node counts and edge counts against GT values printed in the EDA section above.
# We also write a `sanity_submission.csv` and read it back to verify the CSV format is valid
# (correct columns, sentinel values, row counts) before the full test run.

# %%
def process_sample(zarr_path: Path, offset: int = 0) -> tuple[list[NodeRow], list[EdgeRow]]:
    """Run full detection + linking + division pipeline on one zarr volume.

    Args:
        zarr_path: Path to the .zarr directory.
        offset: Node ID offset — add to all local node IDs for global uniqueness across datasets.

    Returns:
        Tuple of (nodes, edges) ready for build_submission.
    """
    dataset_id = zarr_path.stem
    dets = detect_cells(zarr_path)
    links = link_detections(dets)
    div_links = detect_divisions(dets, links)
    all_links = links + div_links
    nodes = [NodeRow(dataset=dataset_id, node_id=offset + i + 1, t=d["t"], z=d["z"], y=d["y"], x=d["x"]) for i, d in enumerate(dets)]
    edges = [EdgeRow(dataset=dataset_id, source_id=offset + si + 1, target_id=offset + ti + 1) for si, ti in all_links]
    print(f"  {dataset_id}: {len(dets)} detections, {len(links)} links, {len(div_links)} division links")
    return nodes, edges

# %%
nodes, edges = process_sample(train_zarr[0])
sanity = build_submission(nodes, edges, OUTPUT_DIR / "sanity_submission.csv")
print(pd.read_csv(sanity)["row_type"].value_counts().to_string())

# %% [markdown]
# ## Full Run: All Test Samples
# Process every test zarr in sequence. `tqdm` shows per-sample progress.
# **Node ID offset:** each test dataset shares the same submission file, so `node_id` must
# be globally unique. We track a running `offset` = total nodes inserted so far and add it
# to each new dataset's local IDs. Edge `source_id`/`target_id` reference these global IDs.

# %%
from tqdm.auto import tqdm

all_nodes: list[NodeRow] = []
all_edges: list[EdgeRow] = []

for zarr_path in tqdm(sorted(TEST_DIR.glob("*.zarr")), desc="Test samples"):
    nodes, edges = process_sample(zarr_path, offset=len(all_nodes))
    all_nodes += nodes
    all_edges += edges

print(f"Total nodes: {len(all_nodes)}, edges: {len(all_edges)}")

# %% [markdown]
# ## Write submission.csv
# Write the final submission to the working directory (`submission.csv`).
# On Kaggle, the working directory is `/kaggle/working/` — the standard location the
# platform reads when you submit output.
# If no test data is found (e.g. running locally without the dataset), we fall back to
# copying the sample submission so the notebook still produces a valid output file.
# After writing, print the row-type breakdown and peek at the first few lines to confirm
# the file looks correct before submitting.

# %%
submission_path = Path("submission.csv")

if all_nodes or all_edges:
    build_submission(all_nodes, all_edges, submission_path)
    df_sub = pd.read_csv(submission_path)
    print(df_sub["row_type"].value_counts())
else:
    import shutil
    shutil.copyfile(SAMPLE_SUBMISSION, submission_path)
    print("No test data - copied sample_submission.csv")

# %%
# ! head -5 {submission_path}
# ! wc -l {submission_path}
