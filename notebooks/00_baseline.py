# %% [markdown]
# # Biohub - Cell Tracking During Development: EDA + Baseline

# **Competition:** [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
# **Task:** Detect zebrafish cells in 3D+time fluorescence microscopy, link across frames, and identify divisions. Output: node rows (centroids) + edge rows (links).

# **Metric:** Edge Jaccard + Division Jaccard (higher = better). **Node over-prediction is penalised.**
# **Baseline strategy (DoG + calibrated count, ~0.73 LB):**
# 1. Anisotropy correction (block-mean XY/4 → isotropic ~1.625 µm/voxel grid)
# 2. **DoG band-pass detection** — multi-scale Difference of Gaussians; recovers dim deep cells that Gaussian+Otsu misses
# 3. **COM refinement** — intensity-weighted centroid in original anisotropic space; sub-voxel float accuracy
# 4. **Physical NMS** — cKDTree dedup in µm space; removes cross-scale duplicate peaks
# 5. **Per-sample count calibration** — learn generous/estimated ratio on train; apply as per-movie topk budget on test
# 6. Two-pass Hungarian linking with velocity prediction (tight gate → full gate for leftovers)
# 7. **Isolated node pruning** — remove unlinked detections (almost all FPs)

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
# **DoG detection** — multi-scale band-pass filter:
# - `DOG_SIGMAS` — three σ values; DoG at each scale catches cells of different apparent sizes.
# - `DOG_K` — ratio between the two Gaussians; 1.6 is the standard LoG approximation factor.
# - `DOG_THR_PCT` — keep peaks above this percentile of positive DoG response per scale (strict).
# - `GENEROUS_DOG_PCT` — permissive threshold used only during count calibration pass.
# - `NMS_RADIUS_UM` — physical-space dedup radius; 4 µm keeps one peak per ~8 µm diameter cell.
#
# **Count calibration** — prevents over-prediction penalty:
# - `BUDGET_SAFETY` — multiply estimated count by this factor (1.15 = 15% headroom above estimate).
#
# **Linking** — two-pass Hungarian:
# - `TIGHT_GATE_UM` / `MAX_LINK_UM` — pass-1 (velocity-predicted) and pass-2 (fallback) gates.
# - `MOTION_FRAC` — weight for velocity extrapolation; 0.5 × prev_velocity = half-step prediction.

# %%
# Scale vector (µm/voxel) — used everywhere for physical distance calculations
SCALE_ZYX = np.array([VOXEL_Z, VOXEL_Y, VOXEL_X])

# DoG detection
DOG_SIGMAS = (1.0, 1.8, 3.0)   # σ values for multi-scale DoG
DOG_K = 1.6                      # σ_high = σ_low * DOG_K
DOG_THR_PCT = 80.0               # strict threshold: pct of positive DoG response
GENEROUS_DOG_PCT = 55.0          # permissive threshold for calibration pass
REFINE_RZ = 2                    # COM half-window in Z (original voxels, ~3.25 µm)
REFINE_RYX = 5                   # COM half-window in Y/X (original voxels, ~2 µm)
MIN_PEAK_DIST = 2                # min separation in isotropic voxels
NMS_RADIUS_UM = 4.0              # physical NMS dedup radius (µm)

# Linking
MAX_LINK_UM = 10.0               # full-gate cutoff (µm)
TIGHT_GATE_UM = 6.0              # pass-1 tight gate (µm)
USE_MOTION = True                # enable velocity prediction
MOTION_FRAC = 0.5                # velocity extrapolation weight

# Division (off by default — FP divisions cost edge FP + division FP; rare cells)
DETECT_DIV = False
DIV_PARENT_DIST_UM = 12.0
DIV_SISTER_DIST_UM = 7.0

# Post-processing
PRUNE_ISOLATED = True            # remove nodes with no edges (almost all FPs)

# Count calibration
USE_COUNT_CALIBRATION = False
BUDGET_SAFETY = 1.15             # safety factor above estimated count

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
# ### Cell detection — DoG band-pass pipeline
# **Why DoG instead of Gaussian+Otsu:**
# Gaussian+Otsu sets a global threshold per frame. Cells deep in the tissue are dim; in a frame
# where bright cells dominate, dim cells fall below the global threshold and are missed entirely.
# DoG (σ_low − σ_high) is a **band-pass filter** that removes the slowly-varying background,
# allowing dim cells to be detected by their *local contrast*, not their absolute intensity.
# Using three DoG scales (σ = 1.0, 1.8, 3.0 isotropic voxels) catches cells of different apparent
# sizes; peaks are unioned then deduplicated via physical NMS.
#
# **Coordinate flow:**
# 1. Detect peaks on isotropic (pooled) volume → integer iso-coords
# 2. Scale back to original: y = y_iso × 4 + 1.5 (block center, not edge)
# 3. Refine with intensity-weighted COM in original anisotropic volume
# 4. Output: float (z, y, x) in original voxel space

# %%

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max


def _dog_scale_back(pk_iso: np.ndarray) -> np.ndarray:
    """Convert isotropic (pooled) peak coords to original-voxel float coords.

    Places y, x at the CENTER of the XY_DOWNSAMPLE block (not its edge).
    Z is unchanged (not pooled).

    Args:
        pk_iso: Integer peaks (N, 3) in isotropic voxel space (z, y_iso, x_iso).

    Returns:
        Float array (N, 3) in original voxel space (z, y_orig, x_orig).
    """
    out = pk_iso.astype(float)
    out[:, 1] = out[:, 1] * XY_DOWNSAMPLE + (XY_DOWNSAMPLE - 1) / 2.0
    out[:, 2] = out[:, 2] * XY_DOWNSAMPLE + (XY_DOWNSAMPLE - 1) / 2.0
    return out


def _com_refine_orig(vol: np.ndarray, zyx: np.ndarray) -> np.ndarray:
    """Intensity-weighted COM in original (anisotropic) volume space.

    Uses asymmetric window — REFINE_RZ in Z, REFINE_RYX in Y/X — which is
    roughly spherical in physical µm (~3.25 µm Z, ~2 µm Y/X radius).

    Args:
        vol: Raw float32 volume (Z, Y, X) in original voxel space.
        zyx: Float initial position (3,) in original voxel coordinates.

    Returns:
        Refined float position (3,) in original voxel coordinates.
    """
    Z, Y, X = vol.shape
    z, y, x = int(round(zyx[0])), int(round(zyx[1])), int(round(zyx[2]))
    z0, z1 = max(0, z - REFINE_RZ), min(Z, z + REFINE_RZ + 1)
    y0, y1 = max(0, y - REFINE_RYX), min(Y, y + REFINE_RYX + 1)
    x0, x1 = max(0, x - REFINE_RYX), min(X, x + REFINE_RYX + 1)
    patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
    w = np.maximum(patch - float(patch.min()), 0.0)
    s = float(w.sum())
    if s < 1e-12:
        return zyx.copy()
    zg, yg, xg = np.mgrid[z0:z1, y0:y1, x0:x1]
    return np.array([(zg * w).sum(), (yg * w).sum(), (xg * w).sum()]) / s


def _nms_physical(coords: np.ndarray, scores: np.ndarray, radius_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Non-maximum suppression in physical µm space via cKDTree.

    Suppresses all lower-score peaks within radius_um of a higher-score peak.

    Args:
        coords: Float peak positions (N, 3) in original voxel space (z, y, x).
        scores: Response scores (N,) — higher is better.
        radius_um: Suppression radius in physical µm.

    Returns:
        Tuple of (kept_coords, kept_scores) after NMS.
    """
    if len(coords) <= 1:
        return coords, scores
    pts = coords * SCALE_ZYX[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    killed = np.zeros(len(coords), bool)
    keep: list[int] = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(pts[i], r=radius_um)] = True
    k = np.array(keep)
    return coords[k], scores[k]


def detect_peaks_dog(
    vol: np.ndarray, dog_thr_pct: float = DOG_THR_PCT, topk: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale DoG detection on a raw anisotropic volume.

    Pools XY by XY_DOWNSAMPLE (isotropic), runs DoG at each σ in DOG_SIGMAS,
    unions all scale peaks, refines centroids in original space, then
    deduplicates via physical NMS.

    Args:
        vol: Raw float32 volume (Z, Y, X) in original voxel space.
        dog_thr_pct: Percentile of positive DoG values used as threshold.
        topk: If set, keep only the top-k peaks by normalised DoG response.

    Returns:
        Tuple of (coords (N, 3) float original voxels, scores (N,) float).
    """
    pooled = make_isotropic(vol)
    all_coords: list[np.ndarray] = []
    all_scores: list[float] = []

    for sigma in DOG_SIGMAS:
        dog = gaussian_filter(pooled, sigma) - gaussian_filter(pooled, sigma * DOG_K)
        pos_vals = dog[dog > 0]
        if pos_vals.size == 0:
            continue
        thr = float(np.percentile(pos_vals, dog_thr_pct))
        iso_peaks = peak_local_max(dog, min_distance=MIN_PEAK_DIST, threshold_abs=thr, exclude_border=False)
        if len(iso_peaks) == 0:
            continue
        resp = dog[iso_peaks[:, 0], iso_peaks[:, 1], iso_peaks[:, 2]].astype(float)
        resp = resp / max(float(resp.max()), 1e-6)  # per-scale normalise
        orig_init = _dog_scale_back(iso_peaks)
        for p, r in zip(orig_init, resp):
            all_coords.append(_com_refine_orig(vol, p))
            all_scores.append(float(r))

    if not all_coords:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)

    coords = np.array(all_coords)
    scores = np.array(all_scores)
    coords, scores = _nms_physical(coords, scores, NMS_RADIUS_UM)

    if topk is not None and len(coords) > topk:
        best = np.argsort(-scores)[: int(topk)]
        coords, scores = coords[best], scores[best]

    return coords, scores


def detect_cells(
    zarr_path: Path, dog_thr_pct: float | None = None, topk_per_frame: int | None = None
) -> list[dict]:
    """Detect cells via DoG across all timepoints of one zarr volume.

    Args:
        zarr_path: Path to .zarr directory.
        dog_thr_pct: Detection threshold percentile (default: DOG_THR_PCT).
        topk_per_frame: If set, keep only top-k detections per frame.

    Returns:
        List of dicts with keys t, z, y, x (float, original voxel space), score.
    """
    from tqdm.auto import tqdm

    pct = dog_thr_pct if dog_thr_pct is not None else DOG_THR_PCT
    T, Z, Y, X = get_volume_shape(zarr_path)
    detections: list[dict] = []
    for t in tqdm(range(T), desc=zarr_path.stem[:20], leave=False):
        vol = load_timepoint(zarr_path, t, (Z, Y, X))
        coords, scores = detect_peaks_dog(vol, dog_thr_pct=pct, topk=topk_per_frame)
        for c, s in zip(coords, scores):
            detections.append({"t": t, "z": float(c[0]), "y": float(c[1]), "x": float(c[2]), "score": float(s)})
    return detections

# %% [markdown]
# ### Frame-pair linking — two-pass Hungarian with velocity prediction
# **Pass 1** uses half the previous-frame velocity to predict where each cell will be, then runs
# Hungarian within `TIGHT_GATE_UM` of those predicted positions. Confident, fast-moving cells
# are committed first.
# **Pass 2** takes the remaining unmatched nodes from both frames and runs Hungarian at the full
# `MAX_LINK_UM` gate — catches slow-moving and newly appearing/disappearing cells.
# Velocity (`prev_vel`) is updated after each frame pair using all matched links.

# %%

from scipy.optimize import linear_sum_assignment


def _det_um(det: dict) -> np.ndarray:
    """Convert detection dict to physical µm coords (z, y, x)."""
    return np.array([det["z"] * VOXEL_Z, det["y"] * VOXEL_Y, det["x"] * VOXEL_X])


def _dist_um(a: dict, b: dict) -> float:
    """Physical µm distance between two detection dicts."""
    return float(np.linalg.norm(_det_um(a) - _det_um(b)))


def link_detections(detections: list[dict]) -> list[tuple[int, int]]:
    """Two-pass Hungarian linking with velocity-prediction motion model.

    Pass 1: tight gate (TIGHT_GATE_UM) on velocity-extrapolated positions.
    Pass 2: full gate (MAX_LINK_UM) on remaining unmatched nodes.

    Args:
        detections: List of dicts with keys t, z, y, x in original voxel space.

    Returns:
        List of (source_global_idx, target_global_idx) pairs.
    """
    if not detections:
        return []

    by_t: dict[int, list[tuple[int, dict]]] = {}
    for idx, det in enumerate(detections):
        by_t.setdefault(det["t"], []).append((idx, det))

    edges: list[tuple[int, int]] = []
    prev_vel: dict[int, np.ndarray] = {}  # global_det_idx -> velocity in µm

    for t_cur in sorted(by_t)[:-1]:
        t_nxt = t_cur + 1
        if t_nxt not in by_t:
            continue

        srcs = by_t[t_cur]
        tgts = by_t[t_nxt]

        src_xyz = np.array([_det_um(d) for _, d in srcs])   # (N, 3)
        tgt_xyz = np.array([_det_um(d) for _, d in tgts])   # (M, 3)
        src_gidx = np.array([i for i, _ in srcs])
        tgt_gidx = np.array([i for i, _ in tgts])
        s_g2l = {int(src_gidx[r]): r for r in range(len(srcs))}
        t_g2l = {int(tgt_gidx[c]): c for c in range(len(tgts))}

        # velocity-predicted positions for pass 1
        src_pred = src_xyz.copy()
        if USE_MOTION:
            for ri, (si, _) in enumerate(srcs):
                if si in prev_vel:
                    src_pred[ri] = src_xyz[ri] + MOTION_FRAC * prev_vel[si]

        # pass 1 — gate on RAW distance, optimize PREDICTED distance (competitor approach).
        # Prevents spurious links from erroneous velocity prediction while still using
        # velocity to rank within the gate.
        BIG = 1e9
        raw1 = np.linalg.norm(src_xyz[:, None] - tgt_xyz[None], axis=2).astype(np.float32)
        pred1 = np.linalg.norm(src_pred[:, None] - tgt_xyz[None], axis=2).astype(np.float32)
        cost1 = np.where(raw1 > TIGHT_GATE_UM, BIG, pred1)
        r1, c1 = linear_sum_assignment(cost1)
        frame_edges: list[tuple[int, int]] = []
        matched_s: set[int] = set()
        matched_t: set[int] = set()
        for ri, ci in zip(r1, c1):
            if cost1[ri, ci] < BIG:
                frame_edges.append((int(src_gidx[ri]), int(tgt_gidx[ci])))
                matched_s.add(ri)
                matched_t.add(ci)

        # pass 2 — full gate on leftovers; still optimize predicted distance.
        fp = [r for r in range(len(srcs)) if r not in matched_s]
        ft = [c for c in range(len(tgts)) if c not in matched_t]
        if fp and ft:
            s2 = np.array(fp)
            t2 = np.array(ft)
            raw2 = np.linalg.norm(src_xyz[s2][:, None] - tgt_xyz[t2][None], axis=2).astype(np.float32)
            pred2 = np.linalg.norm(src_pred[s2][:, None] - tgt_xyz[t2][None], axis=2).astype(np.float32)
            cost2 = np.where(raw2 > MAX_LINK_UM, BIG, pred2)
            r2, c2 = linear_sum_assignment(cost2)
            for ri2, ci2 in zip(r2, c2):
                if cost2[ri2, ci2] < BIG:
                    frame_edges.append((int(src_gidx[s2[ri2]]), int(tgt_gidx[t2[ci2]])))

        # update velocities for all matched pairs this frame
        for si, ti in frame_edges:
            prev_vel[ti] = tgt_xyz[t_g2l[ti]] - src_xyz[s_g2l[si]]

        edges.extend(frame_edges)

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
    z: float   # float COM coords — competition metric accepts sub-voxel
    y: float
    x: float


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
# ## EDA - 4D Slice Viewer (T + Z + Y + X)
# Interactive viewer with four sliders: **T** (timepoint), **Z**, **Y**, **X**.
# Scrubbing T reloads the volume on the fly from disk — no need to pre-load the whole sequence.
# Shows three orthogonal slices plus a 3D cutting-plane diagram.
# <!-- -->
# An optional `transform` callable (e.g. `make_isotropic`) is applied after loading each frame,
# so the same viewer can display raw or isotropic volumes without code duplication.
# <!-- -->
# In batch/commit mode falls back to static middle-frame snapshot saved as PNG.

# %%
import matplotlib.patches as mpatches

from ipywidgets import interact, IntSlider


def show_volume(
    vol: np.ndarray,
    z: int,
    y: int,
    x: int,
    title: str = "",
    fig_size: tuple = (14, 12),
    save_path: Path | None = None,
) -> None:
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
    ax11.legend(
        handles=[
            mpatches.Patch(color="r", label=f"axial z={z}"),
            mpatches.Patch(color="b", label=f"coronal y={y}"),
            mpatches.Patch(color="g", label=f"sagittal x={x}"),
        ],
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=80)
    plt.show()


def interactive_show_4d(
    zarr_path: Path,
    transform=None,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Interactive 4D slice viewer with T + Z + Y + X sliders.

    Loads each timepoint on-demand — the full sequence is never held in memory.
    `transform` is an optional callable applied after loading each frame (e.g. `make_isotropic`).
    Falls back to static middle-frame snapshot in batch/commit mode.

    Args:
        zarr_path: Path to .zarr directory.
        transform: Optional callable (np.ndarray Z,Y,X) -> (np.ndarray Z',Y',X').
        title: Figure suptitle prefix (timepoint appended automatically).
        save_path: Written only in batch mode.
    """
    T_total, Z, Y, X = get_volume_shape(zarr_path)

    def _load_and_show(t: int, z: int, y: int, x: int) -> None:
        vol = load_timepoint(zarr_path, t, (Z, Y, X))
        if transform is not None:
            vol = transform(vol)
        show_volume(vol, z, y, x, title=f"{title}  t={t}")

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive") == "Interactive":
        # Derive slider bounds from (optionally transformed) shape at mid-frame
        vol_mid = load_timepoint(zarr_path, T_total // 2, (Z, Y, X))
        if transform is not None:
            vol_mid = transform(vol_mid)
        vz, vy, vx = vol_mid.shape
        interact(
            _load_and_show,
            t=IntSlider(min=0, max=T_total - 1, step=1, value=T_total // 2, description="T (frame)"),
            z=IntSlider(min=0, max=vz - 1, step=1, value=vz // 2, description="Z-slice"),
            y=IntSlider(min=0, max=vy - 1, step=1, value=vy // 2, description="Y-slice"),
            x=IntSlider(min=0, max=vx - 1, step=1, value=vx // 2, description="X-slice"),
        )
    else:
        vol = load_timepoint(zarr_path, T_total // 2, (Z, Y, X))
        if transform is not None:
            vol = transform(vol)
        vz, vy, vx = vol.shape
        show_volume(
            vol, vz // 2, vy // 2, vx // 2,
            title=f"{title}  t={T_total // 2} [batch — middle frame]",
            save_path=save_path,
        )


# %%
T, Z, Y, X = get_volume_shape(train_zarr[0])
t_mid, z_mid = T // 2, Z // 2
vol_raw = load_timepoint(train_zarr[0], t_mid, (Z, Y, X))
vol_iso = make_isotropic(vol_raw)
print(f"Raw shape: {vol_raw.shape},  Isotropic shape: {vol_iso.shape}")

# %%
interactive_show_4d(train_zarr[0], title="Raw volume", save_path=OUTPUT_DIR / "eda_raw.png")

# %%
interactive_show_4d(train_zarr[0], transform=make_isotropic, title="Isotropic volume", save_path=OUTPUT_DIR / "eda_isotropic.png")

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
# ### Estimated node count calibration
# Each `.geff` metadata (`zarr.json`) stores `estimated_number_of_nodes` — the approximate
# true total cell count across all annotated timepoints.  Comparing this against our
# detection count reveals whether we are over- or under-detecting.  **Over-detection is
# penalised by the metric**; under-detection causes missed edges (FN).  Aim for ≤1.2× the
# estimated count.

# %%
for geff_path in train_geff:
    zarr_path = TRAIN_DIR / (geff_path.stem + ".zarr")
    meta_path = geff_path / "zarr.json"
    if not meta_path.exists():
        continue
    with meta_path.open() as f:
        meta = json.load(f)
    n_est = meta.get("estimated_number_of_nodes", None)
    if n_est is None:
        continue
    T, Z, Y, X = get_volume_shape(zarr_path)
    print(f"{geff_path.stem[:30]}  estimated={n_est}  T={T}  ~{n_est / T:.0f}/frame")

# %% [markdown]
# ## EDA - Detection Sanity Check (one timepoint)
# Verify that DoG detection finds a plausible number of cells at one mid-timepoint.
# Red dots are overlaid on the isotropic XY slice at `z=z_mid`; only peaks within
# ±2 isotropic voxels are shown to avoid overplotting.
# **What to check:**
# - Peaks align with bright spots (not background noise).
# - Cell count is in a plausible range (typically hundreds per timepoint).
# - No large spurious clusters in empty regions.

# %%
# Run DoG detection on the mid-timepoint raw volume
det_coords, det_scores = detect_peaks_dog(vol_raw)
print(f"DoG detected {len(det_coords)} peaks at t={t_mid}")

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(vol_iso[z_mid], cmap="gray")
# coords are in original voxel space — convert to iso for overlay on iso image
near_mask = np.abs(det_coords[:, 0] - z_mid) < 2
near = det_coords[near_mask]
near_iso_y = near[:, 1] / XY_DOWNSAMPLE
near_iso_x = near[:, 2] / XY_DOWNSAMPLE
ax.scatter(near_iso_x, near_iso_y, s=6, c="red", alpha=0.7, label=f"DoG peaks (n={len(near)})")
ax.set_title(f"DoG peaks at t={t_mid} z={z_mid}")
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
def process_sample(
    zarr_path: Path, offset: int = 0, topk_per_frame: int | None = None
) -> tuple[list[NodeRow], list[EdgeRow]]:
    """Run full DoG detection + linking pipeline on one zarr volume.

    Args:
        zarr_path: Path to the .zarr directory.
        offset: Node ID offset — add to all local node IDs for global uniqueness.
        topk_per_frame: If set, keep only top-k detections per frame by DoG score.

    Returns:
        Tuple of (nodes, edges) ready for build_submission.
    """
    dataset_id = zarr_path.stem
    dets = detect_cells(zarr_path, topk_per_frame=topk_per_frame)
    links = link_detections(dets)

    if DETECT_DIV:
        div_links = detect_divisions(dets, links)
        all_links = links + div_links
    else:
        all_links = links

    nodes = [
        NodeRow(dataset=dataset_id, node_id=offset + i + 1, t=d["t"], z=d["z"], y=d["y"], x=d["x"])
        for i, d in enumerate(dets)
    ]
    edges = [
        EdgeRow(dataset=dataset_id, source_id=offset + si + 1, target_id=offset + ti + 1)
        for si, ti in all_links
    ]

    if PRUNE_ISOLATED and edges:
        # Remove nodes unreferenced by any edge — almost always false positives
        used_ids = {e.source_id for e in edges} | {e.target_id for e in edges}
        nodes = [n for n in nodes if n.node_id in used_ids]

    print(f"  {dataset_id}: {len(dets)} dets → {len(nodes)} nodes (pruned), {len(edges)} edges")
    return nodes, edges

# %%
nodes, edges = process_sample(train_zarr[0])
sanity = build_submission(nodes, edges, OUTPUT_DIR / "sanity_submission.csv")
print(pd.read_csv(sanity)["row_type"].value_counts().to_string())

# %% [markdown]
# ### Detection sanity — detected vs estimated node count
# Quick per-sample check before the full run. The calibration section below learns the
# per-movie budget from all train samples; this is a spot-check on the first one.

# %%
_geff0 = train_geff[0]
_meta0_path = _geff0 / "zarr.json"
if _meta0_path.exists():
    with _meta0_path.open() as f:
        _meta0 = json.load(f)
    n_est = _meta0.get("estimated_number_of_nodes", None)
    n_detected = len(nodes)  # after pruning
    if n_est:
        ratio = n_detected / n_est
        flag = "  ⚠ over" if ratio > 1.5 else ("  ⚠ under" if ratio < 0.7 else "  ✓ in range")
        print(f"Estimated nodes : {n_est}")
        print(f"Detected nodes  : {n_detected}  (after isolated-node pruning)")
        print(f"Ratio           : {ratio:.2f}{flag}")
        print(f"Detected/frame  : {n_detected / get_volume_shape(train_zarr[0])[0]:.1f}")
        print(f"Estimated/frame : {n_est / get_volume_shape(train_zarr[0])[0]:.1f}")

# %% [markdown]
# ## Local Proxy Validation
# Port of the competitor's offline scoring harness. Runs the full pipeline on
# embryo-diverse train samples and scores against ground-truth GEFF graphs.
# **Why**: every parameter change otherwise costs a Kaggle submission.
# Metric: node F1 (7 µm matching gate) × 0.5 + edge Jaccard × 0.4 + 0.1 constant.
# Use this to rank parameter settings before submitting.

# %%
PROXY_GATE_UM = 7.0    # metric matching window (µm) — mirrors competition scoring
PROXY_VAL_SAMPLES = 2  # max embryo-diverse train movies to score


def _read_geff_gt(geff_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load GT nodes and edges from a .geff zarr store as DataFrames."""
    try:
        import zarr as _zarr
        g = _zarr.open(str(geff_path), mode="r")
        ids = np.asarray(g["nodes/ids"])
        t_vals = np.asarray(g["nodes/props/t/values"])
        z_vals = np.asarray(g["nodes/props/z/values"])
        y_vals = np.asarray(g["nodes/props/y/values"])
        x_vals = np.asarray(g["nodes/props/x/values"])
        node_df = pd.DataFrame({"node_id": ids, "t": t_vals, "z": z_vals, "y": y_vals, "x": x_vals})
        edge_ids = np.asarray(g["edges/ids"])
        if edge_ids.ndim == 2 and len(edge_ids):
            edge_df = pd.DataFrame({"source_id": edge_ids[:, 0], "target_id": edge_ids[:, 1]})
        else:
            edge_df = pd.DataFrame({"source_id": pd.Series(dtype=int), "target_id": pd.Series(dtype=int)})
        return node_df, edge_df
    except Exception as exc:
        print(f"  geff read failed: {exc}")
        return None, None


def _match_nodes_proxy(
    pred_nodes: pd.DataFrame, gt_nodes: pd.DataFrame, gate_um: float = PROXY_GATE_UM
) -> dict[int, int]:
    """Per-frame Hungarian node matching in µm space.

    Args:
        pred_nodes: DataFrame with columns node_id, t, z, y, x.
        gt_nodes: DataFrame with columns node_id, t, z, y, x.
        gate_um: Maximum matching distance in µm.

    Returns:
        Dict mapping pred node_id -> gt node_id for matched pairs.
    """
    p2g: dict[int, int] = {}
    for t in sorted(set(pred_nodes["t"]) & set(gt_nodes["t"])):
        p = pred_nodes[pred_nodes["t"] == t].reset_index(drop=True)
        g = gt_nodes[gt_nodes["t"] == t].reset_index(drop=True)
        if len(p) == 0 or len(g) == 0:
            continue
        p_um = p[["z", "y", "x"]].values * SCALE_ZYX[None, :]
        g_um = g[["z", "y", "x"]].values * SCALE_ZYX[None, :]
        D = np.sqrt(((p_um[:, None] - g_um[None]) ** 2).sum(2))
        cost = np.where(D <= gate_um, D, 1e6)
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 1e6:
                p2g[int(p.loc[a, "node_id"])] = int(g.loc[b, "node_id"])
    return p2g


def proxy_score_local(
    nodes: list[NodeRow], edges: list[EdgeRow], geff_path: Path
) -> tuple[float | None, dict]:
    """Score predictions against GT GEFF using the competition proxy metric.

    Args:
        nodes: Predicted NodeRow list from process_sample.
        edges: Predicted EdgeRow list from process_sample.
        geff_path: Path to the .geff directory for this sample.

    Returns:
        Tuple of (proxy_score, breakdown_dict). Score is None on GT read failure.
        proxy_score = 0.5 * node_f1 + 0.4 * edge_jaccard + 0.1
    """
    gt_nodes, gt_edges = _read_geff_gt(geff_path)
    if gt_nodes is None:
        return None, {}

    pred_nodes = pd.DataFrame(
        [{"node_id": n.node_id, "t": n.t, "z": float(n.z), "y": float(n.y), "x": float(n.x)}
         for n in nodes]
    )
    pred_edges_df = pd.DataFrame(
        [{"source_id": e.source_id, "target_id": e.target_id} for e in edges]
    ) if edges else pd.DataFrame({"source_id": pd.Series(dtype=int), "target_id": pd.Series(dtype=int)})

    # Clip GT to the volume's timepoint range
    if len(pred_nodes):
        gt_nodes = gt_nodes[gt_nodes["t"] <= int(pred_nodes["t"].max())]
    valid_gt = set(gt_nodes["node_id"])
    gt_edges = gt_edges[gt_edges["source_id"].isin(valid_gt) & gt_edges["target_id"].isin(valid_gt)]

    p2g = _match_nodes_proxy(pred_nodes, gt_nodes)
    tp = len(p2g)
    node_prec = tp / max(tp + len(pred_nodes) - tp, 1)
    node_rec = tp / max(tp + len(gt_nodes) - tp, 1)
    node_f1 = 2 * node_prec * node_rec / max(node_prec + node_rec, 1e-9)

    gt_eset = set(zip(gt_edges["source_id"].astype(int), gt_edges["target_id"].astype(int)))
    pred_mapped = {
        (p2g[s], p2g[t])
        for s, t in zip(pred_edges_df["source_id"].astype(int), pred_edges_df["target_id"].astype(int))
        if s in p2g and t in p2g
    }
    etp = len(pred_mapped & gt_eset)
    edge_prec = etp / max(len(pred_mapped), 1)
    edge_rec = etp / max(len(gt_eset), 1)
    edge_f1 = 2 * edge_prec * edge_rec / max(edge_prec + edge_rec, 1e-9)

    score = round(0.5 * node_f1 + 0.4 * edge_f1 + 0.1, 4)
    breakdown = dict(
        node_f1=round(node_f1, 3), node_recall=round(node_rec, 3), node_prec=round(node_prec, 3),
        edge_f1=round(edge_f1, 3), edge_recall=round(edge_rec, 3), edge_prec=round(edge_prec, 3),
        pred_nodes=len(pred_nodes), gt_nodes=len(gt_nodes),
    )
    return score, breakdown


# %%
# Pick embryo-diverse train movies (one per embryo prefix, max PROXY_VAL_SAMPLES)
_val_pick: list[Path] = []
_val_embryos: set[str] = set()
for _zp in train_zarr:
    _emb = _zp.stem.split("_")[0]
    if _emb in _val_embryos:
        continue
    _val_embryos.add(_emb)
    _val_pick.append(_zp)
    if len(_val_pick) >= PROXY_VAL_SAMPLES:
        break

_val_rows = []
for _zp in _val_pick:
    _geff = TRAIN_DIR / (_zp.stem + ".geff")
    if not _geff.exists():
        print(f"  {_zp.stem}: no geff, skipped")
        continue
    _vn, _ve = process_sample(_zp)
    _sc, _br = proxy_score_local(_vn, _ve, _geff)
    if _sc is None:
        continue
    _val_rows.append({"dataset": _zp.stem[:30], "proxy": _sc, **_br})
    print(
        f"  {_zp.stem[:28]:28s}  proxy={_sc:.4f}"
        f"  node_rec={_br['node_recall']:.3f}  edge_rec={_br['edge_recall']:.3f}"
        f"  pred={_br['pred_nodes']}  gt={_br['gt_nodes']}"
    )

if _val_rows:
    display(pd.DataFrame(_val_rows))
else:
    print("Proxy validation skipped — no train GEFF found.")

# %% [markdown]
# ## Count Calibration
# The competition metric penalises over-prediction: surplus nodes lower the Edge Jaccard
# denominator without adding true positives. The calibration cells below learn the
# correct per-movie topk budget from training data so each test movie stays near the
# estimated true cell count.
#
# **Algorithm:**
# 1. Run *generous* detection (DOG_THR_PCT=55) on each train movie → `D_gen` cells/frame.
# 2. Read `estimated_number_of_nodes` from the `.geff` zarr.json → `D_est` cells/frame.
# 3. `CALIB_FACTOR = median(D_est / D_gen)` — how much to scale down generous detections.
# 4. For each test movie: run generous detection → `topk = BUDGET_SAFETY × CALIB_FACTOR × D_gen`.

# %%
from joblib import Parallel, delayed
from tqdm.auto import tqdm

CALIB_FACTOR = 1.0
_test_topk: dict[str, int] = {}  # zarr name -> topk per frame

if USE_COUNT_CALIBRATION and train_zarr:
    print("=== Calibrating on train movies (joblib threads) ===")

    def _calib_one(zarr_path: Path):
        geff_path = TRAIN_DIR / (zarr_path.stem + ".geff")
        meta_path = geff_path / "zarr.json"
        if not meta_path.exists():
            return None
        with meta_path.open() as f:
            n_est = json.load(f).get("estimated_number_of_nodes")
        if n_est is None:
            return None
        T = get_volume_shape(zarr_path)[0]
        dets_gen = detect_cells(zarr_path, dog_thr_pct=GENEROUS_DOG_PCT)
        D_gen = len(dets_gen) / max(T, 1)
        if D_gen <= 0:
            return None
        return zarr_path.stem, n_est, T, D_gen

    calib_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_calib_one)(zp) for zp in tqdm(train_zarr, desc="Train calibration")
    )

    ratios: list[float] = []
    for r in calib_results:
        if r is None:
            continue
        stem, n_est, T, D_gen = r
        ratio = (n_est / T) / D_gen
        ratios.append(ratio)
        print(f"  {stem[:30]}: est={n_est/T:.0f}/frame  gen={D_gen:.0f}/frame  ratio={ratio:.3f}")

    if ratios:
        CALIB_FACTOR = float(np.median(ratios))
        print(f"\nCALIB_FACTOR = {CALIB_FACTOR:.3f}  (from {len(ratios)} train movies)")
else:
    print("Count calibration skipped (USE_COUNT_CALIBRATION=False or no train data).")

# %%
test_zarrs_sorted = sorted(TEST_DIR.glob("*.zarr"))
print("=== Estimating test budgets (joblib threads) ===")

def _budget_one(zarr_path: Path):
    T = get_volume_shape(zarr_path)[0]
    dets_gen = detect_cells(zarr_path, dog_thr_pct=GENEROUS_DOG_PCT)
    D_gen = len(dets_gen) / max(T, 1)
    topk = max(1, int(np.ceil(BUDGET_SAFETY * CALIB_FACTOR * D_gen)))
    return zarr_path.name, topk, D_gen

budget_results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(_budget_one)(zp) for zp in tqdm(test_zarrs_sorted, desc="Test budgets")
)

for name, topk, D_gen in budget_results:
    _test_topk[name] = topk
    print(f"  {name[:40]}: gen={D_gen:.0f}/frame → topk={topk}/frame")

# %% [markdown]
# ## Full Run: All Test Samples
# Each test movie runs independently via joblib threads — DoG/scipy release the GIL so
# all CPUs are utilised. Results collected in input order then merged with correct node ID
# offsets.

# %%
def _run_test_movie(zarr_path: Path) -> tuple[list[NodeRow], list[EdgeRow]]:
    """Process one test movie with offset=0; caller applies global offset during merge."""
    topk = _test_topk.get(zarr_path.name)
    return process_sample(zarr_path, offset=0, topk_per_frame=topk)

per_movie = Parallel(n_jobs=-1, prefer="threads")(
    delayed(_run_test_movie)(zp) for zp in tqdm(test_zarrs_sorted, desc="Test samples")
)

# Merge: apply global offset so node IDs are globally unique across all datasets
all_nodes: list[NodeRow] = []
all_edges: list[EdgeRow] = []
for nodes, edges in per_movie:
    offset = len(all_nodes)
    for n in nodes:
        n.node_id += offset
    for e in edges:
        e.source_id += offset
        e.target_id += offset
    all_nodes += nodes
    all_edges += edges

print(f"Total nodes: {len(all_nodes)}, edges: {len(all_edges)}  (topk budgets: {_test_topk})")

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
