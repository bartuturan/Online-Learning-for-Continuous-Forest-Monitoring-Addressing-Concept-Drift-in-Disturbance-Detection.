"""Cross-process, disk-backed cache for prepare_raw_features_for_year outputs.

Every notebook process that needs a given (dataset, year, feature flags, pixel
subset) currently recomputes it from the zarr store, which is decompression-bound
at ~25-35s/year regardless of how many pixels are requested. This module makes
that computation reusable across notebook/kernel processes and across reruns.

The cache key is content-fingerprinted, never label-trusted: dataset identity
comes from hashing the zarr store's root `zarr.json` (schema, not values -- see
CACHE_SCHEMA_VERSION below), pixel-set identity comes from hashing the actual
pixel_indices array bytes (not a caller-supplied split name, since e.g. a sanity
check that slices `train_pixel_indices[:10000]` must never collide with the full
train split), and every feature flag -- including impute_window, whose two values
('expanding' for training, 'all_years' legacy-leaky for evaluation/drift -- see
prepare_raw_features_for_year's docstring) must never be conflated -- is part of
the key. A wrong or missing label only costs a cache miss, never a wrong hit.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CACHE_SCHEMA_VERSION = "v1"
CACHE_ROOT_ENV_VAR = "MLP_FEATURE_CACHE_DIR"


def resolve_cache_root(ds_path, cache_root=None):
    """cache_root override > MLP_FEATURE_CACHE_DIR env var > <ds_path's dir>/feature_cache_disk.

    ds_path's own directory is used as the default base (rather than re-deriving a
    project root) because every current caller already opens datasets as
    `PROJECT_ROOT / '<name>.zarr'`, so ds_path.parent already *is* the project root.
    """
    if cache_root is not None:
        return Path(cache_root)
    env_override = os.environ.get(CACHE_ROOT_ENV_VAR)
    if env_override:
        return Path(env_override)
    return Path(ds_path).parent / "feature_cache_disk"


def _dataset_fingerprint(ds_path):
    """sha256 of the zarr store's root zarr.json (zarr v3 inline consolidated_metadata:
    one file covering every variable's shape/dtype/chunking/codec). Any schema-changing
    regeneration (added/removed variable, shape or chunking change) changes this hash.

    Known limitation: an in-place regeneration that keeps the exact schema but writes
    different values into existing chunks is NOT detected here -- checksumming chunk
    data would cost as much as the decompression this cache exists to avoid. Bump
    CACHE_SCHEMA_VERSION by hand if that ever happens, or delete the cache directory.
    """
    zarr_json_path = Path(ds_path) / "zarr.json"
    if not zarr_json_path.exists():
        raise FileNotFoundError(
            f"Expected zarr v3 metadata at {zarr_json_path} -- is {ds_path} a zarr v3 store?"
        )
    full = hashlib.sha256(zarr_json_path.read_bytes()).hexdigest()
    return full, full[:16]


def _pixel_fingerprint(pixel_indices):
    """sha256 of the pixel_indices array's bytes, canonicalized to int64 first so an
    int32 vs int64 caller for the otherwise-identical index set can't diverge."""
    canonical = np.ascontiguousarray(pixel_indices, dtype=np.int64)
    full = hashlib.sha256(canonical.tobytes()).hexdigest()
    return full, full[:16]


def _flags_payload(include_last_year, include_monthly, include_neighbourhood, impute_window, dtype):
    return {
        "include_last_year": bool(include_last_year),
        # Tri-state: None ("use dataset defaults") must never collide with explicit True/False.
        "include_monthly": "None" if include_monthly is None else bool(include_monthly),
        "include_neighbourhood": bool(include_neighbourhood),
        "impute_window": str(impute_window),
        "dtype": "None" if dtype is None else np.dtype(dtype).name,
    }


def _flags_fingerprint(flags_payload):
    encoded = json.dumps(flags_payload, sort_keys=True).encode("utf-8")
    full = hashlib.sha256(encoded).hexdigest()
    return full, full[:8]


def _sanitize_label(label):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(label))


def _leaf_dir(cache_root, dataset_tag, dataset_fp_short, split_label, pixel_fp_short, year_idx, flags_fp_short):
    return (
        Path(cache_root)
        / f"schema_{CACHE_SCHEMA_VERSION}"
        / f"{dataset_tag}__dsfp_{dataset_fp_short}"
        / f"{split_label}__pxfp_{pixel_fp_short}"
        / f"year_{int(year_idx):02d}__flags_{flags_fp_short}"
    )


def _read_and_validate(leaf_dir, expected_dataset_fp, expected_pixel_fp, expected_flags):
    manifest_path = leaf_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if manifest.get("dataset_fingerprint") != expected_dataset_fp:
        return None
    if manifest.get("pixel_fingerprint") != expected_pixel_fp:
        return None
    if manifest.get("flags") != expected_flags:
        return None

    x_path = leaf_dir / "X.npy"
    y_path = leaf_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        return None

    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except (OSError, ValueError):
        return None

    if list(X.shape) != manifest.get("X_shape") or list(y.shape) != manifest.get("y_shape"):
        return None
    if str(X.dtype) != manifest.get("X_dtype") or str(y.dtype) != manifest.get("y_dtype"):
        return None

    return X, y


def _write_atomic(leaf_dir, X, y, manifest):
    """Write X.npy, y.npy, manifest.json via temp-file-then-os.replace, manifest last.

    manifest.json's presence is the "this entry is complete and trustworthy" marker --
    a reader that sees X.npy/y.npy but no manifest.json (a crash mid-write, or a
    concurrent writer that hasn't finished) always treats the entry as a miss and
    safely recomputes/overwrites. No locking: if two processes race on the same key,
    both may recompute and both publish -- harmless, since the content is identical
    and os.replace is atomic on both POSIX and Windows for same-volume renames.
    """
    leaf_dir.mkdir(parents=True, exist_ok=True)
    token = os.urandom(8).hex()

    x_tmp = leaf_dir / f"X.npy.tmp{token}"
    y_tmp = leaf_dir / f"y.npy.tmp{token}"
    manifest_tmp = leaf_dir / f"manifest.json.tmp{token}"

    with open(x_tmp, "wb") as f:
        np.save(f, X)
    with open(y_tmp, "wb") as f:
        np.save(f, y)
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    os.replace(x_tmp, leaf_dir / "X.npy")
    os.replace(y_tmp, leaf_dir / "y.npy")
    os.replace(manifest_tmp, leaf_dir / "manifest.json")


def get_or_compute_year_features(
    ds,
    ds_path,
    pixel_indices,
    year_idx,
    *,
    split_name=None,
    dtype=np.float32,
    include_last_year=True,
    include_monthly=None,
    include_neighbourhood=False,
    impute_window="expanding",
    cache_root=None,
    force_recompute=False,
):
    """Disk-backed, cross-process wrapper around prepare_raw_features_for_year.

    Returns (X, y) identical to what
    prepare_raw_features_for_year(..., return_feature_names=False) would return.
    `split_name` is cosmetic only (used for a human-browsable directory name) --
    correctness never depends on it, only on the hash of pixel_indices itself.
    """
    # Deferred import: src/mlp_replay/data.py imports get_or_compute_year_features
    # (lazily, inside its own functions) so this module can import from data.py at
    # module level without creating an import cycle.
    from src.mlp_replay.data import prepare_raw_features_for_year

    ds_path = Path(ds_path)
    root = resolve_cache_root(ds_path, cache_root)

    dataset_fp_full, dataset_fp_short = _dataset_fingerprint(ds_path)
    pixel_fp_full, pixel_fp_short = _pixel_fingerprint(pixel_indices)
    flags_payload = _flags_payload(include_last_year, include_monthly, include_neighbourhood, impute_window, dtype)
    _, flags_fp_short = _flags_fingerprint(flags_payload)
    split_label = _sanitize_label(split_name) if split_name else "pixels"

    leaf_dir = _leaf_dir(root, ds_path.stem, dataset_fp_short, split_label, pixel_fp_short, year_idx, flags_fp_short)

    if not force_recompute:
        hit = _read_and_validate(leaf_dir, dataset_fp_full, pixel_fp_full, flags_payload)
        if hit is not None:
            return hit

    X, y = prepare_raw_features_for_year(
        ds,
        pixel_indices,
        year_idx,
        dtype=dtype,
        include_last_year=include_last_year,
        include_monthly=include_monthly,
        include_neighbourhood=include_neighbourhood,
        impute_window=impute_window,
    )

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset_path": str(ds_path),
        "dataset_fingerprint": dataset_fp_full,
        "pixel_fingerprint": pixel_fp_full,
        "pixel_count": int(len(pixel_indices)),
        "split_name": split_name,
        "year_idx": int(year_idx),
        "flags": flags_payload,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_atomic(leaf_dir, X, y, manifest)
    return X, y
