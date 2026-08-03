"""Provenance manifest for the pixel sample produced by notebooks/data_prep/Data-prep.ipynb.

Everything downstream of Data-prep addresses pixels *positionally*: data_split.npz stores
integer indices into the `pixel` dimension, the disk feature cache keys on those indices,
and every result under experiments/ was scored against them. That contract breaks silently
if the sampler is re-run and produces a different pixel set or a different order -- nothing
raises, the numbers just quietly start referring to different ground.

This module makes the pixel population identifiable:

- `compute_pixel_identity_hash` reduces (cube name, cube index, y, x) to one digest. Those
  four values are exactly what a positional index resolves to, so two datasets share a
  digest if and only if index `i` means the same pixel in both.
- `write_manifest` records that digest next to the store, together with the sampler
  configuration that produced it.
- `verify_dataset_matches_manifest` and `verify_split_matches_dataset` turn a mismatch into
  an exception instead of a wrong result.

The digest normalises integer width before hashing, so an array built as int64 in memory and
an array read back as int32 from zarr agree. It does not depend on feature values -- adding
a column to the dataset leaves it unchanged, which is the intent: features can be recomputed,
pixel positions cannot be, without invalidating every artifact that references them.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Bumping this invalidates every previously written digest, so change it only when the
#: bytes fed to the hash change meaning.
PIXEL_IDENTITY_HASH_VERSION = "fonda-pixel-identity-v1"

#: Key under which the digest is stored inside data_split.npz.
SPLIT_HASH_KEY = "pixel_identity_sha256"

MANIFEST_VERSION = 1


def manifest_path_for(dataset_path):
    """Where the manifest for `dataset_path` lives (training_data.zarr -> training_data.manifest.json)."""
    return Path(dataset_path).with_suffix(".manifest.json")


def _digest(n_pixels, cube_codes, cube_names_by_code, cube_idx, y, x):
    """Hash a pixel population from its already-reduced parts.

    cube_codes/cube_names_by_code are the sorted distinct cube indices and their names, which
    is what pins `cube_idx` to actual cubes; hashing that table rather than the full per-pixel
    name array keeps the digest cheap to compute from a zarr store.
    """
    h = hashlib.sha256()
    h.update(PIXEL_IDENTITY_HASH_VERSION.encode("utf-8"))
    h.update(b"\x00n_pixels=")
    h.update(str(int(n_pixels)).encode("utf-8"))
    h.update(b"\x00n_cubes=")
    h.update(str(len(cube_codes)).encode("utf-8"))
    h.update(b"\x00")
    h.update("\n".join(str(name) for name in cube_names_by_code).encode("utf-8"))
    h.update(b"\x00")
    for array in (cube_codes, cube_idx, y, x):
        h.update(np.ascontiguousarray(array, dtype="<i8").tobytes())
    return h.hexdigest()


def _reduce_cube_names(cube_idx, cube_names):
    """Sorted distinct cube indices plus one name each, taken from the first pixel of each cube."""
    cube_codes, first_occurrence = np.unique(np.asarray(cube_idx), return_index=True)
    names = np.asarray(cube_names)[first_occurrence]
    return cube_codes, names


def compute_pixel_identity_hash(cube_idx, y, x, cube_names):
    """Digest identifying a pixel population and its order.

    All four arrays must be aligned along `pixel`. `cube_names` is reduced to one name per
    distinct `cube_idx`, so it may be the full per-pixel array or any array that agrees with
    it at each cube's first pixel.
    """
    cube_idx = np.asarray(cube_idx)
    y = np.asarray(y)
    x = np.asarray(x)

    lengths = {len(cube_idx), len(y), len(x), len(np.asarray(cube_names))}
    if len(lengths) != 1:
        raise ValueError(
            f"cube_idx/y/x/cube_names must be aligned along pixel, got lengths {sorted(lengths)}"
        )

    cube_codes, names_by_code = _reduce_cube_names(cube_idx, cube_names)
    return _digest(len(cube_idx), cube_codes, names_by_code, cube_idx, y, x)


def pixel_identity_hash_from_dataset(ds):
    """Same digest, computed from an open dataset without reading the full cube_name array.

    `cube_name` is one fixed-width string per pixel -- ~1 GB on the real store -- but only one
    name per cube is needed, so it is read at the first pixel of each cube instead of whole.
    """
    for var_name in ("cube_idx", "cube_name", "x", "y"):
        if var_name not in ds.variables:
            raise KeyError(f"Dataset is missing '{var_name}', required to identify its pixels")

    cube_idx = ds["cube_idx"].values
    cube_codes, first_occurrence = np.unique(cube_idx, return_index=True)
    names_by_code = ds["cube_name"].isel(pixel=first_occurrence).values

    return _digest(
        len(cube_idx),
        cube_codes,
        names_by_code,
        cube_idx,
        ds["y"].values,
        ds["x"].values,
    )


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def write_manifest(dataset_path, cube_idx, y, x, cube_names, config, source_dataset=None, extra=None):
    """Write the manifest for `dataset_path` and return it.

    `config` should carry every setting that changes which pixels are sampled or in what
    order -- a manifest that omits one records a run nobody can reproduce.
    """
    dataset_path = Path(dataset_path)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": dataset_path.name,
        "source_dataset": str(source_dataset) if source_dataset is not None else None,
        "pixel_identity_hash_version": PIXEL_IDENTITY_HASH_VERSION,
        "pixel_identity_sha256": compute_pixel_identity_hash(cube_idx, y, x, cube_names),
        "config": _to_jsonable(config),
        "stats": _to_jsonable(extra or {}),
    }

    path = manifest_path_for(dataset_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def read_manifest(dataset_path):
    """Load the manifest for `dataset_path`. Raises FileNotFoundError if it has none."""
    path = manifest_path_for(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. The dataset either predates manifest support or was "
            f"written by something other than Data-prep.ipynb."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_dataset_matches_manifest(ds, dataset_path):
    """Raise if `ds` holds a different pixel population than `dataset_path`'s manifest records.

    Used to check that a derived store (training_data_with_features.zarr) still carries the
    pixels the sampler produced, before any positional index is built against it.
    """
    manifest = read_manifest(dataset_path)
    expected = manifest["pixel_identity_sha256"]
    actual = pixel_identity_hash_from_dataset(ds)
    if actual != expected:
        raise ValueError(
            f"Pixel identity mismatch against {manifest_path_for(dataset_path).name}:\n"
            f"  manifest: {expected}\n"
            f"  dataset : {actual}\n"
            f"The dataset was re-sampled or reordered since that manifest was written, so any "
            f"positional pixel index built against it (data_split.npz, the disk feature cache) "
            f"now points at different pixels."
        )
    return actual


def read_split_hash(split_path):
    """The pixel identity digest a split was built against, or None for a pre-manifest split."""
    with np.load(split_path) as split_data:
        if SPLIT_HASH_KEY not in split_data:
            return None
        return str(split_data[SPLIT_HASH_KEY].item())


def verify_split_matches_dataset(split_path, ds):
    """Raise if `split_path`'s indices were built against a different pixel population than `ds`.

    Returns the digest on success, or None when the split predates manifest support -- an
    unstamped split cannot be checked, only regenerated.
    """
    recorded = read_split_hash(split_path)
    if recorded is None:
        return None

    actual = pixel_identity_hash_from_dataset(ds)
    if recorded != actual:
        raise ValueError(
            f"{Path(split_path).name} was built against a different pixel population:\n"
            f"  split   : {recorded}\n"
            f"  dataset : {actual}\n"
            f"Its indices address positions that now hold different pixels. Regenerate the "
            f"split from this dataset before training against it."
        )
    return actual
