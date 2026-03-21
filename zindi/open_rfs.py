"""Open a zarr v3 reference file system as a zarr Group.

Given an RFS dict (or path to a JSON file or parquet directory), this
creates an RfsStore and opens it as a zarr v3 group hierarchy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import zarr

from .rfs_store import RfsStore


def open_rfs(rfs: dict | str, *, local_cache: Any = None) -> zarr.Group:
    """Open a reference file system as a zarr v3 Group.

    Parameters
    ----------
    rfs : dict or str
        Either an RFS dict (with "refs" and "version" keys), a path
        to a JSON file, or a path to a parquet directory containing
        ``metadata.json`` and ``chunk_refs.parquet``.
    local_cache : LocalCache or None
        Optional local cache for persisting remote chunk data on disk.

    Returns
    -------
    zarr.Group
        A read-only zarr v3 Group backed by the reference file system.
    """
    if isinstance(rfs, str):
        if os.path.isdir(rfs):
            rfs = _load_rfs_parquet(rfs)
        else:
            with open(rfs) as f:
                rfs = json.load(f)

    assert isinstance(rfs, dict)

    store = RfsStore(rfs, local_cache=local_cache)
    return zarr.open_group(store, mode="r", zarr_format=3)


def _load_rfs_parquet(dir_path: str) -> dict:
    """Load an RFS from a parquet directory."""
    import pandas as pd

    with open(os.path.join(dir_path, "metadata.json")) as f:
        rfs = json.load(f)

    parquet_path = os.path.join(dir_path, "chunk_refs.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            rfs["refs"][row["key"]] = [row["path"], int(row["offset"]), int(row["size"])]

    return rfs
