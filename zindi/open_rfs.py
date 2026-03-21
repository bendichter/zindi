"""Open a zarr v3 reference file system as a zarr Group.

Given an RFS dict (or path to a JSON file), this creates an RfsStore
and opens it as a zarr v3 group hierarchy.
"""

from __future__ import annotations

import json
from typing import Any

import zarr

from .rfs_store import RfsStore


def open_rfs(
    rfs: dict | str,
    *,
    local_cache: Any = None,
    merge_gap: int = 256 * 1024,
    max_merge_size: int = 50 * 1024 * 1024,
) -> zarr.Group:
    """Open a reference file system as a zarr v3 Group.

    Parameters
    ----------
    rfs : dict or str
        Either an RFS dict (with "refs" and "version" keys), or a path
        to a JSON file containing one.
    local_cache : LocalCache or None
        Optional local cache for persisting remote chunk data on disk.
    merge_gap : int
        Maximum gap in bytes between two HTTP range requests before they
        are fetched separately. Nearby ranges are merged into a single
        request. Default 256 KB.
    max_merge_size : int
        Maximum size in bytes for a single merged HTTP request. Default 50 MB.

    Returns
    -------
    zarr.Group
        A read-only zarr v3 Group backed by the reference file system.
    """
    if isinstance(rfs, str):
        with open(rfs) as f:
            rfs = json.load(f)

    assert isinstance(rfs, dict)

    store = RfsStore(
        rfs, local_cache=local_cache, merge_gap=merge_gap, max_merge_size=max_merge_size
    )
    return zarr.open_group(store, mode="r", zarr_format=3)
