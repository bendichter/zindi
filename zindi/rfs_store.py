"""Zarr v3 Store backed by a reference file system (RFS).

This is a zarr v3 Store that reads data from a reference file system dict.
It handles:
- Inline data (strings, base64-encoded bytes, JSON dicts)
- Remote chunk references [url, offset, size]
- URL template expansion ({{u1}} etc.)
- DANDI URL resolution (redirects + auth)
- Retry with exponential backoff
- Chunk padding for contiguous HDF5 datasets

Ported from lindi's LindiReferenceFileSystemStore, adapted for zarr v3.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import requests
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

from .url_resolver import resolve_url


class RfsStore(Store):
    """A read-only zarr v3 Store backed by a reference file system dict.

    Parameters
    ----------
    rfs : dict
        Reference file system dict with "refs" key, and optional "templates".
    local_cache : LocalCache or None
        Optional local cache for persisting remote chunk data on disk.
    """

    def __init__(self, rfs: dict, *, local_cache: Any = None) -> None:
        super().__init__(read_only=True)
        if "refs" not in rfs:
            raise ValueError("rfs must contain a 'refs' key")
        self.rfs = rfs
        self._local_cache = local_cache
        self._is_open = True

    # -- Abstract method implementations --

    def __eq__(self, value: object) -> bool:
        return isinstance(value, RfsStore) and value.rfs is self.rfs

    @property
    def supports_writes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_deletes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_listing(self) -> bool:  # type: ignore[override]
        return True

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()
        data = await asyncio.to_thread(self._get_bytes, key)
        if data is None:
            return None
        if byte_range is not None:
            data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Any,
    ) -> list[Buffer | None]:
        results = []
        for key, byte_range in key_ranges:
            results.append(await self.get(key, prototype, byte_range))
        return results

    async def exists(self, key: str) -> bool:
        return key in self.rfs["refs"]

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("RfsStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("RfsStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        for key in self.rfs["refs"]:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self.rfs["refs"]:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        prefix_len = len(prefix)
        seen: set[str] = set()
        for key in self.rfs["refs"]:
            if not key.startswith(prefix):
                continue
            remainder = key[prefix_len:]
            if "/" in remainder:
                # It's inside a subdirectory; yield the directory name
                subdir = remainder.split("/")[0]
                if subdir not in seen:
                    seen.add(subdir)
                    yield subdir
            else:
                yield remainder

    # -- Core data resolution --

    def _get_bytes(self, key: str) -> bytes | None:
        """Resolve a key to bytes, handling all reference types."""
        if key not in self.rfs["refs"]:
            return None

        x = self.rfs["refs"][key]

        if isinstance(x, str):
            if x.startswith("base64:"):
                return base64.b64decode(x[len("base64:"):])
            else:
                return x.encode("utf-8")
        elif isinstance(x, dict):
            return json.dumps(x).encode("utf-8")
        elif isinstance(x, list):
            if len(x) != 3:
                raise ValueError(f"Reference list for {key} must have 3 elements")
            url_or_path, offset, length = x[0], x[1], x[2]

            # Expand templates
            if "{{" in url_or_path and "}}" in url_or_path and "templates" in self.rfs:
                for tkey, tval in self.rfs["templates"].items():
                    url_or_path = url_or_path.replace("{{" + tkey + "}}", tval)

            is_url = url_or_path.startswith("http://") or url_or_path.startswith("https://")

            # Check local cache for remote chunks
            if self._local_cache is not None and is_url:
                cached = self._local_cache.get_remote_chunk(
                    url=url_or_path, offset=offset, size=length
                )
                if cached is not None:
                    padded_size = self._get_padded_size(key, cached)
                    if padded_size is not None:
                        cached = cached + b"\0" * (padded_size - len(cached))
                    return cached

            data = _read_bytes_from_url_or_path(url_or_path, offset, length)

            # Store in local cache
            if self._local_cache is not None and is_url:
                from .local_cache import ChunkTooLargeError

                try:
                    self._local_cache.put_remote_chunk(
                        url=url_or_path, offset=offset, size=length, data=data
                    )
                except ChunkTooLargeError:
                    pass  # chunk exceeds SQLite blob limit, skip caching

            # Pad if this is a final chunk in a contiguous dataset
            padded_size = self._get_padded_size(key, data)
            if padded_size is not None:
                data = data + b"\0" * (padded_size - len(data))

            return data
        else:
            raise ValueError(f"Unexpected reference type for {key}: {type(x)}")

    def _get_padded_size(self, key: str, data: bytes) -> int | None:
        """Check if a chunk needs padding (final chunk in contiguous dataset).

        In zarr v3, chunk keys look like: path/c/0/1/2
        """
        parts = key.split("/")
        # Find the 'c' separator - everything after it is chunk indices
        try:
            c_idx = parts.index("c")
        except ValueError:
            return None

        if c_idx >= len(parts) - 1:
            return None

        # Check that everything after 'c' is an integer (chunk index)
        for p in parts[c_idx + 1:]:
            try:
                int(p)
            except ValueError:
                return None

        # Get the zarr.json for this array
        array_path = "/".join(parts[:c_idx])
        meta_key = f"{array_path}/zarr.json" if array_path else "zarr.json"

        if meta_key not in self.rfs["refs"]:
            return None

        meta_bytes = self._get_bytes(meta_key)
        if meta_bytes is None:
            return None
        meta = json.loads(meta_bytes)

        if meta.get("node_type") != "array":
            return None

        chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
        data_type = meta.get("data_type")
        if chunk_shape is None or data_type is None:
            return None

        if isinstance(data_type, dict):
            if data_type.get("name") == "structured":
                fields = data_type["configuration"]["fields"]
                dtype = np.dtype([
                    (f[0], _zarr_field_type_to_numpy(f[1])) for f in fields
                ])
            else:
                return None
        elif isinstance(data_type, str):
            dtype = np.dtype(data_type)
        else:
            return None
        if dtype.kind not in ("i", "u", "f", "V"):
            return None

        expected_size = int(np.prod(chunk_shape)) * dtype.itemsize
        if len(data) < expected_size:
            return expected_size

        return None


def _zarr_field_type_to_numpy(field_type: str | dict) -> str:
    """Convert a zarr v3 field type to a numpy dtype string."""
    if isinstance(field_type, str):
        return field_type
    if isinstance(field_type, dict):
        name = field_type.get("name")
        if name == "null_terminated_bytes":
            length = field_type["configuration"]["length_bytes"]
            return f"S{length}"
        if name == "fixed_length_utf32":
            # length_bytes is total bytes; each UTF-32 char is 4 bytes
            length = field_type["configuration"]["length_bytes"] // 4
            return f"U{length}"
    raise ValueError(f"Unsupported zarr field type: {field_type}")


def _read_bytes_from_url_or_path(url_or_path: str, offset: int, length: int) -> bytes:
    """Read a byte range from a URL or local file path."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return _read_bytes_from_url(url_or_path, offset, length)
    else:
        with open(url_or_path, "rb") as f:
            f.seek(offset)
            return f.read(length)


def _read_bytes_from_url(url: str, offset: int, length: int) -> bytes:
    """Read a byte range from a URL with retry and DANDI resolution."""
    num_retries = 8
    for try_num in range(num_retries):
        try:
            resolved_url = resolve_url(url)
            range_header = f"bytes={offset}-{offset + length - 1}"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Range": range_header,
            }
            response = requests.get(resolved_url, headers=headers)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if try_num == num_retries - 1:
                raise
            delay = 0.1 * 2**try_num
            print(f"Retry {try_num + 1}/{num_retries} for {url} in {delay:.1f}s: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to read from {url}")


def _apply_byte_range(data: bytes, byte_range: ByteRequest) -> bytes:
    """Apply a ByteRequest to raw bytes."""
    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

    if isinstance(byte_range, RangeByteRequest):
        end = byte_range.end if byte_range.end is not None else len(data)
        return data[byte_range.start:end]
    elif isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset:]
    elif isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix:]
    else:
        return data
