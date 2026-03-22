"""Zarr v3 Store backed by a reference file system (RFS).

This is a zarr v3 Store that reads data from a reference file system dict.
It handles:
- Inline data (strings, base64-encoded bytes, JSON dicts)
- Remote chunk references [url, offset, size]
- URL template expansion ({{u1}} etc.)
- DANDI URL resolution (redirects + auth)
- Retry with exponential backoff
- Chunk padding for contiguous HDF5 datasets
- Automatic coalescing of concurrent HTTP range requests

Ported from lindi's LindiReferenceFileSystemStore, adapted for zarr v3.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
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
    merge_gap : int
        Maximum gap in bytes between two ranges before they are fetched
        separately. Ranges within this distance are merged into a single
        HTTP request. Default 256 KB.
    max_merge_size : int
        Maximum size in bytes for a single merged HTTP request. Merged
        ranges that would exceed this are split. Default 50 MB.
    """

    def __init__(
        self,
        rfs: dict,
        *,
        local_cache: Any = None,
        merge_gap: int = 256 * 1024,
        max_merge_size: int = 50 * 1024 * 1024,
    ) -> None:
        super().__init__(read_only=True)
        if "refs" not in rfs:
            raise ValueError("rfs must contain a 'refs' key")
        self.rfs = rfs
        self._local_cache = local_cache
        self._merge_gap = merge_gap
        self._max_merge_size = max_merge_size
        self._executor = ThreadPoolExecutor(max_workers=32)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0"
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

        # Resolve the ref to determine if this is a remote byte-range request
        ref = self.rfs["refs"].get(key)
        if ref is None:
            return None

        # Non-remote refs: handle inline
        if not (isinstance(ref, list) and len(ref) == 3):
            data = self._get_inline_bytes(ref)
            if byte_range is not None:
                data = _apply_byte_range(data, byte_range)
            return prototype.buffer.from_bytes(data)

        # Remote ref: resolve URL and check cache
        url, offset, length = self._resolve_ref(ref)
        is_url = url.startswith("http://") or url.startswith("https://")

        if not is_url:
            # Local file — read directly
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(
                self._executor, _read_local_file, url, offset, length
            )
            data = self._apply_padding(key, data)
            if byte_range is not None:
                data = _apply_byte_range(data, byte_range)
            return prototype.buffer.from_bytes(data)

        # Check local cache
        if self._local_cache is not None:
            cached = self._local_cache.get_remote_chunk(url=url, offset=offset, size=length)
            if cached is not None:
                cached = self._apply_padding(key, cached)
                if byte_range is not None:
                    cached = _apply_byte_range(cached, byte_range)
                return prototype.buffer.from_bytes(cached)

        # Remote URL: fetch via shared session in thread pool
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            self._executor, self._fetch_url, url, offset, length
        )
        data = self._apply_padding(key, data)

        # Store in local cache
        if self._local_cache is not None:
            from .local_cache import ChunkTooLargeError

            try:
                self._local_cache.put_remote_chunk(url=url, offset=offset, size=length, data=data)
            except ChunkTooLargeError:
                pass

        if byte_range is not None:
            data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Any,
    ) -> list[Buffer | None]:
        return list(await asyncio.gather(
            *(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)
        ))

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
                subdir = remainder.split("/")[0]
                if subdir not in seen:
                    seen.add(subdir)
                    yield subdir
            else:
                yield remainder

    # -- Helpers --

    def _fetch_url(self, url: str, offset: int, length: int) -> bytes:
        """Fetch a byte range using the shared HTTP session (connection reuse)."""
        num_retries = 8
        for try_num in range(num_retries):
            try:
                resolved_url = resolve_url(url)
                response = self._session.get(
                    resolved_url,
                    headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                )
                response.raise_for_status()
                return response.content
            except Exception as e:
                if try_num == num_retries - 1:
                    raise
                delay = 0.1 * 2**try_num
                time.sleep(delay)
        raise RuntimeError(f"Failed to read from {url}")

    def _resolve_ref(self, ref: list) -> tuple[str, int, int]:
        """Expand templates in a [url, offset, size] ref."""
        url_or_path = ref[0]
        if "{{" in url_or_path and "}}" in url_or_path and "templates" in self.rfs:
            for tkey, tval in self.rfs["templates"].items():
                url_or_path = url_or_path.replace("{{" + tkey + "}}", tval)
        return url_or_path, ref[1], ref[2]

    def _get_inline_bytes(self, ref: Any) -> bytes:
        """Resolve an inline (non-remote) ref to bytes."""
        if isinstance(ref, str):
            if ref.startswith("base64:"):
                return base64.b64decode(ref[len("base64:"):])
            else:
                return ref.encode("utf-8")
        elif isinstance(ref, dict):
            return json.dumps(ref).encode("utf-8")
        else:
            raise ValueError(f"Unexpected inline ref type: {type(ref)}")

    def _apply_padding(self, key: str, data: bytes) -> bytes:
        """Pad data if this is a final chunk in a contiguous dataset."""
        padded_size = self._get_padded_size(key, data)
        if padded_size is not None:
            return data + b"\0" * (padded_size - len(data))
        return data

    def _get_padded_size(self, key: str, data: bytes) -> int | None:
        """Check if a chunk needs padding (final chunk in contiguous dataset)."""
        parts = key.split("/")
        try:
            c_idx = parts.index("c")
        except ValueError:
            return None

        if c_idx >= len(parts) - 1:
            return None

        for p in parts[c_idx + 1:]:
            try:
                int(p)
            except ValueError:
                return None

        array_path = "/".join(parts[:c_idx])
        meta_key = f"{array_path}/zarr.json" if array_path else "zarr.json"

        if meta_key not in self.rfs["refs"]:
            return None

        meta_ref = self.rfs["refs"][meta_key]
        meta_bytes = self._get_inline_bytes(meta_ref) if not isinstance(meta_ref, list) else None
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


class _RequestBatcher:
    """Coalesces concurrent get() requests into merged HTTP fetches.

    When multiple get() calls arrive on the event loop simultaneously
    (as zarr does via concurrent_map), this batcher collects them during
    one event loop yield, then dispatches merged HTTP requests for nearby
    byte ranges on the same URL.
    """

    def __init__(self, store: RfsStore):
        self._store = store
        self._pending: dict[str, list[tuple[int, int, str, asyncio.Future]]] = {}
        self._dispatch_scheduled = False

    async def request(self, url: str, offset: int, size: int, key: str) -> bytes:
        """Register a request and wait for the merged result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        self._pending.setdefault(url, []).append((offset, size, key, future))

        if not self._dispatch_scheduled:
            self._dispatch_scheduled = True
            # Schedule dispatch after yielding — lets other concurrent
            # get() calls register before we fetch
            asyncio.ensure_future(self._dispatch())

        return await future

    async def _dispatch(self):
        """Wait briefly to collect concurrent requests, then fetch merged ranges."""
        # Small delay to let concurrent get() tasks register.
        # Zarr's concurrent_map dispatches in waves; 5ms captures most of a wave.
        await asyncio.sleep(0.005)

        # Grab all pending requests
        pending = self._pending
        self._pending = {}
        self._dispatch_scheduled = False

        # Dispatch each URL group in parallel
        tasks = []
        for url, reqs in pending.items():
            tasks.append(self._fetch_url_group(url, reqs))
        await asyncio.gather(*tasks)

    async def _fetch_url_group(
        self,
        url: str,
        reqs: list[tuple[int, int, str, asyncio.Future]],
    ) -> None:
        """Merge and fetch all requests for a single URL."""
        # Sort by offset
        reqs.sort(key=lambda r: r[0])

        # Build merged ranges
        merge_gap = self._store._merge_gap
        max_merge_size = self._store._max_merge_size
        merged: list[tuple[int, int, list[tuple[int, int, str, asyncio.Future]]]] = []

        for req in reqs:
            offset, size, key, future = req
            end = offset + size
            if merged:
                cur_start, cur_end, cur_reqs = merged[-1]
                new_end = max(cur_end, end)
                if offset <= cur_end + merge_gap and new_end - cur_start <= max_merge_size:
                    merged[-1] = (cur_start, new_end, cur_reqs + [req])
                    continue
            merged.append((offset, end, [req]))

        # Fetch each merged range in the thread pool
        loop = asyncio.get_running_loop()
        fetch_tasks = []
        for start, end, group_reqs in merged:
            fetch_tasks.append(
                loop.run_in_executor(
                    self._store._executor,
                    self._fetch_and_deliver,
                    loop, url, start, end, group_reqs,
                )
            )
        await asyncio.gather(*fetch_tasks)

    def _fetch_and_deliver(
        self,
        loop: asyncio.AbstractEventLoop,
        url: str,
        start: int,
        end: int,
        reqs: list[tuple[int, int, str, asyncio.Future]],
    ) -> None:
        """Fetch a merged byte range and deliver slices to individual futures."""
        try:
            raw = _read_bytes_from_url(url, start, end - start)
            for offset, size, key, future in reqs:
                chunk_data = raw[offset - start:offset - start + size]
                loop.call_soon_threadsafe(future.set_result, chunk_data)
        except Exception as exc:
            for _, _, _, future in reqs:
                loop.call_soon_threadsafe(future.set_exception, exc)


# -- Module-level helpers --


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
            length = field_type["configuration"]["length_bytes"] // 4
            return f"U{length}"
    raise ValueError(f"Unsupported zarr field type: {field_type}")


def _read_local_file(path: str, offset: int, length: int) -> bytes:
    """Read a byte range from a local file."""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def _read_bytes_from_url_or_path(url_or_path: str, offset: int, length: int) -> bytes:
    """Read a byte range from a URL or local file path."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return _read_bytes_from_url(url_or_path, offset, length)
    else:
        return _read_local_file(url_or_path, offset, length)


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
