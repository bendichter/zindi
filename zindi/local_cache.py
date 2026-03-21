"""Local SQLite-backed cache for remote chunk data.

Persists fetched byte ranges on disk so repeated reads of the same
remote chunks are served from the local cache instead of re-fetching
over HTTP. Mirrors lindi's LocalCache design.
"""

from __future__ import annotations

import os
import sqlite3


class ChunkTooLargeError(Exception):
    pass


class LocalCache:
    """Persistent local cache for remote chunk data.

    Parameters
    ----------
    cache_dir : str or None
        Directory to store the cache database. Defaults to ``~/.zindi/cache``.
    """

    def __init__(self, *, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.zindi/cache")
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)
        self._sqlite_client = _LocalCacheSQLiteClient(
            db_fname=os.path.join(self._cache_dir, "zindi_cache.db")
        )

    def get_remote_chunk(self, *, url: str, offset: int, size: int) -> bytes | None:
        """Retrieve a cached chunk, or None if not cached."""
        return self._sqlite_client.get_remote_chunk(url=url, offset=offset, size=size)

    def put_remote_chunk(self, *, url: str, offset: int, size: int, data: bytes) -> None:
        """Store a chunk in the cache.

        Raises
        ------
        ChunkTooLargeError
            If the chunk is >= 900 MB (SQLite BLOB limit).
        """
        if len(data) != size:
            raise ValueError("data size does not match size")
        self._sqlite_client.put_remote_chunk(url=url, offset=offset, size=size, data=data)


class _LocalCacheSQLiteClient:
    """SQLite backend for LocalCache."""

    def __init__(self, *, db_fname: str):
        self._conn = sqlite3.connect(db_fname)
        self._cursor = self._conn.cursor()
        self._cursor.execute("PRAGMA journal_mode=WAL")
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS remote_chunks (
                url TEXT,
                offset INTEGER,
                size INTEGER,
                data BLOB,
                PRIMARY KEY (url, offset, size)
            )
            """
        )
        self._conn.commit()

    def get_remote_chunk(self, *, url: str, offset: int, size: int) -> bytes | None:
        self._cursor.execute(
            "SELECT data FROM remote_chunks WHERE url = ? AND offset = ? AND size = ?",
            (url, offset, size),
        )
        row = self._cursor.fetchone()
        return row[0] if row is not None else None

    def put_remote_chunk(self, *, url: str, offset: int, size: int, data: bytes) -> None:
        if size >= 900_000_000:
            raise ChunkTooLargeError("Cannot store blobs larger than 900 MB in LocalCache")
        self._cursor.execute(
            "INSERT OR REPLACE INTO remote_chunks (url, offset, size, data) VALUES (?, ?, ?, ?)",
            (url, offset, size, data),
        )
        self._conn.commit()
