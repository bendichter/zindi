"""Tests for local chunk caching."""

import tempfile

import h5py
import numpy as np
import pytest

from zindi import LocalCache, generate_rfs, open_rfs
from zindi.local_cache import ChunkTooLargeError


class TestLocalCacheDirect:
    """Test LocalCache get/put operations directly."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = LocalCache(cache_dir=self.tmpdir)

    def test_put_and_get(self):
        """Stored chunk can be retrieved."""
        data = b"hello world"
        self.cache.put_remote_chunk(url="http://example.com/f.h5", offset=100, size=len(data), data=data)
        result = self.cache.get_remote_chunk(url="http://example.com/f.h5", offset=100, size=len(data))
        assert result == data

    def test_cache_miss(self):
        """Uncached chunk returns None."""
        result = self.cache.get_remote_chunk(url="http://example.com/f.h5", offset=0, size=10)
        assert result is None

    def test_different_keys(self):
        """Different (url, offset, size) tuples are independent."""
        self.cache.put_remote_chunk(url="http://a.com/f.h5", offset=0, size=3, data=b"aaa")
        self.cache.put_remote_chunk(url="http://b.com/f.h5", offset=0, size=3, data=b"bbb")
        assert self.cache.get_remote_chunk(url="http://a.com/f.h5", offset=0, size=3) == b"aaa"
        assert self.cache.get_remote_chunk(url="http://b.com/f.h5", offset=0, size=3) == b"bbb"

    def test_overwrite(self):
        """Storing to the same key overwrites."""
        self.cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=3, data=b"old")
        self.cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=3, data=b"new")
        assert self.cache.get_remote_chunk(url="http://x.com/f.h5", offset=0, size=3) == b"new"

    def test_size_mismatch_raises(self):
        """Mismatched data length raises ValueError."""
        with pytest.raises(ValueError, match="data size does not match"):
            self.cache.put_remote_chunk(url="http://x.com/f.h5", offset=0, size=5, data=b"abc")

    def test_chunk_too_large(self):
        """Chunks >= 900 MB raise ChunkTooLargeError."""
        with pytest.raises(ChunkTooLargeError):
            self.cache.put_remote_chunk(
                url="http://x.com/f.h5", offset=0, size=900_000_000, data=b"\x00" * 900_000_000
            )


class TestLocalCacheIntegration:
    """Test cache integration with RFS read path."""

    def test_cache_with_local_hdf5(self):
        """Cache works end-to-end with a local HDF5 file.

        Local file reads are not cached (only remote URLs), but the
        cache parameter should not break anything.
        """
        tmpdir = tempfile.mkdtemp()
        h5_path = f"{tmpdir}/test.h5"

        data = np.arange(2000, dtype=np.float64)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("arr", data=data, chunks=(500,))

        rfs = generate_rfs(h5_path)
        cache = LocalCache(cache_dir=tmpdir)
        root = open_rfs(rfs, local_cache=cache)
        result = root["arr"][:]
        np.testing.assert_array_equal(result, data)
