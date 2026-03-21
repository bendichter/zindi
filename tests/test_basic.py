"""End-to-end tests for zindi: HDF5 → zarr v3 RFS → read back."""

import json
import tempfile

import h5py
import numpy as np
import pytest
import zarr

from zindi import generate_rfs, open_rfs


def _create_test_hdf5(path: str) -> None:
    """Create a test HDF5 file with various dataset types."""
    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["description"] = "test file"
        f.attrs["version"] = 42

        # Group with attributes
        g = f.create_group("acquisition")
        g.attrs["rate"] = 30000.0
        g.attrs["unit"] = "volts"

        # Chunked float64 dataset (large enough to not be inlined)
        data = np.random.randn(2000).astype(np.float64)
        g.create_dataset("timeseries", data=data, chunks=(500,))

        # Chunked int32 dataset
        idata = np.arange(1500, dtype=np.int32)
        g.create_dataset("indices", data=idata, chunks=(300,))

        # Small dataset (will be inlined)
        small = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        g.create_dataset("small", data=small)

        # Scalar dataset
        g.create_dataset("scalar_int", data=np.int64(99))

        # String scalar
        g.create_dataset("name", data="my_timeseries")

        # Compressed dataset (gzip)
        compressed = np.random.randn(5000).astype(np.float64)
        g.create_dataset(
            "compressed", data=compressed, chunks=(1000,), compression="gzip", compression_opts=4
        )

        # 2D dataset
        data2d = np.arange(600, dtype=np.float32).reshape(20, 30)
        g.create_dataset("matrix", data=data2d, chunks=(10, 15))

        # Contiguous dataset (no chunking specified, large enough)
        contig = np.arange(2000, dtype=np.float64)
        g.create_dataset("contiguous", data=contig)

        # String array
        g.create_dataset("labels", data=np.array(["a", "bb", "ccc"], dtype=h5py.string_dtype()))

        # Second group with soft link
        proc = f.create_group("processing")
        proc.attrs["info"] = "processed data"
        f["processing/source"] = h5py.SoftLink("/acquisition/timeseries")

        # Boolean dataset
        g.create_dataset("mask", data=np.array([True, False, True]))

        # NaN/Inf in attributes
        g.attrs["nan_value"] = float("nan")
        g.attrs["inf_value"] = float("inf")

        # Object reference datasets
        target = g["timeseries"]
        g.create_dataset("refs_array", data=[target.ref, target.ref], dtype=h5py.ref_dtype)
        g.create_dataset("ref_scalar", data=target.ref, dtype=h5py.ref_dtype)


class TestBasicRoundtrip:
    """Test generating and reading back RFS."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.h5_path = f"{self.tmpdir}/test.h5"
        _create_test_hdf5(self.h5_path)
        self.rfs = generate_rfs(self.h5_path)

    def test_rfs_structure(self):
        """RFS has the expected top-level keys."""
        assert "refs" in self.rfs
        assert "version" in self.rfs
        assert self.rfs["version"] == 1

    def test_root_group(self):
        """Root group metadata is present and correct."""
        root_meta = json.loads(self.rfs["refs"]["zarr.json"])
        assert root_meta["zarr_format"] == 3
        assert root_meta["node_type"] == "group"
        assert root_meta["attributes"]["description"] == "test file"
        assert root_meta["attributes"]["version"] == 42

    def test_open_rfs_root(self):
        """Can open the RFS as a zarr group."""
        root = open_rfs(self.rfs)
        assert isinstance(root, zarr.Group)
        assert root.attrs["description"] == "test file"

    def test_chunked_float_dataset(self):
        """Chunked float64 dataset round-trips correctly."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/timeseries"]

        with h5py.File(self.h5_path, "r") as f:
            expected = f["acquisition/timeseries"][:]

        result = arr[:]
        np.testing.assert_array_equal(result, expected)

    def test_chunked_int_dataset(self):
        """Chunked int32 dataset round-trips correctly."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/indices"]

        with h5py.File(self.h5_path, "r") as f:
            expected = f["acquisition/indices"][:]

        result = arr[:]
        np.testing.assert_array_equal(result, expected)

    def test_small_inline_dataset(self):
        """Small dataset is inlined and reads back correctly."""
        root = open_rfs(self.rfs)
        result = root["acquisition/small"][:]
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_scalar_dataset(self):
        """Scalar dataset round-trips."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/scalar_int"]
        assert arr.shape == (1,)
        assert arr[0] == 99

    def test_compressed_dataset(self):
        """Gzip-compressed dataset reads correctly via numcodecs.zlib."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/compressed"]

        with h5py.File(self.h5_path, "r") as f:
            expected = f["acquisition/compressed"][:]

        result = arr[:]
        np.testing.assert_array_equal(result, expected)

    def test_2d_dataset(self):
        """2D chunked dataset round-trips."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/matrix"]

        with h5py.File(self.h5_path, "r") as f:
            expected = f["acquisition/matrix"][:]

        result = arr[:]
        np.testing.assert_array_equal(result, expected)

    def test_contiguous_dataset(self):
        """Contiguous (non-chunked) dataset round-trips."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/contiguous"]

        with h5py.File(self.h5_path, "r") as f:
            expected = f["acquisition/contiguous"][:]

        result = arr[:]
        np.testing.assert_array_equal(result, expected)

    def test_string_scalar(self):
        """String scalar dataset reads back."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/name"]
        val = arr[0]
        assert val == "my_timeseries"

    def test_string_array(self):
        """String array dataset reads back."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/labels"]
        result = arr[:]
        assert list(result) == ["a", "bb", "ccc"]

    def test_group_attrs(self):
        """Group attributes are preserved."""
        root = open_rfs(self.rfs)
        acq = root["acquisition"]
        assert acq.attrs["rate"] == 30000.0
        assert acq.attrs["unit"] == "volts"

    def test_nan_inf_attrs(self):
        """NaN and Inf are encoded as strings in attributes."""
        root_meta = json.loads(self.rfs["refs"]["acquisition/zarr.json"])
        attrs = root_meta["attributes"]
        assert attrs["nan_value"] == "NaN"
        assert attrs["inf_value"] == "Infinity"

    def test_soft_link_in_links(self):
        """Soft links appear in parent group's _LINKS attribute."""
        proc_meta = json.loads(self.rfs["refs"]["processing/zarr.json"])
        links = proc_meta["attributes"].get("_LINKS", [])
        assert len(links) == 1
        assert links[0]["name"] == "source"
        assert links[0]["path"] == "/acquisition/timeseries"
        assert links[0]["source"] == "."

    def test_boolean_dataset(self):
        """Boolean dataset round-trips."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/mask"]
        result = arr[:]
        np.testing.assert_array_equal(result, [True, False, True])

    def test_object_reference_array(self):
        """Object reference array stores target paths as plain strings."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/refs_array"]
        result = arr[:]
        assert len(result) == 2
        for val in result:
            assert str(val) == "/acquisition/timeseries"

    def test_object_reference_scalar(self):
        """Scalar object reference stores target path as plain string."""
        root = open_rfs(self.rfs)
        arr = root["acquisition/ref_scalar"]
        assert arr.shape == (1,)
        assert str(arr[0]) == "/acquisition/timeseries"

    def test_object_reference_dtype_attr(self):
        """Object reference datasets have _DTYPE in zarr.json metadata."""
        meta = json.loads(self.rfs["refs"]["acquisition/refs_array/zarr.json"])
        assert meta["attributes"]["_DTYPE"] == "object_reference"
        assert meta["data_type"] == "string"

        meta_scalar = json.loads(self.rfs["refs"]["acquisition/ref_scalar/zarr.json"])
        assert meta_scalar["attributes"]["_DTYPE"] == "object_reference"
        assert meta_scalar["attributes"]["_SCALAR"] is True

    def test_write_and_read_json(self):
        """RFS can be written to JSON and read back."""
        from zindi.generate_rfs import write_rfs

        json_path = f"{self.tmpdir}/test.zindi.json"
        write_rfs(self.rfs, json_path)

        root = open_rfs(json_path)
        assert root.attrs["description"] == "test file"

        arr = root["acquisition/timeseries"]
        with h5py.File(self.h5_path, "r") as f:
            expected = f["acquisition/timeseries"][:]
        np.testing.assert_array_equal(arr[:], expected)
