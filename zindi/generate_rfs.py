"""Generate a zarr v3 reference file system (RFS) from an HDF5 file.

The RFS is a JSON-serializable dict that describes the HDF5 file's group/array
hierarchy using zarr v3 metadata, with chunk references pointing to byte ranges
in the original HDF5 file. This allows zarr to read HDF5 data without copying.

The RFS follows the unified convention from:
  https://github.com/NeurodataWithoutBorders/lindi/issues/125
  https://github.com/hdmf-dev/hdmf-zarr/issues/335
"""

from __future__ import annotations

import base64
import json
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from .attr_conversion import h5_attr_to_zarr
from .h5_chunk_utils import (
    apply_to_all_chunk_info,
    get_byte_range_for_contiguous_dataset,
    get_max_num_chunks,
)
from .h5_filters_to_codecs import h5_filters_to_codec_pipeline


def generate_rfs(
    hdf5_url_or_path: str,
    *,
    local_hdf5_path: str | None = None,
    h5f: h5py.File | None = None,
) -> dict:
    """Generate a zarr v3 reference file system from an HDF5 file.

    Parameters
    ----------
    hdf5_url_or_path : str
        URL or local path of the HDF5 file. If a URL (http/https), it is
        used both as the source for reading metadata and as the target for
        chunk references. For remote files, zindi uses its built-in Remfile
        to read the file over HTTP.
    local_hdf5_path : str or None
        Path to a local copy of the HDF5 file to read metadata from. If
        provided, metadata is read from this local file but chunk references
        still point to hdf5_url_or_path.
    h5f : h5py.File or None
        An already-open h5py.File object. If provided, it is used directly
        and neither hdf5_url_or_path nor local_hdf5_path are opened.

    Returns
    -------
    dict
        A reference file system dict with keys "refs" and "version".
    """
    refs: dict[str, Any] = {}

    if h5f is not None:
        _process_group(h5f, "", refs, hdf5_url_or_path, h5f)
    elif local_hdf5_path is not None:
        with h5py.File(local_hdf5_path, "r") as opened:
            _process_group(opened, "", refs, hdf5_url_or_path, opened)
    elif hdf5_url_or_path.startswith("http://") or hdf5_url_or_path.startswith("https://"):
        from .remfile import ZindiRemfile

        remf = ZindiRemfile(hdf5_url_or_path)
        with h5py.File(remf, "r") as opened:
            _process_group(opened, "", refs, hdf5_url_or_path, opened)
    else:
        with h5py.File(hdf5_url_or_path, "r") as opened:
            _process_group(opened, "", refs, hdf5_url_or_path, opened)

    rfs = {"refs": refs, "version": 1}
    _apply_templates(rfs)
    return rfs


def write_rfs(rfs: dict, output_path: str) -> None:
    """Write a reference file system dict to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(rfs, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Internal: recursive group/dataset processing
# ---------------------------------------------------------------------------


def _process_group(
    item: h5py.Group,
    path: str,
    refs: dict,
    url: str,
    h5f: h5py.File,
) -> None:
    """Process an HDF5 group, adding zarr v3 metadata to refs."""
    # Check for soft link - if so, record in parent's _LINKS and skip children
    if path:
        link = h5f.get("/" + path, getlink=True)
        if isinstance(link, h5py.SoftLink):
            # Soft links are handled via _LINKS on the parent group
            # (added by the parent's processing). Don't recurse into the target.
            return

    # Build group zarr.json
    attrs = _collect_attrs(item, h5f=h5f, label=path or "(root)")

    # Collect _LINKS for any child soft links (unified convention)
    links = _collect_child_links(item, h5f)
    if links:
        attrs["_LINKS"] = links

    group_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": attrs,
    }
    meta_key = f"{path}/zarr.json" if path else "zarr.json"
    refs[meta_key] = json.dumps(group_meta, separators=(",", ":"))

    # Process children
    for name in item.keys():
        child_path = f"{path}/{name}" if path else name

        # Check if this child is a soft link
        child_link = h5f.get("/" + child_path, getlink=True)
        if isinstance(child_link, h5py.SoftLink):
            # Already recorded in parent _LINKS; skip
            continue

        child = item[name]
        if isinstance(child, h5py.Group):
            _process_group(child, child_path, refs, url, h5f)
        elif isinstance(child, h5py.Dataset):
            _process_dataset(child, child_path, refs, url, h5f)


def _process_dataset(
    ds: h5py.Dataset,
    path: str,
    refs: dict,
    url: str,
    h5f: h5py.File,
) -> None:
    """Process an HDF5 dataset, adding zarr v3 array metadata and chunk refs."""
    attrs = _collect_attrs(ds, h5f=h5f, label=path)

    shape = list(ds.shape)
    dtype = ds.dtype
    is_scalar = ds.ndim == 0

    # Handle scalar datasets
    if is_scalar:
        attrs["_SCALAR"] = True
        shape = [1]

    # Determine if this should be inlined
    inline = _should_inline(ds)

    if inline:
        _process_inline_dataset(ds, path, refs, attrs, is_scalar, h5f)
        return

    # Build codec pipeline
    codec_pipeline = h5_filters_to_codec_pipeline(ds)

    # Determine chunks
    chunks = list(ds.chunks) if ds.chunks else list(ds.shape)
    # Zarr doesn't allow zero-size chunks
    chunks = [max(c, 1) for c in chunks]

    # Zarr v3 data_type
    if dtype.kind == "V" and dtype.fields is not None:
        # Compound dtype
        data_type, compound_dtype_attr = _compound_dtype_to_zarr_v3(dtype)
        attrs["_COMPOUND_DTYPE"] = compound_dtype_attr
        fill_value = _encode_compound_fill_value(dtype)
    else:
        data_type = _numpy_dtype_to_zarr_v3(dtype)
        fill_value = _encode_fill_value(ds.fillvalue, dtype)

    array_meta: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": chunks},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codec_pipeline,
        "attributes": attrs,
        "storage_transformers": [],
    }

    refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

    # Add chunk references
    if np.prod(ds.shape) > 0:
        _add_chunk_refs(ds, path, refs, url)


def _process_inline_dataset(
    ds: h5py.Dataset,
    path: str,
    refs: dict,
    attrs: dict,
    is_scalar: bool,
    h5f: h5py.File,
) -> None:
    """Process a small dataset by inlining its data."""
    data = ds[()]

    if is_scalar:
        shape = [1]
        if isinstance(data, h5py.Reference):
            # Scalar object reference — store target path as plain string
            attrs["_SCALAR"] = True
            attrs["_DTYPE"] = "object_reference"
            target = h5f[data]
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))
            chunk_bytes = _encode_vlen_utf8([target.name])
            refs[f"{path}/c/0"] = "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            return
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, str):
            # String scalar
            attrs["_SCALAR"] = True
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))
            chunk_bytes = _encode_vlen_utf8([data])
            refs[f"{path}/c/0"] = "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            return
        else:
            data = np.array([data])
    else:
        if h5py.check_dtype(ref=ds.dtype) == h5py.Reference:
            # Object reference array — store target paths as plain strings
            data = ds[...]
            path_strs = []
            for item in np.nditer(data, flags=["refs_ok"]):
                val = item.item()
                if isinstance(val, h5py.Reference):
                    target = h5f[val]
                    path_strs.append(target.name)
                else:
                    path_strs.append("")

            shape = list(ds.shape)
            attrs["_DTYPE"] = "object_reference"
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

            chunk_bytes = _encode_vlen_utf8(path_strs)
            chunk_key = "c/" + "/".join(["0"] * max(ds.ndim, 1))
            refs[f"{path}/{chunk_key}"] = (
                "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            )
            return

        if ds.dtype.kind in ("O", "U", "S"):
            # String array
            data = ds[...]
            str_data = []
            for item in np.nditer(data, flags=["refs_ok"]):
                val = item.item()
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                str_data.append(str(val) if val is not None else "")

            shape = list(ds.shape)
            array_meta = _make_string_array_meta(shape, attrs)
            refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

            chunk_bytes = _encode_vlen_utf8(str_data)
            chunk_key = "c/" + "/".join(["0"] * max(ds.ndim, 1))
            refs[f"{path}/{chunk_key}"] = (
                "base64:" + base64.b64encode(chunk_bytes).decode("ascii")
            )
            return

    # Compound inline data
    if ds.dtype.kind == "V" and ds.dtype.fields is not None:
        shape = list(data.shape)
        dtype = data.dtype
        data_type, compound_dtype_attr = _compound_dtype_to_zarr_v3(dtype)
        attrs["_COMPOUND_DTYPE"] = compound_dtype_attr
        fill_value = _encode_compound_fill_value(dtype)

        codec_pipeline = [
            {"name": "bytes", "configuration": {"endian": "little"}}
        ]

        array_meta: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": data_type,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": fill_value,
            "codecs": codec_pipeline,
            "attributes": attrs,
            "storage_transformers": [],
        }

        refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

        # Ensure little-endian byte order
        if dtype.byteorder == ">":
            data = data.astype(dtype.newbyteorder("<"))
        chunk_bytes = data.tobytes()
        chunk_key = "c/" + "/".join(["0"] * max(len(shape), 1))
        _add_inline_ref(refs, f"{path}/{chunk_key}", chunk_bytes)
        return

    # Numeric inline data
    shape = list(data.shape)
    dtype = data.dtype
    data_type = _numpy_dtype_to_zarr_v3(dtype)
    fill_value = _encode_fill_value(ds.fillvalue, dtype)

    codec_pipeline = [
        {"name": "bytes", "configuration": {"endian": "little"}}
    ]

    array_meta: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codec_pipeline,
        "attributes": attrs,
        "storage_transformers": [],
    }

    refs[f"{path}/zarr.json"] = json.dumps(array_meta, separators=(",", ":"))

    # Inline the chunk data
    # Ensure little-endian byte order
    if dtype.byteorder == ">":
        data = data.astype(dtype.newbyteorder("<"))
    chunk_bytes = data.tobytes()
    chunk_key = "c/" + "/".join(["0"] * max(len(shape), 1))
    _add_inline_ref(refs, f"{path}/{chunk_key}", chunk_bytes)


def _make_string_array_meta(shape: list[int], attrs: dict) -> dict:
    """Create zarr v3 array metadata for a string array."""
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "string",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": "",
        "codecs": [
            {"name": "vlen-utf8", "configuration": {}},
        ],
        "attributes": attrs,
        "storage_transformers": [],
    }


def _add_chunk_refs(
    ds: h5py.Dataset,
    path: str,
    refs: dict,
    url: str,
) -> None:
    """Add chunk references for a non-inline dataset."""
    if ds.chunks is not None:
        # Chunked dataset
        chunk_size = ds.chunks
        num_chunks = get_max_num_chunks(shape=ds.shape, chunk_size=chunk_size)
        pbar = tqdm(
            total=num_chunks,
            desc=f"Chunk refs for {path}",
            leave=True,
            delay=2,
        )

        def store_chunk_info(chunk_info: Any) -> None:
            chunk_offset = chunk_info.chunk_offset
            byte_offset = chunk_info.byte_offset
            byte_count = chunk_info.size
            # zarr v3 chunk key: c/<idx0>/<idx1>/...
            indices = [str(a // b) for a, b in zip(chunk_offset, chunk_size)]
            chunk_key = f"{path}/c/" + "/".join(indices)
            refs[chunk_key] = [url, byte_offset, byte_count]
            pbar.update()

        apply_to_all_chunk_info(ds, store_chunk_info)
        pbar.close()
    else:
        # Contiguous dataset - single chunk
        byte_offset, byte_count = get_byte_range_for_contiguous_dataset(ds)
        indices = "/".join(["0"] * ds.ndim)
        chunk_key = f"{path}/c/{indices}"
        refs[chunk_key] = [url, byte_offset, byte_count]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_attrs(
    item: h5py.Group | h5py.Dataset, *, h5f: h5py.File, label: str
) -> dict[str, Any]:
    """Collect and convert all attributes from an HDF5 item."""
    attrs: dict[str, Any] = {}
    for key in item.attrs:
        try:
            val = h5_attr_to_zarr(item.attrs[key], label=f"{label}.{key}", h5f=h5f)
            if val is not None:
                attrs[key] = val
        except (ValueError, Exception):
            # Skip attributes that can't be converted
            pass
    return attrs


def _collect_child_links(item: h5py.Group, h5f: h5py.File) -> list[dict]:
    """Collect soft links among children of a group (unified convention)."""
    links = []
    for name in item.keys():
        child_path = item.name.rstrip("/") + "/" + name
        if child_path.startswith("//"):
            child_path = child_path[1:]
        child_link = h5f.get(child_path, getlink=True)
        if isinstance(child_link, h5py.SoftLink):
            links.append({
                "name": name,
                "source": ".",
                "path": child_link.path,
            })
    return links


def _should_inline(ds: h5py.Dataset) -> bool:
    """Decide whether a dataset should be inlined in the RFS."""
    if ds.ndim == 0:
        return True
    if ds.dtype.kind in ("O", "U", "S"):
        # String/object data - always inline
        return True
    if ds.dtype.kind == "V" and ds.dtype.fields is not None:
        # Compound type - inline small ones, use byte-range refs for large
        if ds.size < 1000:
            return True
        return False
    if ds.size < 1000:
        return True
    return False


def _numpy_dtype_to_zarr_v3(dtype: np.dtype) -> str:
    """Convert a numpy dtype to a zarr v3 data_type string."""
    kind = dtype.kind
    itemsize = dtype.itemsize
    mapping = {
        ("f", 2): "float16",
        ("f", 4): "float32",
        ("f", 8): "float64",
        ("i", 1): "int8",
        ("i", 2): "int16",
        ("i", 4): "int32",
        ("i", 8): "int64",
        ("u", 1): "uint8",
        ("u", 2): "uint16",
        ("u", 4): "uint32",
        ("u", 8): "uint64",
        ("b", 1): "bool",
    }
    result = mapping.get((kind, itemsize))
    if result is None:
        raise ValueError(f"Unsupported dtype for zarr v3: {dtype}")
    return result


def _compound_dtype_to_zarr_v3(dtype: np.dtype) -> tuple[dict, list]:
    """Convert a numpy structured dtype to zarr v3 structured data_type and _COMPOUND_DTYPE attr.

    Returns
    -------
    (data_type_dict, compound_dtype_attr)
        data_type_dict: zarr v3 data_type for zarr.json
        compound_dtype_attr: list of {"name", "dtype"} dicts for _COMPOUND_DTYPE attribute
    """
    fields_zarr = []
    fields_attr = []
    for field_name in dtype.names:
        field_dtype = dtype[field_name]
        if field_dtype.kind == "S":
            # Fixed-length byte string
            zarr_type = {
                "name": "null_terminated_bytes",
                "configuration": {"length_bytes": field_dtype.itemsize},
            }
            attr_type = str(field_dtype)  # e.g. "|S10"
        else:
            zarr_type = _numpy_dtype_to_zarr_v3(field_dtype)
            attr_type = zarr_type
        fields_zarr.append([field_name, zarr_type])
        fields_attr.append({"name": field_name, "dtype": attr_type})

    data_type_dict = {
        "name": "structured",
        "configuration": {"fields": fields_zarr},
    }
    return data_type_dict, fields_attr


def _encode_compound_fill_value(dtype: np.dtype) -> str:
    """Encode a compound fill value as base64 zero bytes."""
    return base64.b64encode(b"\x00" * dtype.itemsize).decode("ascii")


def _encode_fill_value(fill_value: Any, dtype: np.dtype) -> Any:
    """Encode a fill value for JSON serialization."""
    if fill_value is None:
        return 0 if dtype.kind in ("i", "u", "f") else None
    if isinstance(fill_value, (np.integer,)):
        return int(fill_value)
    if isinstance(fill_value, (np.floating,)):
        v = float(fill_value)
        if np.isnan(v):
            return "NaN"
        if v == float("inf"):
            return "Infinity"
        if v == float("-inf"):
            return "-Infinity"
        return v
    if isinstance(fill_value, (np.bool_,)):
        return bool(fill_value)
    if isinstance(fill_value, bytes):
        return fill_value.decode("utf-8", errors="replace")
    return fill_value


def _encode_vlen_utf8(strings: list[str]) -> bytes:
    """Encode a list of strings in numcodecs vlen-utf8 format.

    Format: 4-byte LE count, then for each string: 4-byte LE length + utf8 bytes.
    """
    import struct

    parts = [struct.pack("<I", len(strings))]
    for s in strings:
        encoded = s.encode("utf-8")
        parts.append(struct.pack("<I", len(encoded)))
        parts.append(encoded)
    return b"".join(parts)


def _add_inline_ref(refs: dict, key: str, data: bytes) -> None:
    """Add inline data to refs, base64-encoding if needed."""
    if data.startswith(b"base64:"):
        refs[key] = "base64:" + base64.b64encode(data).decode("ascii")
    else:
        try:
            refs[key] = data.decode("ascii")
        except UnicodeDecodeError:
            refs[key] = "base64:" + base64.b64encode(data).decode("ascii")


def _apply_templates(rfs: dict) -> None:
    """Replace frequently-used URLs with template placeholders."""
    refs = rfs["refs"]
    url_counts: dict[str, int] = {}
    for val in refs.values():
        if isinstance(val, list) and len(val) == 3:
            url = val[0]
            url_counts[url] = url_counts.get(url, 0) + 1

    templates: dict[str, str] = {}
    template_idx = 0
    for url, count in url_counts.items():
        if count >= 5:
            template_key = f"u{template_idx}"
            templates[template_key] = url
            template_idx += 1

    if not templates:
        return

    # Build reverse lookup
    url_to_template = {url: key for key, url in templates.items()}

    # Replace URLs with template references
    for ref_key, val in refs.items():
        if isinstance(val, list) and len(val) == 3:
            url = val[0]
            if url in url_to_template:
                val[0] = "{{" + url_to_template[url] + "}}"

    rfs["templates"] = templates
