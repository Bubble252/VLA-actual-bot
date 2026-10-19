#!/usr/bin/env python3
"""Forensically recover complete RGB frames from truncated teleop HDF5 files.

This tool never modifies an input file.  It repairs just enough metadata in a
temporary copy to read the surviving dataset descriptions, then scans legacy
HDF5 v1 chunk B-tree leaves directly.  Complete time chunks are decoded and
written as MP4 files.  Missing pixels/robot states are never fabricated.

The output intentionally has no LeRobot ``meta/info.json`` or parquet files:
without ``action`` and ``observations/qpos`` it is recovered media, not a valid
training dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import mmap
import shutil
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
TREE_SIGNATURE = b"TREE"
UNDEFINED_ADDRESS = (1 << 64) - 1


@dataclass(frozen=True)
class ChunkRecord:
    coords: tuple[int, ...]
    address: int
    stored_size: int
    filter_mask: int
    leaf_address: int


@dataclass
class LeafNode:
    address: int
    left: int
    right: int
    records: list[ChunkRecord]


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _patch_probe_copy(source: Path, probe: Path) -> dict[str, int]:
    """Patch a disposable copy so h5py can inspect surviving metadata."""
    shutil.copyfile(source, probe)
    actual_size = source.stat().st_size

    with probe.open("r+b") as f:
        head = f.read(128)
        if head[:8] != HDF5_SIGNATURE:
            raise ValueError(f"{source}: not an HDF5 file")
        superblock_version = head[8]
        offset_size = head[13]
        length_size = head[14]
        if superblock_version != 0 or offset_size != 8 or length_size != 8:
            raise ValueError(
                f"{source}: unsupported HDF5 superblock "
                f"v{superblock_version}, offsets={offset_size}, lengths={length_size}"
            )

        # HDF5 superblock v0: base, free-space, EOF, driver addresses begin at 24.
        eof_field = 24 + 2 * offset_size
        stored_eof = struct.unpack_from("<Q", head, eof_field)[0]

        # The root symbol-table entry follows the four superblock addresses.
        root_entry = 24 + 4 * offset_size
        root_object_header = struct.unpack_from("<Q", head, root_entry + offset_size)[0]
        cache_type = struct.unpack_from("<I", head, root_entry + 2 * offset_size)[0]
        if cache_type != 1:
            raise ValueError(f"{source}: root group is not a legacy symbol table")
        root_btree = struct.unpack_from("<Q", head, root_entry + 2 * offset_size + 8)[0]
        root_heap = struct.unpack_from("<Q", head, root_entry + 3 * offset_size + 8)[0]

        # Root object header v1 starts with a continuation message in these files.
        f.seek(root_object_header)
        object_head = f.read(64)
        if not object_head or object_head[0] != 1:
            raise ValueError(f"{source}: unsupported root object header")
        first_message = root_object_header + 16
        message_type, message_size = struct.unpack_from("<HH", object_head, 16)
        if message_type != 0x10 or message_size != 2 * offset_size:
            raise ValueError(f"{source}: root continuation message not found")
        continuation = struct.unpack_from("<Q", object_head, 24)[0]

        f.seek(continuation)
        nested = f.read(24)
        nested_type, nested_size = struct.unpack_from("<HH", nested, 0)
        if nested_size != 2 * offset_size or nested_type not in (0x10, 0x11):
            raise ValueError(f"{source}: unsupported final root message")

        # Clamp the temporary file's EOA to bytes that actually exist.
        f.seek(eof_field)
        f.write(struct.pack("<Q", actual_size))

        if nested_type == 0x10:
            # The missing final continuation held the symbol-table message.  The
            # superblock caches both addresses, so that message can be reconstructed.
            f.seek(continuation)
            f.write(
                struct.pack(
                    "<HHB3xQQ",
                    0x11,  # H5O_STAB_ID: legacy group symbol-table message
                    2 * offset_size,
                    0,
                    root_btree,
                    root_heap,
                )
            )

    return {
        "actual_size": actual_size,
        "stored_eof": stored_eof,
        "root_object_header": root_object_header,
        "root_continuation": continuation,
        "root_first_message": first_message,
    }


def _inspect_surviving_metadata(source: Path) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        prefix=f"{source.stem}_recovery_", suffix=".hdf5", dir="/tmp", delete=False
    ) as tmp:
        probe = Path(tmp.name)

    try:
        superblock = _patch_probe_copy(source, probe)
        out: dict[str, Any] = {
            "superblock": superblock,
            "attrs": {},
            "camera_names": [],
            "image_datasets": {},
            "robot_datasets": {},
        }
        with h5py.File(probe, "r") as f:
            out["attrs"] = {str(k): _jsonable(v) for k, v in f.attrs.items()}
            raw_camera_names = out["attrs"].get("camera_names", "[]")
            try:
                camera_names = json.loads(str(raw_camera_names))
            except json.JSONDecodeError:
                camera_names = []

            try:
                image_group = f["observations/images"]
                linked_names = list(image_group.keys())
            except Exception:
                image_group = None
                linked_names = []
            camera_names = [str(x) for x in (camera_names or linked_names)]
            out["camera_names"] = camera_names

            if image_group is not None:
                for name in camera_names:
                    try:
                        dset = image_group[name]
                        out["image_datasets"][name] = {
                            "available_metadata": True,
                            "shape": [int(x) for x in dset.shape],
                            "chunks": [int(x) for x in dset.chunks or ()],
                            "dtype": dset.dtype.str,
                            "compression": dset.compression,
                            "compression_opts": _jsonable(dset.compression_opts),
                            "shuffle": bool(dset.shuffle),
                            "fletcher32": bool(dset.fletcher32),
                        }
                    except Exception as exc:
                        out["image_datasets"][name] = {
                            "available_metadata": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }

            for key in ("action", "observations/qpos", "observations/eef"):
                try:
                    dset = f[key]
                    shape = [int(x) for x in dset.shape]
                    # Access one element as well: an object header alone is not data.
                    _ = dset[0]
                    out["robot_datasets"][key] = {
                        "readable": True,
                        "shape": shape,
                        "dtype": dset.dtype.str,
                    }
                except Exception as exc:
                    out["robot_datasets"][key] = {
                        "readable": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        return out
    finally:
        probe.unlink(missing_ok=True)


def _scan_chunk_leaves(path: Path, chunk_shape: tuple[int, ...]) -> list[LeafNode]:
    """Scan surviving legacy raw-chunk B-tree leaf nodes without a root node."""
    rank = len(chunk_shape)
    key_dimensions = rank + 1  # legacy HDF5 stores one extra byte-offset dimension
    key_size = 8 + 8 * key_dimensions
    record_stride = key_size + 8
    uncompressed_size = int(np.prod(chunk_shape, dtype=np.int64))
    leaves: list[LeafNode] = []

    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        position = 0
        file_size = len(mm)
        while True:
            position = mm.find(TREE_SIGNATURE, position)
            if position < 0:
                break
            if position + 24 > file_size:
                break
            node_type, level, entries = struct.unpack_from("<BBH", mm, position + 4)
            if node_type != 1 or level != 0 or entries == 0 or entries > 4096:
                position += 4
                continue
            node_end = position + 24 + entries * record_stride + key_size
            if node_end > file_size:
                position += 4
                continue

            left, right = struct.unpack_from("<QQ", mm, position + 8)
            records: list[ChunkRecord] = []
            valid_layout = True
            for index in range(entries):
                key = position + 24 + index * record_stride
                stored_size, filter_mask = struct.unpack_from("<II", mm, key)
                all_coords = struct.unpack_from(f"<{key_dimensions}Q", mm, key + 8)
                coords = tuple(int(x) for x in all_coords[:rank])
                extra_offset = int(all_coords[-1])
                data_address = struct.unpack_from("<Q", mm, key + key_size)[0]

                if (
                    stored_size <= 0
                    or stored_size > uncompressed_size * 2
                    or extra_offset != 0
                    or any(c % step != 0 for c, step in zip(coords, chunk_shape))
                    or data_address == UNDEFINED_ADDRESS
                ):
                    valid_layout = False
                    break
                records.append(
                    ChunkRecord(
                        coords=coords,
                        address=int(data_address),
                        stored_size=int(stored_size),
                        filter_mask=int(filter_mask),
                        leaf_address=position,
                    )
                )

            if valid_layout and records:
                leaves.append(
                    LeafNode(
                        address=position,
                        left=int(left),
                        right=int(right),
                        records=records,
                    )
                )
            position += 4
    return leaves


def _leaf_components(leaves: list[LeafNode]) -> list[list[LeafNode]]:
    by_address = {node.address: node for node in leaves}
    adjacency: dict[int, set[int]] = {address: set() for address in by_address}
    for node in leaves:
        for sibling in (node.left, node.right):
            if sibling in by_address:
                adjacency[node.address].add(sibling)
                adjacency[sibling].add(node.address)

    components: list[list[LeafNode]] = []
    unseen = set(by_address)
    while unseen:
        start = unseen.pop()
        stack = [start]
        addresses = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    addresses.append(neighbor)
        components.append([by_address[a] for a in addresses])

    def first_data_address(component: list[LeafNode]) -> int:
        return min(record.address for node in component for record in node.records)

    return sorted(components, key=first_data_address)


def _component_records(component: Iterable[LeafNode]) -> dict[tuple[int, ...], ChunkRecord]:
    records: dict[tuple[int, ...], ChunkRecord] = {}
    for node in component:
        for record in node.records:
            records[record.coords] = record
    return records


def _decode_chunk(mm: mmap.mmap, record: ChunkRecord, expected_size: int) -> bytes:
    end = record.address + record.stored_size
    if end > len(mm):
        raise EOFError(f"chunk ends at {end}, file ends at {len(mm)}")
    stored = mm[record.address:end]
    if record.filter_mask & 1:
        raw = bytes(stored)
    else:
        raw = zlib.decompress(stored)
    if len(raw) != expected_size:
        raise ValueError(f"decoded chunk has {len(raw)} bytes, expected {expected_size}")
    return raw


def _expected_spatial_coords(
    shape: tuple[int, ...], chunk_shape: tuple[int, ...], time_start: int
) -> Iterable[tuple[int, ...]]:
    axes = [range(0, shape[i], chunk_shape[i]) for i in range(1, len(shape))]
    for rest in itertools.product(*axes):
        yield (time_start, *rest)


def _complete_time_starts(
    records: dict[tuple[int, ...], ChunkRecord],
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    file_size: int,
) -> list[int]:
    complete: list[int] = []
    for time_start in range(0, shape[0], chunk_shape[0]):
        expected = list(_expected_spatial_coords(shape, chunk_shape, time_start))
        if all(
            coords in records
            and records[coords].address + records[coords].stored_size <= file_size
            for coords in expected
        ):
            complete.append(time_start)
    return complete


def _prefix_time_starts(complete: list[int], chunk_time: int) -> list[int]:
    found = set(complete)
    prefix: list[int] = []
    current = 0
    while current in found:
        prefix.append(current)
        current += chunk_time
    return prefix


def _start_ffmpeg(path: Path, width: int, height: int, fps: float) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_recovered_video(
    *,
    source: Path,
    output: Path,
    records: dict[tuple[int, ...], ChunkRecord],
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    dtype: np.dtype[Any],
    time_starts: list[int],
    fps: float,
) -> dict[str, Any]:
    if dtype != np.dtype(np.uint8):
        raise ValueError(f"unsupported image dtype {dtype}; expected uint8")
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"unsupported image shape {shape}; expected T,H,W,3")

    height, width = shape[1], shape[2]
    temporary_output = output.with_suffix(".partial.mp4")
    process = _start_ffmpeg(temporary_output, width, height, fps)
    assert process.stdin is not None
    expected_chunk_bytes = int(np.prod(chunk_shape, dtype=np.int64)) * dtype.itemsize
    written = 0
    pixel_sum = 0
    pixel_count = 0
    pixel_min = 255
    pixel_max = 0

    try:
        with source.open("rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:
            for time_start in time_starts:
                block_length = min(chunk_shape[0], shape[0] - time_start)
                block = np.empty((block_length, *shape[1:]), dtype=dtype)
                for coords in _expected_spatial_coords(shape, chunk_shape, time_start):
                    record = records[coords]
                    raw = _decode_chunk(mm, record, expected_chunk_bytes)
                    chunk = np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)

                    destination_slices = [slice(0, block_length)]
                    source_slices = [slice(0, block_length)]
                    for axis in range(1, len(shape)):
                        start = coords[axis]
                        length = min(chunk_shape[axis], shape[axis] - start)
                        destination_slices.append(slice(start, start + length))
                        source_slices.append(slice(0, length))
                    block[tuple(destination_slices)] = chunk[tuple(source_slices)]

                for frame in block:
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
                written += block_length
                pixel_sum += int(block.sum(dtype=np.uint64))
                pixel_count += int(block.size)
                pixel_min = min(pixel_min, int(block.min()))
                pixel_max = max(pixel_max, int(block.max()))
                print(
                    f"      recovered {written} frames -> {output.name}",
                    flush=True,
                )
    finally:
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr.strip()}")
    temporary_output.replace(output)
    return {
        "video": str(output),
        "frames": written,
        "fps": fps,
        "resolution": [height, width],
        "pixel_mean": pixel_sum / pixel_count if pixel_count else None,
        "pixel_min": pixel_min if pixel_count else None,
        "pixel_max": pixel_max if pixel_count else None,
    }


def recover_file(
    source: Path,
    output_root: Path,
    episode_index: int,
) -> dict[str, Any]:
    print(f"--- {source.name} ---", flush=True)
    metadata = _inspect_surviving_metadata(source)
    attrs = metadata["attrs"]
    camera_names: list[str] = metadata["camera_names"]
    fps = float(attrs.get("fps", 20.0))

    readable_image_metadata = [
        item
        for item in metadata["image_datasets"].values()
        if item.get("available_metadata")
    ]
    if not readable_image_metadata:
        raise RuntimeError(f"{source}: no surviving image dataset metadata")
    template = readable_image_metadata[0]
    shape = tuple(int(x) for x in template["shape"])
    chunk_shape = tuple(int(x) for x in template["chunks"])
    dtype = np.dtype(template["dtype"])
    if template.get("compression") != "gzip":
        raise ValueError(f"{source}: unsupported compression {template.get('compression')!r}")
    if template.get("shuffle") or template.get("fletcher32"):
        raise ValueError(f"{source}: unsupported HDF5 filter pipeline")

    leaves = _scan_chunk_leaves(source, chunk_shape)
    components = [c for c in _leaf_components(leaves) if len(c) >= 2]
    file_size = source.stat().st_size
    print(
        f"  found {len(leaves)} chunk leaves in {len(components)} dataset component(s)",
        flush=True,
    )

    cameras: dict[str, Any] = {}
    for component_index, component in enumerate(components):
        name = (
            camera_names[component_index]
            if component_index < len(camera_names)
            else f"unknown_camera_{component_index}"
        )
        dataset_meta = metadata["image_datasets"].get(name, {})
        component_shape = (
            tuple(int(x) for x in dataset_meta["shape"])
            if dataset_meta.get("available_metadata")
            else shape
        )
        component_chunks = (
            tuple(int(x) for x in dataset_meta["chunks"])
            if dataset_meta.get("available_metadata")
            else chunk_shape
        )
        records = _component_records(component)
        present_records = {
            coords: record
            for coords, record in records.items()
            if record.address + record.stored_size <= file_size
        }
        complete = _complete_time_starts(
            records, component_shape, component_chunks, file_size
        )
        prefix = _prefix_time_starts(complete, component_chunks[0])
        recoverable_frames = sum(
            min(component_chunks[0], component_shape[0] - start) for start in prefix
        )
        print(
            f"  {name}: {len(present_records)}/{len(records)} chunks present, "
            f"{recoverable_frames}/{component_shape[0]} complete prefix frames",
            flush=True,
        )

        camera_report: dict[str, Any] = {
            "declared_shape": list(component_shape),
            "chunk_shape": list(component_chunks),
            "leaf_nodes": len(component),
            "indexed_chunks": len(records),
            "physically_complete_chunks": len(present_records),
            "complete_time_chunk_starts": complete,
            "complete_prefix_frames": recoverable_frames,
            "metadata_inferred_from_first_camera": not bool(
                dataset_meta.get("available_metadata")
            ),
        }
        if prefix:
            video_key = f"observation.images.{name}"
            output = (
                output_root
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            camera_report.update(
                _write_recovered_video(
                    source=source,
                    output=output,
                    records=records,
                    shape=component_shape,
                    chunk_shape=component_chunks,
                    dtype=dtype,
                    time_starts=prefix,
                    fps=fps,
                )
            )
        cameras[name] = camera_report

    stored_eof = int(metadata["superblock"]["stored_eof"])
    return {
        "episode_index": episode_index,
        "source": str(source),
        "source_file": source.name,
        "actual_bytes": file_size,
        "declared_bytes": stored_eof,
        "missing_bytes": max(0, stored_eof - file_size),
        "task": str(attrs.get("task", "manipulation task")),
        "fps": fps,
        "attrs": attrs,
        "robot_datasets": metadata["robot_datasets"],
        "cameras": cameras,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover complete RGB frames from truncated teleop HDF5 files."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow an existing output directory and atomically replace matching videos.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} under {input_dir}")
    # A failed probe may leave empty directories but never useful output.
    if (
        output_dir.exists()
        and any(path.is_file() for path in output_dir.rglob("*"))
        and not args.resume
    ):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")

    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format_status": "PARTIAL_MEDIA_RECOVERY_NOT_A_VALID_LEROBOT_DATASET",
        "reason": (
            "All source files are truncated before action/observation.state. "
            "Only fully intact RGB chunks were exported; no values were fabricated."
        ),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "episodes": [],
    }

    for episode_index, source in enumerate(files):
        report["episodes"].append(
            recover_file(source, output_dir, episode_index)
        )

    report_path = output_dir / "meta" / "recovery_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 72)
    print(f"Recovered media: {output_dir}")
    print(f"Recovery report: {report_path}")
    print("No info.json/parquet was written: robot action/state data is absent.")


if __name__ == "__main__":
    main()
