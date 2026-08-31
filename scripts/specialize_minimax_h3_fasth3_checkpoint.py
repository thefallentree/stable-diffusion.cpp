#!/usr/bin/env python3
"""Create a zero-copy, schedule-specialized FastH3 transformer manifest.

MiniMax-H3's full timestep/AdaLN path accounts for tens of gigabytes even
after the ordinary linear weights are quantized.  A four-step FastH3 student
only visits eight distinct clean-time values: four for video (shift 12) and
four for audio (shift 3).  This tool evaluates the full AdaLN projections at
those exact values, writes compact FP32 lookup tables, and emits a new
safetensors index that:

* references unchanged source shards through relative symbolic links;
* omits the full time embedder and 51 full AdaLN projections;
* adds 50 block tables and one final-layer table;
* fuses Diffusers Q/K/V weights into the native H3 QKV layout;
* materializes the native RoPE inverse-frequency tensor;
* optionally omits VSA gate tensors for a dense-attention runtime.

The output is a conversion manifest, not another copy of the checkpoint.  It
is intended as input to ``sd-cli -M convert --convert-name``.  The generated
GGUF requires stable-diffusion.cpp's native FastH3 direct-AdaLN support and the
published ``999,749,500,250`` schedule.

Example::

    PYTHONPATH=/path/to/llama.cpp/gguf-py python3 \
      scripts/specialize_minimax_h3_fasth3_checkpoint.py \
      --dense-transformer /models/FastH3/transformer \
      --output-dir /models/FastH3-specialized-transformer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np

from specialize_minimax_h3_fasth3 import (
    COPY_BUFFER_SIZE,
    FASTH3_TRAINING_INDICES,
    INDEX_NAME,
    SafeTensorFile,
    TIME_EMBEDDER_KEYS,
    compute_adaln_input,
    direct_adaln_table,
    load_index,
    schedule_union,
    shifted_sigmas,
)


DIRECT_SHARD = "fasth3_direct_adaln.safetensors"
NATIVE_QKV_SHARD = "fasth3_native_qkv.safetensors"
SPECIALIZATION_REPORT = "fasth3_specialization.json"
FINAL_ADALN_KEYS = {"norm_out.linear.weight", "norm_out.linear.bias"}
VSA_GATE_RE = re.compile(r"^transformer_blocks\.\d+\.attn\.to_gate_compress\.")
QKV_RE = re.compile(
    r"^(transformer_blocks\.\d+|token_refiner\.refiner_blocks\.\d+)"
    r"\.attn\.to_([qkv])\.weight$"
)
QKV_ORDER = ("q", "k", "v")
ROPE_INV_FREQ = "rope.inv_freq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dense-transformer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument(
        "--drop-vsa-gates",
        action="store_true",
        help=(
            "omit to_gate_compress tensors for a dense-attention runtime; "
            "required when the source contains VSA gates"
        ),
    )
    parser.add_argument(
        "--keep-full-adaln",
        action="store_true",
        help=(
            "retain the checkpoint's full time embedder and AdaLN projections; "
            "only fuse Q/K/V and materialize native RoPE (useful as a control "
            "for validating direct-AdaLN specialization)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files owned by a previous specialization in output-dir",
    )
    return parser.parse_args()


def copy_safetensor_entry(
    source: SafeTensorFile,
    name: str,
    output,
) -> None:
    entry = source.entries[name]
    remaining = entry.size
    with source.path.open("rb") as input_file:
        input_file.seek(source.data_offset + entry.start)
        while remaining:
            chunk = input_file.read(min(remaining, COPY_BUFFER_SIZE))
            if not chunk:
                raise ValueError(f"unexpected EOF while copying {name!r} from {source.path}")
            output.write(chunk)
            remaining -= len(chunk)


def write_specialized_safetensors(
    path: Path,
    f32_tensors: dict[str, np.ndarray],
    concatenated_tensors: dict[str, tuple[str, ...]],
    source_tensor: Callable[[str], SafeTensorFile],
) -> tuple[str, int]:
    """Write small FP32 tensors and stream-concatenated source tensors atomically."""

    header: dict[str, object] = {
        "__metadata__": {
            "format": "fasth3-native-direct-adaln-v1",
            "training_indices": "999,749,500,250",
            "storage": "mixed",
        }
    }
    tensor_layouts: dict[str, tuple[str, list[int], int]] = {}
    for name, array in f32_tensors.items():
        array = np.ascontiguousarray(array, dtype="<f4")
        f32_tensors[name] = array
        tensor_layouts[name] = ("F32", list(array.shape), array.nbytes)

    for name, source_names in concatenated_tensors.items():
        if not source_names:
            raise ValueError(f"concatenated tensor {name!r} has no source tensors")
        entries = [source_tensor(source_name).entries[source_name] for source_name in source_names]
        dtype = entries[0].dtype
        if dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"unsupported concatenated dtype {dtype!r} for {name!r}")
        if any(entry.dtype != dtype for entry in entries):
            raise ValueError(f"mixed source dtypes for concatenated tensor {name!r}")
        if any(len(entry.shape) != 2 for entry in entries):
            raise ValueError(f"concatenated tensor {name!r} must contain matrices")
        width = entries[0].shape[1]
        if any(entry.shape[1] != width for entry in entries):
            raise ValueError(f"source width mismatch for concatenated tensor {name!r}")
        shape = [sum(entry.shape[0] for entry in entries), width]
        tensor_layouts[name] = (dtype, shape, sum(entry.size for entry in entries))

    offset = 0
    for name in sorted(tensor_layouts):
        dtype, shape, size = tensor_layouts[name]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for name in sorted(tensor_layouts):
            if name in f32_tensors:
                output.write(f32_tensors[name].tobytes(order="C"))
                continue
            for source_name in concatenated_tensors[name]:
                copy_safetensor_entry(source_tensor(source_name), source_name, output)
        output.flush()
        os.fsync(output.fileno())
    expected = 8 + len(encoded) + offset
    if temporary.stat().st_size != expected:
        raise ValueError(f"direct table shard size mismatch: {temporary.stat().st_size} != {expected}")
    os.replace(temporary, path)

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest(), offset


def collect_qkv_groups(index: dict[str, str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, dict[str, str]] = {}
    for key in index:
        match = QKV_RE.match(key)
        if match is None:
            continue
        prefix, projection = match.groups()
        groups.setdefault(prefix, {})[projection] = key

    expected_prefixes = {
        *(f"transformer_blocks.{block}" for block in range(50)),
        *(f"token_refiner.refiner_blocks.{block}" for block in range(2)),
    }
    missing_prefixes = sorted(expected_prefixes - set(groups))
    unexpected_prefixes = sorted(set(groups) - expected_prefixes)
    if missing_prefixes or unexpected_prefixes:
        raise ValueError(
            "unexpected H3 attention block set: "
            f"missing={missing_prefixes}, unexpected={unexpected_prefixes}"
        )

    result: dict[str, tuple[str, ...]] = {}
    for prefix in sorted(groups):
        projections = groups[prefix]
        if set(projections) != set(QKV_ORDER):
            raise ValueError(f"incomplete QKV group {prefix!r}: {sorted(projections)}")
        result[f"{prefix}.attn.qkv_proj.weight"] = tuple(
            projections[projection] for projection in QKV_ORDER
        )
    return result


def replace_owned_file(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {path}")
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"refusing to replace directory: {path}")
    path.unlink()


def link_source_shard(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    replace_owned_file(destination, overwrite)
    relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative_source)


def write_json_atomic(path: Path, value: object, overwrite: bool) -> None:
    replace_owned_file(path, overwrite)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source_dir = args.dense_transformer.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    source_index_path = source_dir / INDEX_NAME
    if not source_index_path.is_file():
        raise FileNotFoundError(source_index_path)

    index = load_index(source_dir)
    qkv_groups = collect_qkv_groups(index)
    gate_keys = sorted(key for key in index if VSA_GATE_RE.match(key))
    if gate_keys and not args.drop_vsa_gates:
        raise ValueError(
            f"source contains {len(gate_keys)} VSA gate tensors; pass "
            "--drop-vsa-gates for the dense sd.cpp runtime"
        )

    timesteps = schedule_union(args.video_shift, args.audio_shift, 4)
    video_sigmas = shifted_sigmas(args.video_shift, 4)
    shard_cache: dict[str, SafeTensorFile] = {}

    def source_file(key: str) -> SafeTensorFile:
        shard_name = index[key]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = SafeTensorFile(source_dir / shard_name)
        return shard_cache[shard_name]

    def source_tensor(key: str) -> np.ndarray:
        return source_file(key).tensor_f32(key)

    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    hidden_size = int(config["hidden_size"])
    attention_inner = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    for output_name, source_names in qkv_groups.items():
        entries = [source_file(name).entries[name] for name in source_names]
        expected_projection_shape = (attention_inner, hidden_size)
        if any(entry.shape != expected_projection_shape for entry in entries):
            raise ValueError(
                f"QKV shape mismatch for {output_name!r}: "
                f"got={[entry.shape for entry in entries]}, "
                f"expected={expected_projection_shape}"
            )
        if any(entry.dtype != entries[0].dtype for entry in entries):
            raise ValueError(f"QKV dtype mismatch for {output_name!r}")
        source_prefix = output_name.removesuffix(".attn.qkv_proj.weight")
        out_name = f"{source_prefix}.attn.to_out.0.weight"
        if out_name not in index:
            raise KeyError(f"source checkpoint is missing {out_name!r}")
        out_entry = source_file(out_name).entries[out_name]
        if out_entry.shape != (hidden_size, attention_inner):
            raise ValueError(
                f"attention output shape mismatch for {out_name!r}: "
                f"got={out_entry.shape}, expected={(hidden_size, attention_inner)}"
            )

    generated_f32: dict[str, np.ndarray] = {}
    table_stats: dict[str, dict[str, float]] = {}
    expected_adaln: set[str] = set(FINAL_ADALN_KEYS)
    for block in range(50):
        expected_adaln.update(
            {
                f"transformer_blocks.{block}.adaln_proj.linear.weight",
                f"transformer_blocks.{block}.adaln_proj.linear.bias",
            }
        )
    missing = sorted(expected_adaln - set(index))
    if missing:
        raise KeyError(f"source checkpoint is missing AdaLN tensors: {missing}")

    if not args.keep_full_adaln:
        adaln_input = compute_adaln_input(source_dir, index, timesteps)
        for block in range(50):
            source_prefix = f"transformer_blocks.{block}.adaln_proj.linear"
            output_name = f"transformer_blocks.{block}.adaln_schedule.weight"
            table, stats = direct_adaln_table(
                adaln_input,
                source_tensor(source_prefix + ".weight"),
                source_tensor(source_prefix + ".bias"),
            )
            generated_f32[output_name] = table
            table_stats[output_name] = stats
            print(
                f"[{block + 1:02d}/51] {output_name}: shape={table.shape} "
                f"rms={stats['rms']:.4e} abs_max={stats['abs_max']:.4e}"
            )

        final_table, final_stats = direct_adaln_table(
            adaln_input,
            source_tensor("norm_out.linear.weight"),
            source_tensor("norm_out.linear.bias"),
        )
        generated_f32["norm_out.adaln_schedule.weight"] = final_table
        table_stats["norm_out.adaln_schedule.weight"] = final_stats
        print(
            f"[51/51] norm_out.adaln_schedule.weight: shape={final_table.shape} "
            f"rms={final_stats['rms']:.4e} abs_max={final_stats['abs_max']:.4e}"
        )

    rope_dimension = int(config.get("rope_freq_dim", 16))
    rope_theta = float(config.get("rope_theta", 10000.0))
    generated_f32[ROPE_INV_FREQ] = np.power(
        np.float32(rope_theta),
        -np.arange(rope_dimension, dtype=np.float32) / np.float32(rope_dimension),
    ).astype(np.float32)

    excluded: set[str] = set()
    if not args.keep_full_adaln:
        excluded.update(TIME_EMBEDDER_KEYS)
        excluded.update(expected_adaln)
    excluded.update(name for names in qkv_groups.values() for name in names)
    if args.drop_vsa_gates:
        excluded.update(gate_keys)
    retained = {key: shard for key, shard in index.items() if key not in excluded}
    generated_names = set(generated_f32) | set(qkv_groups)
    collisions = sorted(set(retained) & generated_names)
    if collisions:
        raise ValueError(f"generated tensor names collide with source: {collisions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specialization_format = (
        "fasth3-native-qkv-v1" if args.keep_full_adaln else "fasth3-native-direct-adaln-v1"
    )
    generated_shard = NATIVE_QKV_SHARD if args.keep_full_adaln else DIRECT_SHARD
    direct_path = args.output_dir / generated_shard
    replace_owned_file(direct_path, args.overwrite)
    digest, direct_bytes = write_specialized_safetensors(
        direct_path,
        generated_f32,
        qkv_groups,
        source_file,
    )

    used_shards = sorted(set(retained.values()))
    for shard_name in used_shards:
        source_shard = source_dir / shard_name
        if not source_shard.is_file():
            raise FileNotFoundError(source_shard)
        link_source_shard(source_shard, args.output_dir / shard_name, args.overwrite)

    output_weight_map = dict(retained)
    output_weight_map.update({name: generated_shard for name in generated_f32})
    output_weight_map.update({name: generated_shard for name in qkv_groups})
    source_bytes = 0
    for key, shard_name in retained.items():
        if shard_name not in shard_cache:
            shard_cache[shard_name] = SafeTensorFile(source_dir / shard_name)
        source_bytes += shard_cache[shard_name].entries[key].size
    output_index = {
        "metadata": {
            "total_size": source_bytes + direct_bytes,
            "fasth3_specialization": specialization_format,
            "training_indices": "999,749,500,250",
            "video_shift": args.video_shift,
            "audio_shift": args.audio_shift,
            "dropped_vsa_gate_tensors": len(gate_keys) if args.drop_vsa_gates else 0,
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    write_json_atomic(args.output_dir / INDEX_NAME, output_index, args.overwrite)

    config_destination = args.output_dir / "config.json"
    replace_owned_file(config_destination, args.overwrite)
    shutil.copy2(config_path, config_destination)

    report = {
        "format": specialization_format,
        "source_transformer": str(source_dir),
        "training_indices": [int(value) for value in FASTH3_TRAINING_INDICES],
        "video_shift": args.video_shift,
        "audio_shift": args.audio_shift,
        "video_sigma_grid": [float(value) for value in video_sigmas] + [0.0],
        "adaln_schedule_union": [float(value) for value in timesteps],
        "retained_tensors": len(retained),
        "generated_tensors": len(generated_names),
        "fused_qkv_tensors": len(qkv_groups),
        "excluded_split_qkv_tensors": sum(len(names) for names in qkv_groups.values()),
        "excluded_full_adaln_tensors": 0 if args.keep_full_adaln else len(expected_adaln),
        "excluded_time_embedder_tensors": 0 if args.keep_full_adaln else len(TIME_EMBEDDER_KEYS),
        "dropped_vsa_gate_tensors": len(gate_keys) if args.drop_vsa_gates else 0,
        "indexed_tensor_bytes": source_bytes + direct_bytes,
        "direct_shard_sha256": digest,
        "direct_tables": table_stats,
    }
    write_json_atomic(args.output_dir / SPECIALIZATION_REPORT, report, args.overwrite)
    print(
        f"wrote {args.output_dir}: retained={len(retained)}, "
        f"generated={len(generated_names)}, linked_shards={len(used_shards)}, "
        f"indexed={((source_bytes + direct_bytes) / 1e9):.3f} GB"
    )
    print(f"direct shard sha256 {digest}")


if __name__ == "__main__":
    main()
