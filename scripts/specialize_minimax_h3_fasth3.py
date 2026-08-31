#!/usr/bin/env python3
"""Specialize a FastH3 adapter for a rank-pruned MiniMax-H3 GGUF.

FastH3 is trained against the full 2,688-wide H3 timestep/AdaLN path.  Public
``pruned`` H3 GGUFs replace that path with a shared eight-value curve and
eight-wide per-block projections.  Loading the FastH3 AdaLN LoRAs directly is
therefore shape-incompatible and silently drops part of the distilled model.

This tool consumes the full dense FastH3 transformer only as an offline
teacher.  At the published 999/749/500/250 training schedule it evaluates
every full AdaLN projection and stores those outputs directly.  The result is
a normal sd.cpp safetensors adapter containing:

* the original compatible attention, MLP, refiner, and boundary updates;
* 50 FP32 block modulation tables and one FP32 final-layer table;
* no incompatible full-rank timestep or AdaLN tensors.

Pascal GPUs cannot execute BF16 LoRA matmuls.  ``--retained-bf16 f16``
converts the compatible BF16 adapter tensors to FP16 while writing the
specialized artifact, avoiding an unsupported CUDA conversion at runtime.

The direct tables bypass both the ill-conditioned 1,025-point compact curve
fit and the per-block AdaLN GEMMs.  They require the exact trained sigma grid;
the matching runtime rejects other schedules instead of silently degrading.

The dense checkpoint is not copied to the output and is not needed at runtime.

Example::

    PYTHONPATH=/path/to/llama.cpp/gguf-py python3 \
      scripts/specialize_minimax_h3_fasth3.py \
      --dense-transformer /path/to/FastH3/transformer \
      --base-gguf MiniMax-H3-FL2VA-pruned-Q4_K_M.gguf \
      --adapter dense-datafree/adapter_model.safetensors \
      --output fasth3-dense-datafree-pruned-r8.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

try:
    from gguf import GGUFReader
except ImportError as error:
    raise SystemExit(
        "gguf-py is required; add llama.cpp/gguf-py to PYTHONPATH"
    ) from error


INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
TIME_EMBEDDER_KEYS = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
)
MAIN_ADALN_RE = re.compile(
    r"^transformer_blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)$"
)
PRUNED_ADALN_RE = re.compile(r"^blocks\.\d+\.adaln_proj\.linear\.weight$")
COPY_BUFFER_SIZE = 16 * 1024 * 1024
MAX_HEADER_SIZE = 256 * 1024 * 1024
FASTH3_TRAINING_INDICES = np.asarray([999.0, 749.0, 500.0, 250.0], dtype=np.float64)


@dataclass(frozen=True)
class SafeTensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class SafeTensorFile:
    """Header parser and mmap reader for the dtypes used by H3."""

    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_offset = self._read_header(path)
        self.entries = {
            name: self._entry(name, value)
            for name, value in self.header.items()
            if name != "__metadata__"
        }

    @staticmethod
    def _read_header(path: Path) -> tuple[dict, int]:
        file_size = path.stat().st_size
        with path.open("rb") as file:
            raw_size = file.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated safetensors header size: {path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            if not 0 < header_size <= MAX_HEADER_SIZE:
                raise ValueError(f"invalid safetensors header size {header_size}: {path}")
            raw_header = file.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header is not an object: {path}")
        data_offset = 8 + header_size
        if data_offset > file_size:
            raise ValueError(f"safetensors data begins past EOF: {path}")
        return header, data_offset

    def _entry(self, name: str, value: object) -> SafeTensorEntry:
        if not isinstance(value, dict):
            raise ValueError(f"invalid tensor entry {name!r} in {self.path}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError(f"invalid dtype/shape for {name!r} in {self.path}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid offsets for {name!r} in {self.path}")
        entry = SafeTensorEntry(
            name=name,
            dtype=dtype,
            shape=tuple(int(item) for item in shape),
            start=int(offsets[0]),
            end=int(offsets[1]),
        )
        item_sizes = {"F32": 4, "F16": 2, "BF16": 2}
        if dtype not in item_sizes:
            raise ValueError(f"unsupported dtype {dtype!r} for {name!r}")
        expected = math.prod(entry.shape) * item_sizes[dtype]
        if entry.size != expected:
            raise ValueError(
                f"size mismatch for {name!r}: header={entry.size}, expected={expected}"
            )
        if entry.start < 0 or entry.end < entry.start:
            raise ValueError(f"invalid data range for {name!r}")
        if self.data_offset + entry.end > self.path.stat().st_size:
            raise ValueError(f"tensor {name!r} extends past EOF in {self.path}")
        return entry

    def tensor_f32(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        offset = self.data_offset + entry.start
        if entry.dtype == "F32":
            source = np.memmap(
                self.path, mode="r", dtype="<f4", offset=offset, shape=entry.shape
            )
            return np.asarray(source, dtype=np.float32)
        if entry.dtype == "F16":
            source = np.memmap(
                self.path, mode="r", dtype="<f2", offset=offset, shape=entry.shape
            )
            return np.asarray(source, dtype=np.float32)
        source = np.memmap(
            self.path, mode="r", dtype="<u2", offset=offset, shape=entry.shape
        )
        bits = np.asarray(source, dtype=np.uint32) << np.uint32(16)
        return bits.view(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dense-transformer", required=True, type=Path)
    parser.add_argument("--base-gguf", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--denoise-steps",
        type=int,
        default=4,
        help="DiT forwards; FastH3 Preview v1 is trained for exactly four",
    )
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output"
    )
    parser.add_argument(
        "--retained-bf16",
        choices=("keep", "f16"),
        default="keep",
        help=(
            "storage for compatible BF16 adapter tensors; use f16 on Pascal "
            "GPUs, which cannot execute BF16 LoRA matmuls"
        ),
    )
    return parser.parse_args()


def shifted_sigmas(shift: float, denoise_steps: int) -> np.ndarray:
    if denoise_steps != len(FASTH3_TRAINING_INDICES):
        raise ValueError("FastH3 Preview v1 requires four denoise forwards")
    base = FASTH3_TRAINING_INDICES / np.float64(1000.0)
    return (shift * base / (1.0 + (shift - 1.0) * base)).astype(np.float32)


def schedule_union(video_shift: float, audio_shift: float, denoise_steps: int) -> np.ndarray:
    video = np.float32(1.0) - shifted_sigmas(video_shift, denoise_steps)
    audio = np.float32(1.0) - shifted_sigmas(audio_shift, denoise_steps)
    return np.unique(np.concatenate((video, audio)))


def timestep_embedding(timesteps: np.ndarray, dim: int = 256) -> np.ndarray:
    half = dim // 2
    exponent = -math.log(10000.0) * np.arange(half, dtype=np.float32) / np.float32(half)
    angles = timesteps.astype(np.float32)[:, None] * np.exp(exponent)[None, :]
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)


def silu(value: np.ndarray) -> np.ndarray:
    return value / (np.float32(1.0) + np.exp(-value))


def round_bf16(value: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 (RNE), returning the rounded values as FP32."""
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def load_index(transformer: Path) -> dict[str, str]:
    raw = json.loads((transformer / INDEX_NAME).read_text())
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"missing weight_map in {transformer / INDEX_NAME}")
    return {str(key): str(value) for key, value in weight_map.items()}


def load_dense_keys(
    transformer: Path, index: dict[str, str], keys: Iterator[str] | tuple[str, ...]
) -> dict[str, np.ndarray]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index[key], []).append(key)
    result: dict[str, np.ndarray] = {}
    for shard_name, shard_keys in by_shard.items():
        shard = SafeTensorFile(transformer / shard_name)
        for key in shard_keys:
            result[key] = np.array(shard.tensor_f32(key), copy=True)
    return result


def compute_adaln_input(
    transformer: Path, index: dict[str, str], timesteps: np.ndarray
) -> np.ndarray:
    weights = load_dense_keys(transformer, index, TIME_EMBEDDER_KEYS)
    hidden = timestep_embedding(timesteps)
    hidden = hidden @ weights["time_embedder.linear_1.weight"].T
    hidden += weights["time_embedder.linear_1.bias"]
    hidden = silu(hidden)
    temb = hidden @ weights["time_embedder.linear_2.weight"].T
    temb += weights["time_embedder.linear_2.bias"]
    # Full H3 applies SiLU in FP32, then casts to the BF16 AdaLN weight dtype.
    return round_bf16(silu(temb))


def validate_pruned_base(base_gguf: Path) -> None:
    reader = GGUFReader(str(base_gguf), "r")
    table_shape: tuple[int, ...] | None = None
    block_weights = 0
    for tensor in reader.tensors:
        if tensor.name == "adaln_t_table":
            table_shape = tuple(int(value) for value in tensor.data.shape)
        elif PRUNED_ADALN_RE.match(tensor.name):
            block_weights += 1
    if table_shape is None or min(table_shape) != 8 or max(table_shape) != 1025:
        raise ValueError(f"expected a 1025x8 rank-pruned AdaLN table, got {table_shape}")
    if block_weights != 50:
        raise ValueError(f"expected 50 compact AdaLN block weights, got {block_weights}")


def direct_adaln_table(
    adaln_input: np.ndarray,
    dense_weight: np.ndarray,
    dense_bias: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if dense_weight.ndim != 2 or dense_bias.shape != (dense_weight.shape[0],):
        raise ValueError(
            f"invalid dense AdaLN shapes weight={dense_weight.shape}, bias={dense_bias.shape}"
        )
    if adaln_input.shape[1] != dense_weight.shape[1]:
        raise ValueError(
            f"AdaLN input/weight mismatch: input={adaln_input.shape}, weight={dense_weight.shape}"
        )
    table = adaln_input @ dense_weight.T
    table += dense_bias
    table = np.ascontiguousarray(table, dtype=np.float32)
    if not np.isfinite(table).all():
        raise ValueError("non-finite values in direct AdaLN table")
    stats = {
        "abs_max": float(np.max(np.abs(table))),
        "rms": float(np.sqrt(np.mean(table.astype(np.float64) ** 2))),
    }
    return table, stats


def excluded_adapter_tensor(name: str) -> bool:
    if name.startswith("time_embedder."):
        return True
    if name in {"norm_out.linear.diff", "norm_out.linear.diff_b"}:
        return True
    if re.match(
        r"^transformer_blocks\.\d+\.adaln_proj\.linear\."
        r"(?:lora_A\.weight|lora_B\.weight|diff_b)$",
        name,
    ):
        return True
    return False


def copy_exact(source: BinaryIO, destination: BinaryIO, size: int) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, COPY_BUFFER_SIZE))
        if not chunk:
            raise ValueError("unexpected EOF while copying adapter tensor")
        destination.write(chunk)
        remaining -= len(chunk)


def copy_bf16_as_f16(source: BinaryIO, destination: BinaryIO, size: int) -> None:
    if size % 2:
        raise ValueError(f"BF16 tensor has odd byte size: {size}")
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, COPY_BUFFER_SIZE))
        if not chunk:
            raise ValueError("unexpected EOF while converting BF16 adapter tensor")
        bf16 = np.frombuffer(chunk, dtype="<u2")
        fp32 = (bf16.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if not np.isfinite(fp32).all():
            raise ValueError("non-finite BF16 value in retained adapter tensor")
        fp16 = fp32.astype("<f2")
        if not np.isfinite(fp16).all():
            raise ValueError("BF16 adapter value overflows FP16 storage")
        destination.write(fp16.tobytes())
        remaining -= len(chunk)


def write_adapter(
    source_path: Path,
    output_path: Path,
    generated: dict[str, np.ndarray],
    metadata_updates: dict[str, str],
    overwrite: bool,
    retained_bf16: str,
) -> tuple[int, int, str]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")
    source = SafeTensorFile(source_path)
    retained = {
        name: entry
        for name, entry in source.entries.items()
        if not excluded_adapter_tensor(name)
    }
    duplicates = sorted(retained.keys() & generated.keys())
    if duplicates:
        raise ValueError(f"generated tensor names collide with source: {duplicates}")

    metadata = dict(source.header.get("__metadata__") or {})
    metadata.update(metadata_updates)
    output_header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for name, entry in retained.items():
        output_dtype = "F16" if retained_bf16 == "f16" and entry.dtype == "BF16" else entry.dtype
        output_header[name] = {
            "dtype": output_dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.size],
        }
        offset += entry.size
    for name, array in generated.items():
        array = np.ascontiguousarray(array, dtype="<f4")
        generated[name] = array
        output_header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes

    encoded = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source_file, temporary.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for entry in retained.values():
            source_file.seek(source.data_offset + entry.start)
            if retained_bf16 == "f16" and entry.dtype == "BF16":
                copy_bf16_as_f16(source_file, output, entry.size)
            else:
                copy_exact(source_file, output, entry.size)
        for array in generated.values():
            output.write(array.tobytes(order="C"))
        output.flush()
        os.fsync(output.fileno())
    expected_size = 8 + len(encoded) + offset
    if temporary.stat().st_size != expected_size:
        raise ValueError(
            f"output size mismatch: {temporary.stat().st_size} != {expected_size}"
        )
    os.replace(temporary, output_path)

    digest = hashlib.sha256()
    with output_path.open("rb") as file:
        while chunk := file.read(COPY_BUFFER_SIZE):
            digest.update(chunk)
    return len(retained), len(generated), digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.denoise_steps != 4:
        raise ValueError("FastH3 Preview v1 is trained for exactly four DiT forwards")
    for path in (args.dense_transformer, args.base_gguf, args.adapter):
        if not path.exists():
            raise FileNotFoundError(path)

    index = load_index(args.dense_transformer)
    timesteps = schedule_union(args.video_shift, args.audio_shift, args.denoise_steps)
    video_sigmas = shifted_sigmas(args.video_shift, args.denoise_steps)
    print("training indices: 999, 749, 500, 250")
    print(
        "video sigma grid:",
        ", ".join(f"{value:.9f}" for value in video_sigmas),
        ", 0.000000000",
    )
    print("AdaLN schedule union:", ", ".join(f"{value:.9f}" for value in timesteps))
    adaln_input = compute_adaln_input(args.dense_transformer, index, timesteps)
    validate_pruned_base(args.base_gguf)

    generated: dict[str, np.ndarray] = {}
    all_stats: list[dict[str, float]] = []
    dense_adaln_keys = sorted(
        key
        for key in index
        if MAIN_ADALN_RE.match(key) or key in {"norm_out.linear.weight", "norm_out.linear.bias"}
    )
    shard_cache: dict[str, SafeTensorFile] = {}

    def dense_tensor(key: str) -> np.ndarray:
        shard_name = index[key]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = SafeTensorFile(args.dense_transformer / shard_name)
        return shard_cache[shard_name].tensor_f32(key)

    pairs: list[tuple[str, str]] = []
    for block in range(50):
        dense_prefix = f"transformer_blocks.{block}.adaln_proj.linear"
        pairs.append((dense_prefix, f"diffusion_model.blocks.{block}.adaln_schedule"))
    pairs.append(
        (
            "norm_out.linear",
            "diffusion_model.final_layer.adaln_schedule",
        )
    )
    expected_keys = {
        f"{dense_prefix}.{kind}"
        for dense_prefix, _ in pairs
        for kind in ("weight", "bias")
    }
    missing = sorted(expected_keys - set(dense_adaln_keys))
    if missing:
        raise KeyError(f"dense checkpoint is missing AdaLN tensors: {missing}")

    for pair_index, (dense_prefix, output_prefix) in enumerate(pairs, start=1):
        table, stats = direct_adaln_table(
            adaln_input,
            dense_tensor(f"{dense_prefix}.weight"),
            dense_tensor(f"{dense_prefix}.bias"),
        )
        generated[f"{output_prefix}.diff"] = table
        all_stats.append(stats)
        print(
            f"[{pair_index:02d}/{len(pairs)}] {dense_prefix}: "
            f"shape={table.shape} rms={stats['rms']:.4e} "
            f"abs_max={stats['abs_max']:.4e}"
        )

    worst_value = max(item["abs_max"] for item in all_stats)
    print(f"direct tables: count={len(generated)}, worst_abs={worst_value:.4e}")

    retained, added, digest = write_adapter(
        args.adapter,
        args.output,
        generated,
        {
            "format": "fastvideo-lora-v2-sdcpp-direct-adaln-v1",
            "source_adapter": args.adapter.name,
            "dense_teacher": args.dense_transformer.name,
            "base_gguf": args.base_gguf.name,
            "transformer_forwards": str(args.denoise_steps),
            "video_shift": f"{args.video_shift:g}",
            "audio_shift": f"{args.audio_shift:g}",
            "training_indices": "999,749,500,250",
            "video_sigma_grid": ",".join(
                [*(f"{value:.9g}" for value in video_sigmas), "0"]
            ),
            "adaln_schedule": ",".join(f"{value:.9g}" for value in timesteps),
            "adaln_storage": "direct-fp32",
            "retained_bf16_storage": args.retained_bf16,
            "adaln_worst_abs_value": f"{worst_value:.9g}",
        },
        args.overwrite,
        args.retained_bf16,
    )
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1e9:.3f} GB, "
        f"retained={retained}, generated={added})"
    )
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
